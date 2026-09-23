//! ModernBERT encoder followed by the Laya decision head, which scores every
//! answer option and predicts whether to act.
const Model = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const kernels = @import("kernels.zig");
const SafeTensors = @import("SafeTensors.zig");

hidden_size: usize,
intermediate_size: usize,
vocab_size: usize,
max_positions: usize,
heads: usize,
head_heads: usize,
/// Positions visible on each side of a token in sliding-window layers.
local_window: usize,
global_rope_theta: f32,
local_rope_theta: f32,
embeddings: []const f32,
embedding_norm: Norm,
final_norm: Norm,
layers: []EncoderLayer,
head: []HeadLayer,
type_embeddings: []const f32,
scorer_norm: Norm,
scorer_in: Linear,
scorer_out: Linear,
act_in: Linear,
act_out: Linear,

/// Question kinds, in the order of the type embedding rows.
pub const QuestionType = enum { choice, score, noul };

pub const Output = struct {
    /// One logit per option.
    logits: []f32,
    act_logits: []f32,

    pub fn deinit(output: Output, gpa: Allocator) void {
        gpa.free(output.logits);
        gpa.free(output.act_logits);
    }
};

const head_norm_eps = 1e-5;
const act_hidden_size = 256;

const EncoderConfig = struct {
    model_type: enum { modernbert } = .modernbert,
    hidden_activation: enum { gelu } = .gelu,
    attention_bias: bool = false,
    mlp_bias: bool = false,
    norm_bias: bool = false,
    layer_types: ?[]const enum { full_attention, sliding_attention } = null,
    rope_parameters: ?std.json.Value = null,
    hidden_size: usize = 1024,
    num_attention_heads: usize = 16,
    num_hidden_layers: usize = 28,
    intermediate_size: usize = 2624,
    vocab_size: usize = 50368,
    global_attn_every_n_layers: usize = 3,
    local_attention: usize = 128,
    max_position_embeddings: usize = 8192,
    norm_eps: f32 = 1e-5,
    global_rope_theta: f32 = 160000,
    local_rope_theta: f32 = 10000,

    /// Takes the RoPE bases from `rope_parameters` when it is present.
    fn resolveRope(config: *EncoderConfig) !void {
        const rope = config.rope_parameters orelse return;
        if (rope != .object) return error.InvalidConfiguration;
        if (rope.object.get("rope_theta") != null) {
            config.global_rope_theta = try ropeTheta(rope);
            config.local_rope_theta = config.global_rope_theta;
        } else {
            const full = rope.object.get("full_attention") orelse return error.InvalidConfiguration;
            config.global_rope_theta = try ropeTheta(full);
            const sliding = rope.object.get("sliding_attention") orelse return error.InvalidConfiguration;
            config.local_rope_theta = try ropeTheta(sliding);
        }
    }

    fn validate(config: EncoderConfig) !void {
        if (config.attention_bias or config.mlp_bias or config.norm_bias) return error.UnsupportedConfiguration;
        if (config.hidden_size > 8192 or config.intermediate_size > 65536 or
            config.vocab_size > 1 << 20 or config.num_hidden_layers > 256)
        {
            return error.UnsupportedConfiguration;
        }

        if (config.hidden_size == 0 or config.num_attention_heads == 0 or config.num_hidden_layers == 0 or
            config.intermediate_size == 0 or config.vocab_size == 0 or config.global_attn_every_n_layers == 0)
        {
            return error.InvalidConfiguration;
        }
        if (config.max_position_embeddings == 0 or config.max_position_embeddings > 8192) return error.InvalidConfiguration;
        if (config.hidden_size % config.num_attention_heads != 0) return error.InvalidConfiguration;
        if ((config.hidden_size / config.num_attention_heads) % 2 != 0) return error.InvalidConfiguration;
        for ([_]f32{ config.norm_eps, config.global_rope_theta, config.local_rope_theta }) |value| {
            if (!std.math.isFinite(value) or value <= 0) return error.InvalidConfiguration;
        }
        if (config.layer_types) |layer_types| {
            if (layer_types.len != config.num_hidden_layers) return error.InvalidConfiguration;
        }
    }
};

fn ropeTheta(value: std.json.Value) !f32 {
    if (value != .object) return error.InvalidConfiguration;
    if (value.object.get("rope_type")) |rope_type| {
        if (rope_type != .string or !std.mem.eql(u8, rope_type.string, "default")) return error.UnsupportedConfiguration;
    }
    const theta = value.object.get("rope_theta") orelse return error.InvalidConfiguration;
    return switch (theta) {
        .float => |float| @floatCast(float),
        .integer => |integer| @floatFromInt(integer),
        else => error.InvalidConfiguration,
    };
}

/// Rejects float32 matrices that do not fit in the address space.
fn checkMatrixSize(rows: usize, cols: usize) !void {
    const elements = std.math.mul(usize, rows, cols) catch return error.UnsupportedConfiguration;
    _ = std.math.mul(usize, elements, @sizeOf(f32)) catch return error.UnsupportedConfiguration;
}

const Linear = struct {
    weight: []const f32,
    bias: ?[]const f32,
    inputs: usize,
    outputs: usize,

    fn run(linear: Linear, x: []const f32, out: []f32) void {
        kernels.linear(x, linear.weight, linear.bias, out, linear.inputs, linear.outputs);
    }
};

const Norm = struct {
    weight: []const f32,
    bias: ?[]const f32,
    eps: f32,

    fn run(norm: Norm, x: []const f32, out: []f32) void {
        kernels.layerNorm(x, norm.weight, norm.bias, out, norm.eps);
    }
};

const EncoderLayer = struct {
    /// Global layers attend to the whole sequence, and the others use a sliding window.
    global: bool,
    attention_norm: ?Norm,
    mlp_norm: Norm,
    qkv: Linear,
    attention_out: Linear,
    mlp_in: Linear,
    mlp_out: Linear,
};

const HeadLayer = struct {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    attention_out: Linear,
    linear1: Linear,
    linear2: Linear,
};

/// Converts the weights into `weights` and allocates everything else in `arena`.
/// Nothing is freed on failure, so both allocators should be discarded on error.
pub fn init(
    arena: Allocator,
    weights: Allocator,
    tensors: *const SafeTensors,
    encoder_json: []const u8,
    head_layers: usize,
    actions: usize,
) !Model {
    // Negative or oversized integers fail with Overflow.
    var config = std.json.parseFromSliceLeaky(EncoderConfig, arena, encoder_json, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
        error.Overflow, error.InvalidEnumTag => return error.UnsupportedConfiguration,
        else => |e| return e,
    };
    try config.resolveRope();
    try config.validate();
    if (head_layers > 256 or actions > 256) return error.UnsupportedConfiguration;

    const d = config.hidden_size;
    const f = config.intermediate_size;
    // Upstream gives the decision head one attention head per 64 dimensions.
    const head_heads = @max(1, d / 64);
    if (d % head_heads != 0) return error.UnsupportedConfiguration;
    try checkMatrixSize(config.vocab_size, d);
    try checkMatrixSize(2 * f, d);
    try checkMatrixSize(config.max_position_embeddings, @max(2 * f, 4 * d));

    const loader: Loader = .{ .arena = arena, .weights = weights, .tensors = tensors };
    const layers = try arena.alloc(EncoderLayer, config.num_hidden_layers);
    for (layers, 0..) |*layer, i| {
        const scope = try loader.scope(try std.fmt.allocPrint(arena, "encoder.layers.{d}", .{i}));
        const global = if (config.layer_types) |layer_types|
            layer_types[i] == .full_attention
        else
            i % config.global_attn_every_n_layers == 0;
        layer.* = .{
            .global = global,
            .attention_norm = if (i == 0) null else try scope.norm("attn_norm", d, false, config.norm_eps),
            .mlp_norm = try scope.norm("mlp_norm", d, false, config.norm_eps),
            .qkv = try scope.linear("attn.Wqkv", d, 3 * d, false),
            .attention_out = try scope.linear("attn.Wo", d, d, false),
            .mlp_in = try scope.linear("mlp.Wi", d, 2 * f, false),
            .mlp_out = try scope.linear("mlp.Wo", f, d, false),
        };
    }

    const head = try arena.alloc(HeadLayer, head_layers);
    for (head, 0..) |*layer, i| {
        const scope = try loader.scope(try std.fmt.allocPrint(arena, "head.layers.{d}", .{i}));
        layer.* = .{
            .norm1 = try scope.norm("norm1", d, true, head_norm_eps),
            .norm2 = try scope.norm("norm2", d, true, head_norm_eps),
            .qkv = .{
                .weight = try scope.weight("self_attn.in_proj_weight", &.{ 3 * d, d }),
                .bias = try scope.weight("self_attn.in_proj_bias", &.{3 * d}),
                .inputs = d,
                .outputs = 3 * d,
            },
            .attention_out = try scope.linear("self_attn.out_proj", d, d, true),
            .linear1 = try scope.linear("linear1", d, 4 * d, true),
            .linear2 = try scope.linear("linear2", 4 * d, d, true),
        };
    }

    return .{
        .hidden_size = d,
        .intermediate_size = f,
        .vocab_size = config.vocab_size,
        .max_positions = config.max_position_embeddings,
        .heads = config.num_attention_heads,
        .head_heads = head_heads,
        .local_window = config.local_attention / 2,
        .global_rope_theta = config.global_rope_theta,
        .local_rope_theta = config.local_rope_theta,
        .embeddings = try loader.weight("encoder.embeddings.tok_embeddings.weight", &.{ config.vocab_size, d }),
        .embedding_norm = try loader.norm("encoder.embeddings.norm", d, false, config.norm_eps),
        .final_norm = try loader.norm("encoder.final_norm", d, false, config.norm_eps),
        .layers = layers,
        .head = head,
        .type_embeddings = try loader.weight("type_emb.weight", &.{ std.enums.values(QuestionType).len, d }),
        .scorer_norm = try loader.norm("scorer.0", d, true, head_norm_eps),
        .scorer_in = try loader.linear("scorer.1", d, d, true),
        .scorer_out = try loader.linear("scorer.3", d, 1, true),
        .act_in = try loader.linear("act_head.0", d + 4, act_hidden_size, true),
        .act_out = try loader.linear("act_head.2", act_hidden_size, actions, true),
    };
}

/// Returns one logit per option marker, and the action logits.
pub fn forward(model: *const Model, gpa: Allocator, ids: []const u32, markers: []const usize, qtype: QuestionType) !Output {
    const d = model.hidden_size;
    const f = model.intermediate_size;
    const n = ids.len;
    if (n == 0 or n > model.max_positions or markers.len == 0) return error.InvalidInput;
    for (ids) |id| if (id >= model.vocab_size) return error.InvalidToken;
    for (markers) |pos| if (pos >= n) return error.InvalidMarker;

    var arena_instance: std.heap.ArenaAllocator = .init(gpa);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();
    const x = try arena.alloc(f32, n * d);
    const y = try arena.alloc(f32, n * d);
    const z = try arena.alloc(f32, n * d);
    const qkv = try arena.alloc(f32, n * 3 * d);
    const ff = try arena.alloc(f32, n * @max(2 * f, 4 * d));
    const gate = try arena.alloc(f32, n * f);
    const scores = try arena.alloc(f32, n * n);
    const global_rope: kernels.Rope = try .init(arena, n, d / model.heads, model.global_rope_theta);
    const local_rope: kernels.Rope = try .init(arena, n, d / model.heads, model.local_rope_theta);

    for (ids, 0..) |id, i| @memcpy(y[i * d ..][0..d], model.embeddings[id * d ..][0..d]);
    model.embedding_norm.run(y, x);

    for (model.layers) |layer| {
        const normed = if (layer.attention_norm) |norm| blk: {
            norm.run(x, y);
            break :blk y;
        } else x;
        layer.qkv.run(normed, qkv);
        const window = if (layer.global) null else model.local_window;
        const rope = if (layer.global) global_rope else local_rope;
        kernels.attention(qkv, z, scores, n, d, model.heads, window, rope);
        layer.attention_out.run(z, y);
        add(x, y);

        layer.mlp_norm.run(x, y);
        layer.mlp_in.run(y, ff[0 .. n * 2 * f]);
        for (0..n) |row| for (0..f) |j| {
            gate[row * f + j] = kernels.gelu(ff[row * 2 * f + j]) * ff[row * 2 * f + f + j];
        };
        layer.mlp_out.run(gate, y);
        add(x, y);
    }
    model.final_norm.run(x, y);

    const type_embedding = model.type_embeddings[@backingInt(qtype) * d ..][0..d];
    for (x, y, 0..) |*value, encoded, i| value.* = encoded + type_embedding[i % d];

    for (model.head) |layer| {
        layer.norm1.run(x, y);
        layer.qkv.run(y, qkv);
        kernels.attention(qkv, z, scores, n, d, model.head_heads, null, null);
        layer.attention_out.run(z, y);
        add(x, y);

        layer.norm2.run(x, y);
        const hidden = ff[0 .. n * 4 * d];
        layer.linear1.run(y, hidden);
        for (hidden) |*value| value.* = @max(0, value.*);
        layer.linear2.run(hidden, y);
        add(x, y);
    }

    const logits = try gpa.alloc(f32, markers.len);
    errdefer gpa.free(logits);
    for (markers, logits) |pos, *logit| {
        model.scorer_norm.run(x[pos * d ..][0..d], y[0..d]);
        model.scorer_in.run(y[0..d], z[0..d]);
        for (z[0..d]) |*value| value.* = kernels.gelu(value.*);
        model.scorer_out.run(z[0..d], logit[0..1]);
    }

    // The action head sees the first token and a summary of the option distribution.
    const probs = try arena.dupe(f32, logits);
    kernels.softmax(probs);
    var top1: f32 = 0;
    var top2: f32 = 0;
    var entropy: f32 = 0;
    for (probs) |p| {
        if (p > top1) {
            top2 = top1;
            top1 = p;
        } else if (p > top2) {
            top2 = p;
        }
        entropy -= p * @log(@max(p, 1e-9));
    }
    const count: f32 = @floatFromInt(@max(markers.len, 2));
    const features = try arena.alloc(f32, d + 4);
    @memcpy(features[0..d], x[0..d]);
    features[d..][0..4].* = .{ top1, top1 - top2, entropy / @log(count), count / 255.0 };

    var act_hidden: [act_hidden_size]f32 = undefined;
    model.act_in.run(features, &act_hidden);
    for (&act_hidden) |*value| value.* = kernels.gelu(value.*);
    const act_logits = try gpa.alloc(f32, model.act_out.outputs);
    model.act_out.run(&act_hidden, act_logits);

    return .{ .logits = logits, .act_logits = act_logits };
}

fn add(dst: []f32, src: []const f32) void {
    for (dst, src) |*a, b| a.* += b;
}

/// Converts named tensors to float32 after checking their shapes.
const Loader = struct {
    arena: Allocator,
    weights: Allocator,
    tensors: *const SafeTensors,
    /// Prepended to every tensor name.
    prefix: []const u8 = "",

    /// Returns a loader for the tensors under `name`.
    fn scope(loader: Loader, name: []const u8) !Loader {
        var scoped = loader;
        scoped.prefix = try std.mem.concat(loader.arena, u8, &.{ loader.prefix, name, "." });
        return scoped;
    }

    fn weight(loader: Loader, name: []const u8, shape: []const usize) ![]const f32 {
        const full_name = try std.mem.concat(loader.arena, u8, &.{ loader.prefix, name });
        const tensor = loader.tensors.get(full_name) orelse return error.MissingTensor;
        if (!std.mem.eql(usize, tensor.shape, shape)) return error.InvalidTensorShape;
        return loader.tensors.toF32(loader.weights, tensor);
    }

    fn norm(loader: Loader, name: []const u8, dim: usize, bias: bool, eps: f32) !Norm {
        const scoped = try loader.scope(name);
        return .{
            .weight = try scoped.weight("weight", &.{dim}),
            .bias = if (bias) try scoped.weight("bias", &.{dim}) else null,
            .eps = eps,
        };
    }

    fn linear(loader: Loader, name: []const u8, inputs: usize, outputs: usize, bias: bool) !Linear {
        const scoped = try loader.scope(name);
        return .{
            .weight = try scoped.weight("weight", &.{ outputs, inputs }),
            .bias = if (bias) try scoped.weight("bias", &.{outputs}) else null,
            .inputs = inputs,
            .outputs = outputs,
        };
    }
};

test "encoder and decision heads match PyTorch for every question type" {
    const gpa = testing.allocator;
    var tensors: SafeTensors = try .init(gpa, @embedFile("fixtures/tiny.safetensors"));
    defer tensors.deinit();
    var arena_instance: std.heap.ArenaAllocator = .init(gpa);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();
    const model: Model = try .init(arena, arena, &tensors, @embedFile("fixtures/tiny-encoder.json"), 2, 2);

    const Expected = struct { qtype: usize, logits: []f32, act_logits: []f32 };
    const expected = try std.json.parseFromSlice([]Expected, gpa, @embedFile("fixtures/tiny-expected.json"), .{});
    defer expected.deinit();
    for (expected.value) |row| {
        const qtype: QuestionType = @fromBackingInt(@intCast(row.qtype));
        const output = try model.forward(gpa, &.{ 1, 4, 9, 3, 7, 5, 2 }, &.{ 2, 4, 5 }, qtype);
        defer output.deinit(gpa);
        for (row.logits, output.logits) |want, got| try testing.expectApproxEqAbs(want, got, 2e-5);
        for (row.act_logits, output.act_logits) |want, got| try testing.expectApproxEqAbs(want, got, 2e-5);
    }

    try testing.expectError(error.InvalidInput, model.forward(gpa, &.{}, &.{0}, .choice));
    try testing.expectError(error.InvalidToken, model.forward(gpa, &.{19}, &.{0}, .choice));
    try testing.expectError(error.InvalidMarker, model.forward(gpa, &.{1}, &.{1}, .choice));
}

test "unsupported encoder configurations fail before loading weights" {
    var tensors: SafeTensors = try .init(testing.allocator, @embedFile("fixtures/tiny.safetensors"));
    defer tensors.deinit();
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const unsupported = [_][]const u8{
        \\{"model_type":"bert"}
        ,
        \\{"hidden_activation":"relu"}
        ,
        \\{"attention_bias":true}
        ,
        \\{"mlp_bias":true}
        ,
        \\{"norm_bias":true}
        ,
        \\{"hidden_size":18446744073709551615}
        ,
        \\{"hidden_size":-1}
        ,
        \\{"num_hidden_layers":1,"layer_types":["chunked_attention"]}
        ,
        \\{"rope_parameters":{"rope_theta":10000,"rope_type":"linear"}}
        ,
    };
    for (unsupported) |config| {
        try testing.expectError(error.UnsupportedConfiguration, init(arena, arena, &tensors, config, 2, 2));
    }

    const invalid = [_][]const u8{
        \\{"hidden_size":0}
        ,
        \\{"num_attention_heads":0}
        ,
        \\{"norm_eps":0}
        ,
        \\{"global_rope_theta":0}
        ,
        \\{"rope_parameters":0}
        ,
        \\{"rope_parameters":{"full_attention":{"rope_theta":160000}}}
        ,
        \\{"num_hidden_layers":2,"layer_types":["full_attention"]}
        ,
    };
    for (invalid) |config| {
        try testing.expectError(error.InvalidConfiguration, init(arena, arena, &tensors, config, 2, 2));
    }
}
