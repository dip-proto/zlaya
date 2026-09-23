//! Loads a Laya checkpoint once and answers the questions of any number of requests.
const Engine = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const testing = std.testing;
const kernels = @import("kernels.zig");
const sequence = @import("sequence.zig");
const Model = @import("Model.zig");
const QuestionType = Model.QuestionType;
const SafeTensors = @import("SafeTensors.zig");
const Tokenizer = @import("Tokenizer.zig");

gpa: Allocator,
/// Holds the tokenizer and everything in the model except its weights.
arena: *std.heap.ArenaAllocator,
weights: []f32,
tokenizer: Tokenizer,
model: Model,
max_len: usize,
head_max_len: usize,
temperatures: std.EnumArray(QuestionType, [buckets.len]f32),

/// Option counts that share a calibrated temperature.
const buckets = [_][]const u8{ "2", "3-5", "6-10", "11+" };

fn bucket(option_count: usize) usize {
    if (option_count <= 2) return 0;
    if (option_count <= 5) return 1;
    if (option_count <= 10) return 2;
    return 3;
}

const AgentConfig = struct {
    head_layers: usize = 2,
    max_len: usize = 512,
    head_max_len: usize = 192,
    act_costs: ?std.json.ArrayHashMap(Value) = null,
    temperature: ?[3]Value = null,
    temperature_by_options: ?std.json.ArrayHashMap(Value) = null,

    /// Prefers the temperature fitted for this option count, then the one for the question type.
    fn temperatureFor(config: AgentConfig, qtype: QuestionType, bucket_index: usize) f32 {
        var buffer: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&buffer, "{t}:{s}", .{ qtype, buckets[bucket_index] }) catch unreachable;
        if (config.temperature_by_options) |fitted| {
            if (fitted.map.get(name)) |value| return clampTemperature(value);
        }
        if (config.temperature) |defaults| return clampTemperature(defaults[@backingInt(qtype)]);
        return 1;
    }
};

/// Returns a temperature within the range the upstream implementation allows.
fn clampTemperature(value: Value) f32 {
    const t: f32 = switch (value) {
        .float => |float| @floatCast(float),
        .integer => |integer| @floatFromInt(integer),
        else => return 1,
    };
    return if (std.math.isFinite(t)) std.math.clamp(t, 0.5, 5) else 1;
}

/// `gpa` is also used for scratch memory during predictions.
/// Progress is reported under `progress`, one item per converted tensor.
pub fn load(gpa: Allocator, io: std.Io, path: []const u8, progress: std.Progress.Node) !Engine {
    const arena = try gpa.create(std.heap.ArenaAllocator);
    errdefer gpa.destroy(arena);
    arena.* = .init(gpa);
    errdefer arena.deinit();

    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);

    const config_json = try dir.readFileAlloc(io, "rl_agent_config.json", arena.allocator(), .limited(1024 * 1024));
    // Negative or oversized integers fail with Overflow.
    const config = std.json.parseFromSliceLeaky(AgentConfig, arena.allocator(), config_json, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
        error.Overflow => return error.InvalidConfig,
        else => |e| return e,
    };
    for ([_]usize{ config.max_len, config.head_max_len }) |limit| {
        if (limit < 1 or limit > 8192) return error.InvalidConfig;
    }
    var temperatures: std.EnumArray(QuestionType, [buckets.len]f32) = .initUndefined();
    var it = temperatures.iterator();
    while (it.next()) |entry| {
        for (entry.value, 0..) |*temperature, i| temperature.* = config.temperatureFor(entry.key, i);
    }

    const tokenizer_json = try dir.readFileAlloc(io, "tokenizer/tokenizer.json", gpa, .limited(32 * 1024 * 1024));
    defer gpa.free(tokenizer_json);
    const tokenizer: Tokenizer = try .init(arena.allocator(), tokenizer_json);

    const encoder_json = try dir.readFileAlloc(io, "encoder/config.json", gpa, .limited(1024 * 1024));
    defer gpa.free(encoder_json);
    const file = try dir.openFile(io, "model.safetensors", .{});
    defer file.close(io);
    var tensors: SafeTensors = try .open(gpa, io, file);
    defer tensors.deinit();
    // Only an estimate, since a checkpoint can hold tensors that the model does not use.
    const node = progress.start("Loading model", tensors.tensorCount());
    defer node.end();

    // Allocate all weights in one block.
    // A growing arena rounds its blocks up to powers of two, which may not fit in WebAssembly's 4 GiB.
    const weights = try gpa.alloc(f32, tensors.totalLen());
    errdefer gpa.free(weights);
    var weight_buffer: std.heap.FixedBufferAllocator = .init(std.mem.sliceAsBytes(weights));
    const actions = if (config.act_costs) |costs| costs.map.count() + 1 else 2;
    const model: Model = try .init(arena.allocator(), weight_buffer.allocator(), &tensors, encoder_json, config.head_layers, actions, node);

    return .{
        .gpa = gpa,
        .arena = arena,
        .weights = weights,
        .tokenizer = tokenizer,
        .model = model,
        .max_len = config.max_len,
        .head_max_len = config.head_max_len,
        .temperatures = temperatures,
    };
}

pub fn deinit(engine: *Engine) void {
    engine.gpa.free(engine.weights);
    engine.arena.deinit();
    engine.gpa.destroy(engine.arena);
    engine.* = undefined;
}

/// Returns the response to `request`, allocated in `arena`.
/// The response borrows question IDs and criteria from `request`, which must outlive it.
/// Progress is reported under `progress`, one item per question.
pub fn predict(engine: *const Engine, arena: Allocator, request: Value, raw: bool, progress: std.Progress.Node) !Value {
    if (request != .object) return error.InvalidRequest;
    const state = request.object.get("state") orelse return error.MissingState;
    const questions = request.object.get("questions") orelse return error.MissingQuestions;
    if (questions != .object or questions.object.count() == 0) return error.InvalidQuestions;

    var state_arena: std.heap.ArenaAllocator = .init(engine.gpa);
    defer state_arena.deinit();
    const state_tokens: sequence.State = try .encode(state_arena.allocator(), &engine.tokenizer, state);

    const node = progress.start("Answering questions", questions.object.count());
    defer node.end();
    var scratch: std.heap.ArenaAllocator = .init(engine.gpa);
    defer scratch.deinit();
    var answers: Value = .{ .object = .empty };
    for (questions.object.keys(), questions.object.values()) |id, question| {
        _ = scratch.reset(.retain_capacity);
        const reply = try engine.answer(arena, scratch.allocator(), state_tokens, question, raw);
        try answers.object.put(arena, id, reply);
        node.completeOne();
    }

    var response: Value = .{ .object = .empty };
    try response.object.put(arena, "answers", answers);
    return response;
}

fn answer(
    engine: *const Engine,
    arena: Allocator,
    scratch: Allocator,
    state: sequence.State,
    question: Value,
    raw: bool,
) !Value {
    const seq = try sequence.build(scratch, &engine.tokenizer, state, question, engine.max_len, engine.head_max_len);
    const result = try engine.model.forward(scratch, seq.ids, seq.markers, seq.qtype);

    const probs = try scratch.dupe(f32, result.logits);
    const temperature = engine.temperatures.get(seq.qtype)[bucket(probs.len)];
    for (probs) |*p| p.* /= temperature;
    kernels.softmax(probs);
    const act_probs = try scratch.dupe(f32, result.act_logits);
    kernels.softmax(act_probs);
    for (act_probs) |p| if (!std.math.isFinite(p)) return error.NonFiniteOutput;

    var entropy: f32 = 0;
    var expected: f32 = 0;
    for (probs, 0..) |p, i| {
        if (!std.math.isFinite(p)) return error.NonFiniteOutput;
        entropy -= p * @log(@max(p, 1e-12));
        expected += @as(f32, @floatFromInt(i)) * p;
    }
    const option_count: f32 = @floatFromInt(probs.len);
    const confidence: f32 = if (seq.qtype == .noul)
        @max(probs[1], 1 - probs[1])
    else if (probs.len < 2)
        1
    else
        std.math.clamp(1 - entropy / @log(option_count), 0, 1);

    var out: Value = .{ .object = .empty };
    try out.object.put(arena, "type", .{ .string = @tagName(seq.qtype) });
    try out.object.put(arena, "confidence", rounded(confidence));
    var action: Value = .{ .object = .empty };
    try action.object.put(arena, "act_probability", rounded(act_probs[0]));
    try out.object.put(arena, "action", action);

    // Labels live in scratch memory, so they are copied.
    switch (seq.qtype) {
        .noul => try out.object.put(arena, "noul", rounded(probs[1])),
        .choice, .score => {
            var probabilities: Value = .{ .object = .empty };
            for (probs, seq.labels) |p, label| {
                try probabilities.object.put(arena, try arena.dupe(u8, label), rounded(p));
            }
            try out.object.put(arena, "probabilities", probabilities);

            if (seq.qtype == .choice) {
                const choice = try arena.dupe(u8, seq.labels[std.mem.findMax(f32, probs)]);
                try out.object.put(arena, "choice", .{ .string = choice });
            } else {
                try out.object.put(arena, "score", rounded(expected));
                var legend: Value = .{ .object = .empty };
                const criteria = question.object.get("criteria").?.array.items;
                for (seq.labels, criteria) |label, criterion| {
                    try legend.object.put(arena, try arena.dupe(u8, label), criterion);
                }
                try out.object.put(arena, "legend", legend);
            }
        },
    }

    if (raw) {
        try out.object.put(arena, "ids", try numericArray(arena, seq.ids));
        try out.object.put(arena, "markers", try numericArray(arena, seq.markers));
        try out.object.put(arena, "logits", try numericArray(arena, result.logits));
        try out.object.put(arena, "act_logits", try numericArray(arena, result.act_logits));
    }
    return out;
}

/// Rounds to four decimal places, like the upstream implementation.
fn rounded(value: f32) Value {
    return .{ .float = @round(@as(f64, value) * 10000) / 10000 };
}

fn numericArray(arena: Allocator, values: anytype) !Value {
    var array: std.json.Array = .init(arena);
    try array.ensureTotalCapacity(values.len);
    for (values) |value| array.appendAssumeCapacity(switch (@typeInfo(@TypeOf(value))) {
        .float => .{ .float = value },
        .int => .{ .integer = @intCast(value) },
        else => @compileError("expected integers or floats"),
    });
    return .{ .array = array };
}

test clampTemperature {
    try testing.expectEqual(0.5, clampTemperature(.{ .float = 0.10058 }));
    try testing.expectEqual(5, clampTemperature(.{ .integer = 9 }));
    try testing.expectEqual(1, clampTemperature(.null));
}
