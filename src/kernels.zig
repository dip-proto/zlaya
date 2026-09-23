//! CPU kernels for the forward pass, using BLAS for matrix products when it is enabled.
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const build_options = @import("build_options");

extern "c" fn erff(f32) f32;

const CblasOrder = enum(c_int) { row_major = 101 };
const CblasTranspose = enum(c_int) { no_trans = 111, trans = 112 };

extern "c" fn cblas_sgemm(
    order: CblasOrder,
    trans_a: CblasTranspose,
    trans_b: CblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) void;

/// Row-major `c = alpha * a * b`, with `b` optionally transposed.
fn sgemm(
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    c: []f32,
    ldc: usize,
) void {
    cblas_sgemm(
        .row_major,
        .no_trans,
        if (transpose_b) .trans else .no_trans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
        0,
        c.ptr,
        @intCast(ldc),
    );
}

/// Computes `out = x * weight^T + bias` for every row of `x`.
/// `weight` is stored with one row per output, as in PyTorch.
pub fn linear(x: []const f32, weight: []const f32, bias: ?[]const f32, out: []f32, in_dim: usize, out_dim: usize) void {
    const rows = x.len / in_dim;
    assert(x.len == rows * in_dim);
    assert(weight.len == out_dim * in_dim);
    assert(out.len == rows * out_dim);

    if (build_options.blas) {
        sgemm(true, rows, out_dim, in_dim, 1, x, in_dim, weight, in_dim, out, out_dim);
    } else {
        for (0..rows) |r| for (0..out_dim) |o| {
            out[r * out_dim + o] = dot(x[r * in_dim ..][0..in_dim], weight[o * in_dim ..][0..in_dim]);
        };
    }
    if (bias) |b| for (0..rows) |r| {
        for (0..out_dim) |o| out[r * out_dim + o] += b[o];
    };
}

fn dot(a: []const f32, b: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        const va: V = a[i..][0..8].*;
        const vb: V = b[i..][0..8].*;
        acc += va * vb;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

pub fn layerNorm(x: []const f32, weight: []const f32, bias: ?[]const f32, out: []f32, eps: f32) void {
    const dim = weight.len;
    const n: f32 = @floatFromInt(dim);
    for (0..x.len / dim) |r| {
        const row = x[r * dim ..][0..dim];
        var mean: f32 = 0;
        for (row) |v| mean += v;
        mean /= n;
        var variance: f32 = 0;
        for (row) |v| variance += (v - mean) * (v - mean);
        const scale = 1 / @sqrt(variance / n + eps);
        for (row, out[r * dim ..][0..dim], 0..) |v, *o, i| {
            o.* = (v - mean) * scale * weight[i] + if (bias) |b| b[i] else 0;
        }
    }
}

/// Exact GELU, not the tanh approximation.
pub fn gelu(x: f32) f32 {
    return 0.5 * x * (1 + erff(x / @sqrt(@as(f32, 2))));
}

pub fn softmax(values: []f32) void {
    const max = std.mem.max(f32, values);
    var sum: f32 = 0;
    for (values) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    for (values) |*v| v.* /= sum;
}

/// Rotary embedding tables, shared by every head and layer that uses the same base.
pub const Rope = struct {
    cos: []const f32,
    sin: []const f32,

    pub fn init(arena: Allocator, seq: usize, head_dim: usize, theta: f32) !Rope {
        const half = head_dim / 2;
        const cos = try arena.alloc(f32, seq * half);
        const sin = try arena.alloc(f32, seq * half);
        for (0..seq) |t| for (0..half) |i| {
            const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
            const angle = @as(f32, @floatFromInt(t)) / std.math.pow(f32, theta, exponent);
            cos[t * half + i] = @cos(angle);
            sin[t * half + i] = @sin(angle);
        };
        return .{ .cos = cos, .sin = sin };
    }
};

/// Multi-head attention over rows packed as `[q | k | v]`.
/// A `window` limits each position to that many neighbors on each side.
/// `scores` needs room for `seq * seq` values.
pub fn attention(
    qkv: []f32,
    out: []f32,
    scores: []f32,
    seq: usize,
    dim: usize,
    heads: usize,
    window: ?usize,
    rope: ?Rope,
) void {
    assert(dim % heads == 0);
    assert(scores.len >= seq * seq);
    const d = dim / heads;
    const stride = 3 * dim;

    if (rope) |table| {
        const half = d / 2;
        for (0..seq) |t| for (0..2) |qk| for (0..heads) |h| {
            const x = qkv[t * stride + qk * dim + h * d ..][0..d];
            const cos = table.cos[t * half ..][0..half];
            const sin = table.sin[t * half ..][0..half];
            for (x[0..half], x[half..], cos, sin) |*a, *b, c, s| {
                const a0 = a.*;
                a.* = a0 * c - b.* * s;
                b.* = b.* * c + a0 * s;
            }
        };
    }

    const scale = 1 / @sqrt(@as(f32, @floatFromInt(d)));
    for (0..heads) |h| {
        const q = qkv[h * d ..];
        const k = qkv[dim + h * d ..];
        const v = qkv[2 * dim + h * d ..];
        if (build_options.blas) {
            sgemm(true, seq, seq, d, scale, q, stride, k, stride, scores, seq);
            for (0..seq) |t| {
                const lo, const hi = span(t, seq, window);
                const row = scores[t * seq ..][0..seq];
                softmax(row[lo..hi]);
                @memset(row[0..lo], 0);
                @memset(row[hi..], 0);
            }
            sgemm(false, seq, d, seq, 1, scores, seq, v, stride, out[h * d ..], dim);
        } else for (0..seq) |t| {
            const lo, const hi = span(t, seq, window);
            const row = scores[0 .. hi - lo];
            for (row, lo..) |*s, j| s.* = dot(q[t * stride ..][0..d], k[j * stride ..][0..d]) * scale;
            softmax(row);
            const target = out[t * dim + h * d ..][0..d];
            @memset(target, 0);
            for (row, lo..) |p, j| {
                for (target, v[j * stride ..][0..d]) |*o, value| o.* += p * value;
            }
        }
    }
}

/// Returns the range of positions that position `t` attends to.
fn span(t: usize, seq: usize, window: ?usize) struct { usize, usize } {
    const w = window orelse return .{ 0, seq };
    return .{ t -| w, @min(seq, t + w + 1) };
}

test linear {
    var out: [4]f32 = undefined;
    linear(&.{ 1, 2, 3, 4 }, &.{ 1, 0, 2, 3 }, &.{ 1, -1 }, &out, 2, 2);
    try testing.expectEqualSlices(f32, &.{ 2, 7, 4, 17 }, &out);
}

test attention {
    var qkv = [_]f32{ 0, 0, 2, 0, 0, 4, 0, 0, 9 };
    var out: [3]f32 = undefined;
    var scores: [9]f32 = undefined;

    attention(&qkv, &out, &scores, 3, 1, 1, 0, null);
    try testing.expectEqualSlices(f32, &.{ 2, 4, 9 }, &out);

    attention(&qkv, &out, &scores, 3, 1, 1, 1, null);
    try testing.expectApproxEqAbs(3, out[0], 1e-6);
    try testing.expectApproxEqAbs(5, out[1], 1e-6);
    try testing.expectApproxEqAbs(6.5, out[2], 1e-6);
}
