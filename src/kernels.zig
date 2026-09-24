//! CPU kernels for the forward pass.
//! Matrix products run on a Zig kernel that reads weights rearranged by `packWeights`, and share the work with BLAS when it is enabled.
//! Kernels spread their work over the threads of a `Pool`.
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const testing = std.testing;
const build_options = @import("build_options");

const CblasOrder = enum(c_int) { row_major = 101 };
const CblasTranspose = enum(c_int) { no_trans = 111 };

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

/// Row-major `c = a * b + beta * c`.
fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    beta: f32,
    c: []f32,
    ldc: usize,
) void {
    cblas_sgemm(
        .row_major,
        .no_trans,
        .no_trans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
        beta,
        c.ptr,
        @intCast(ldc),
    );
}

/// Worker threads shared by the kernels.
///
/// A forward pass runs hundreds of short batches of jobs in a row.
/// A batch is over as soon as its jobs are done, without waiting for idle threads to wake up.
/// Idle threads spin for a moment before they go to sleep.
pub const Pool = struct {
    /// Null when kernels run on the calling thread only, which is always the case in single-threaded builds.
    shared: if (builtin.single_threaded) ?noreturn else ?*Shared,

    pub const serial: Pool = .{ .shared = null };

    const Shared = struct {
        io: Io,
        threads: []std.Thread,
        /// Lets one batch run at a time.
        mutex: Io.Mutex = .init,
        /// The job count of the current batch in the high half, and the next job to hand out in the low half.
        state: std.atomic.Value(u64) = .init(0),
        /// Goes up with every batch, and idle threads wait for it to change.
        generation: std.atomic.Value(u32) = .init(0),
        /// Threads inside `claim`.
        /// A batch isn't over until they leave, so none of them can carry anything over to the next batch.
        active: std.atomic.Value(u32) = .init(0),
        completed: std.atomic.Value(u32) = .init(0),
        /// Hands out `worker` numbers to the threads that take jobs in the current batch.
        slots: std.atomic.Value(u32) = .init(0),
        /// Threads that are asleep, or about to be, until the next batch.
        sleeping: std.atomic.Value(u32) = .init(0),
        /// Set while the thread that started the batch sleeps until it is done.
        waiting: std.atomic.Value(u32) = .init(0),
        stopping: std.atomic.Value(bool) = .init(false),
        batch: Batch = undefined,
    };

    const Batch = struct {
        context: *anyopaque,
        job: *const fn (context: *anyopaque, index: usize, worker: usize) void,
    };

    /// About 17 microseconds on an Apple M5, which is longer than most gaps between batches in a forward pass.
    const idle_spins = 2_000;
    const finish_spins = 100_000;

    /// Starts a thread for every CPU but one, unless the build is single-threaded.
    /// The threads use `io` to wait for work, so it must outlive the pool.
    pub fn init(gpa: Allocator, io: Io) !Pool {
        if (builtin.single_threaded) return .serial;
        const cpus = std.Thread.getCpuCount() catch 1;
        if (cpus < 2) return .serial;

        const shared = try gpa.create(Shared);
        errdefer gpa.destroy(shared);
        shared.* = .{ .io = io, .threads = try gpa.alloc(std.Thread, cpus - 1) };
        errdefer gpa.free(shared.threads);
        for (shared.threads, 0..) |*thread, i| {
            thread.* = std.Thread.spawn(.{ .stack_size = 1024 * 1024 }, work, .{shared}) catch |err| {
                stop(shared, shared.threads[0..i]);
                return err;
            };
        }
        return .{ .shared = shared };
    }

    pub fn deinit(pool: Pool, gpa: Allocator) void {
        const shared = pool.shared orelse return;
        stop(shared, shared.threads);
        gpa.free(shared.threads);
        gpa.destroy(shared);
    }

    fn stop(shared: *Shared, started: []std.Thread) void {
        shared.stopping.store(true, .release);
        _ = shared.generation.fetchAdd(1, .seq_cst);
        shared.io.futexWake(u32, &shared.generation.raw, std.math.maxInt(u32));
        for (started) |thread| thread.join();
    }

    fn work(shared: *Shared) void {
        var seen: u32 = 0;
        while (true) {
            var spins: usize = 0;
            while (shared.generation.load(.acquire) == seen) : (spins += 1) {
                if (shared.stopping.load(.acquire)) return;
                if (spins < idle_spins) {
                    std.atomic.spinLoopHint();
                } else {
                    // Either `run` sees this thread as sleeping and wakes it, or the wait sees the new generation.
                    // The timeout only matters if the counter wraps around to `seen` while the pool stops.
                    _ = shared.sleeping.fetchAdd(1, .seq_cst);
                    const timeout: Io.Timeout = .{ .duration = .{ .raw = .fromSeconds(1), .clock = .awake } };
                    shared.io.futexWaitTimeout(u32, &shared.generation.raw, seen, timeout) catch {};
                    _ = shared.sleeping.fetchSub(1, .seq_cst);
                }
            }
            if (shared.stopping.load(.acquire)) return;
            seen = claim(shared);
        }
    }

    /// Runs jobs of the current batch until none are left.
    /// Returns the batch's generation, so that a thread joins each batch only once and never gets two worker numbers in it.
    fn claim(shared: *Shared) u32 {
        _ = shared.active.fetchAdd(1, .seq_cst);
        defer _ = shared.active.fetchSub(1, .release);
        var slot: ?usize = null;
        while (true) {
            const state = shared.state.load(.acquire);
            const jobs: u32 = @truncate(state >> 32);
            const next: u32 = @truncate(state);
            // No new batch can start while this thread is in here.
            if (next >= jobs) return shared.generation.load(.acquire);
            if (shared.state.cmpxchgWeak(state, state + 1, .acquire, .monotonic) != null) continue;

            const worker = slot orelse shared.slots.fetchAdd(1, .monotonic);
            slot = worker;
            const batch = shared.batch;
            batch.job(batch.context, next, worker);
            if (shared.completed.fetchAdd(1, .seq_cst) + 1 == jobs and shared.waiting.load(.seq_cst) != 0) {
                shared.io.futexWake(u32, &shared.completed.raw, 1);
            }
        }
    }

    /// Number of threads, the calling one included.
    pub fn threads(pool: Pool) usize {
        const shared = pool.shared orelse return 1;
        return shared.threads.len + 1;
    }

    /// Number of different `worker` values that `run` can pass for `jobs` jobs.
    fn workers(pool: Pool, jobs: usize) usize {
        return @max(1, @min(pool.threads(), jobs));
    }

    /// Calls `job(context, index, worker)` for every index below `jobs`, where `context` is a pointer.
    /// Threads take jobs one at a time, so faster cores end up doing more of them.
    ///
    /// `worker` is below `pool.workers(jobs)`, and no two jobs of one call see the same value at the same time.
    /// That lets it pick scratch memory owned by the caller.
    pub fn run(
        pool: Pool,
        jobs: usize,
        context: anytype,
        comptime job: fn (@TypeOf(context), usize, usize) void,
    ) void {
        comptime assert(@typeInfo(@TypeOf(context)) == .pointer);
        const Erased = struct {
            fn call(erased: *anyopaque, index: usize, worker: usize) void {
                job(@ptrCast(@alignCast(erased)), index, worker);
            }
        };

        const shared = (if (jobs < 2) null else pool.shared) orelse {
            for (0..jobs) |index| job(context, index, 0);
            return;
        };
        const count: u64 = std.math.cast(u32, jobs) orelse @panic("too many jobs for one batch");
        shared.mutex.lockUncancelable(shared.io);
        defer shared.mutex.unlock(shared.io);

        shared.batch = .{ .context = @ptrCast(@constCast(context)), .job = Erased.call };
        shared.completed.store(0, .monotonic);
        shared.slots.store(0, .monotonic);
        shared.state.store(count << 32, .release);
        _ = shared.generation.fetchAdd(1, .seq_cst);
        if (shared.sleeping.load(.seq_cst) != 0) {
            shared.io.futexWake(u32, &shared.generation.raw, std.math.maxInt(u32));
        }
        _ = claim(shared);

        var spins: usize = 0;
        while (shared.completed.load(.acquire) != count) : (spins += 1) {
            if (spins < finish_spins) {
                std.atomic.spinLoopHint();
            } else {
                shared.waiting.store(1, .seq_cst);
                const done = shared.completed.load(.seq_cst);
                if (done != count) shared.io.futexWaitUncancelable(u32, &shared.completed.raw, done);
                shared.waiting.store(0, .seq_cst);
            }
        }
        // Wait for threads still in `claim`, which have nothing left to do but may have been paused by the system.
        spins = 0;
        while (shared.active.load(.acquire) != 0) : (spins += 1) {
            if (spins < finish_spins) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
        }
    }

    /// Calls `function(context, row)` for every row below `rows`, a few rows per job.
    fn eachRow(
        pool: Pool,
        rows: usize,
        context: anytype,
        comptime function: fn (@TypeOf(context), usize) void,
    ) void {
        const rows_per_job = 8;
        const Rows = struct {
            count: usize,
            context: @TypeOf(context),

            fn run(batch: *const @This(), job: usize, _: usize) void {
                const first = job * rows_per_job;
                for (first..@min(first + rows_per_job, batch.count)) |row| function(batch.context, row);
            }
        };
        const batch: Rows = .{ .count = rows, .context = context };
        pool.run(@divCeil(rows, rows_per_job), &batch, Rows.run);
    }
};

const vector_len = std.simd.suggestVectorLength(f32) orelse 4;
const V = @Vector(vector_len, f32);

/// Returns `a * b + c`, fused where the target has a fused instruction.
/// `@mulAdd` would call a slow software routine on other targets, WebAssembly included.
inline fn mulAdd(a: V, b: V, c: V) V {
    @setFloatMode(.optimized);
    return a * b + c;
}

/// Loads the vector at `index`, filling the part past the end of `values` with `fill`.
fn load(values: []const f32, index: usize, fill: f32) V {
    if (index + vector_len <= values.len) return values[index..][0..vector_len].*;
    var padded: [vector_len]f32 = @splat(fill);
    @memcpy(padded[0 .. values.len - index], values[index..]);
    return padded;
}

/// Stores the part of `vector` that fits in `values`.
fn store(values: []f32, index: usize, vector: V) void {
    if (index + vector_len <= values.len) {
        values[index..][0..vector_len].* = vector;
    } else {
        const array: [vector_len]f32 = vector;
        @memcpy(values[index..], array[0 .. values.len - index]);
    }
}

/// Returns e^x for x up to 88, and zero below -87, with the Cephes single-precision algorithm.
fn exp(x: V) V {
    // Adding 1.5 * 2^23 rounds to the nearest integer and leaves it in the low bits.
    const round: V = @splat(0x1.8p23);
    const shifted = mulAdd(x, @splat(std.math.log2e), round);
    const n = shifted - round;
    var r = mulAdd(n, @splat(-0.693359375), x);
    r = mulAdd(n, @splat(2.12194440e-4), r);

    var p: V = @splat(1.9875691500e-4);
    p = mulAdd(p, r, @splat(1.3981999507e-3));
    p = mulAdd(p, r, @splat(8.3334519073e-3));
    p = mulAdd(p, r, @splat(4.1665795894e-2));
    p = mulAdd(p, r, @splat(1.6666665459e-1));
    p = mulAdd(p, r, @splat(5.0000001201e-1));
    const y = mulAdd(p, r * r, r) + @as(V, @splat(1));

    const U = @Vector(vector_len, u32);
    const bits: U = @bitCast(shifted);
    const scale: V = @bitCast((bits << @splat(23)) +% @as(U, @splat(127 << 23)));
    return @select(f32, x < @as(V, @splat(-87)), @as(V, @splat(0)), y * scale);
}

/// Exact GELU, not the tanh approximation, accurate to 5e-7.
/// erf comes from Abramowitz and Stegun 7.1.26, like in PyTorch's vectorized CPU code.
fn geluVector(x: V) V {
    const z = x * @as(V, @splat(std.math.sqrt1_2));
    const t = @as(V, @splat(1)) / mulAdd(@abs(z), @splat(0.3275911), @splat(1));
    var p: V = @splat(1.061405429);
    p = mulAdd(p, t, @splat(-1.453152027));
    p = mulAdd(p, t, @splat(1.421413741));
    p = mulAdd(p, t, @splat(-0.284496736));
    p = mulAdd(p, t, @splat(0.254829592));
    // This is erfc(|z|), which gives 1 + erf(z) without losing precision when z is negative.
    const erfc = p * t * exp(-z * z);
    const cdf = @select(f32, z < @as(V, @splat(0)), erfc, @as(V, @splat(2)) - erfc);
    return @as(V, @splat(0.5)) * x * cdf;
}

pub fn gelu(values: []f32) void {
    var i: usize = 0;
    while (i < values.len) : (i += vector_len) store(values, i, geluVector(load(values, i, 0)));
}

/// Computes `gelu(a) * b` for rows laid out as `[a | b]`, both halves `width` wide.
pub fn geluGate(pool: Pool, input: []const f32, out: []f32, width: usize) void {
    assert(input.len == 2 * out.len);
    const Gate = struct {
        input: []const f32,
        out: []f32,
        width: usize,

        fn row(gate: *const @This(), r: usize) void {
            const a = gate.input[2 * r * gate.width ..][0..gate.width];
            const b = gate.input[(2 * r + 1) * gate.width ..][0..gate.width];
            const target = gate.out[r * gate.width ..][0..gate.width];
            var i: usize = 0;
            while (i < gate.width) : (i += vector_len) {
                store(target, i, geluVector(load(a, i, 0)) * load(b, i, 0));
            }
        }
    };
    const gate: Gate = .{ .input = input, .out = out, .width = width };
    pool.eachRow(out.len / width, &gate, Gate.row);
}

pub fn softmax(values: []f32) void {
    var max: V = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i < values.len) : (i += vector_len) max = @max(max, load(values, i, -std.math.inf(f32)));
    const shift: V = @splat(@reduce(.Max, max));

    var sums: V = @splat(0);
    i = 0;
    while (i < values.len) : (i += vector_len) {
        const e = exp(load(values, i, -std.math.inf(f32)) - shift);
        store(values, i, e);
        sums += e;
    }
    const sum: V = @splat(@reduce(.Add, sums));
    i = 0;
    while (i < values.len) : (i += vector_len) store(values, i, load(values, i, 0) / sum);
}

pub fn layerNorm(
    pool: Pool,
    x: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    out: []f32,
    eps: f32,
) void {
    const Norm = struct {
        x: []const f32,
        weight: []const f32,
        bias: ?[]const f32,
        out: []f32,
        eps: f32,

        fn row(norm: *const @This(), r: usize) void {
            const dim = norm.weight.len;
            const n: f32 = @floatFromInt(dim);
            const values = norm.x[r * dim ..][0..dim];
            const target = norm.out[r * dim ..][0..dim];

            var sums: V = @splat(0);
            var i: usize = 0;
            while (i < dim) : (i += vector_len) sums += load(values, i, 0);
            const mean = @reduce(.Add, sums) / n;

            const means: V = @splat(mean);
            sums = @splat(0);
            i = 0;
            while (i < dim) : (i += vector_len) {
                const centered = load(values, i, mean) - means;
                sums += centered * centered;
            }
            const scale: V = @splat(1 / @sqrt(@reduce(.Add, sums) / n + norm.eps));

            i = 0;
            while (i < dim) : (i += vector_len) {
                const normalized = (load(values, i, 0) - means) * scale * load(norm.weight, i, 0);
                store(target, i, if (norm.bias) |b| normalized + load(b, i, 0) else normalized);
            }
        }
    };
    const norm: Norm = .{ .x = x, .weight = weight, .bias = bias, .out = out, .eps = eps };
    pool.eachRow(x.len / weight.len, &norm, Norm.row);
}

fn dot(a: []const f32, b: []const f32) f32 {
    var sums: [4]V = @splat(@splat(0));
    var i: usize = 0;
    while (i + 4 * vector_len <= a.len) : (i += 4 * vector_len) {
        inline for (&sums, 0..) |*sum, j| {
            const offset = i + j * vector_len;
            sum.* = mulAdd(a[offset..][0..vector_len].*, b[offset..][0..vector_len].*, sum.*);
        }
    }
    while (i < a.len) : (i += vector_len) sums[0] = mulAdd(load(a, i, 0), load(b, i, 0), sums[0]);
    return @reduce(.Add, (sums[0] + sums[1]) + (sums[2] + sums[3]));
}

/// Outputs that the matrix kernel computes together.
const tile_width = 16;
const tile_vectors = tile_width / vector_len;
/// AArch64 can multiply by a single vector lane, so there the matrix kernel loads several inputs of each row at once.
/// Broadcasting one value at a time is faster elsewhere, WebAssembly on AArch64 included.
const lane_multiply = builtin.cpu.arch.isAARCH64();
/// Input rows that the matrix kernel reads together, as many as the registers allow.
const tile_rows = if (lane_multiply) 5 else 6;

/// Multiplies `rows` rows of `a` by `tile_width` columns of `b`, over `len` inputs.
/// Rows of `a` are `a_stride` apart, and each row of `b` starts `b_stride` after the previous one.
inline fn tile(
    comptime rows: usize,
    a: [*]const f32,
    a_stride: usize,
    b: [*]const f32,
    b_stride: usize,
    len: usize,
) [rows][tile_vectors]V {
    var sums: [rows][tile_vectors]V = @splat(@splat(@splat(0)));
    var k: usize = 0;
    if (lane_multiply) while (k + vector_len <= len) : (k += vector_len) {
        var a_vectors: [rows]V = undefined;
        inline for (&a_vectors, 0..) |*vector, i| vector.* = a[i * a_stride + k ..][0..vector_len].*;
        inline for (0..vector_len) |lane| {
            const bk = loadTileRow(b + (k + lane) * b_stride);
            inline for (&sums, a_vectors) |*row_sums, a_vector| {
                const ak: V = @splat(a_vector[lane]);
                inline for (row_sums, bk) |*sum, vector| sum.* = mulAdd(ak, vector, sum.*);
            }
        }
    };
    while (k < len) : (k += 1) {
        const bk = loadTileRow(b + k * b_stride);
        inline for (&sums, 0..) |*row_sums, i| {
            const ak: V = @splat(a[i * a_stride + k]);
            inline for (row_sums, bk) |*sum, vector| sum.* = mulAdd(ak, vector, sum.*);
        }
    }
    return sums;
}

inline fn loadTileRow(row: [*]const f32) [tile_vectors]V {
    var vectors: [tile_vectors]V = undefined;
    inline for (&vectors, 0..) |*vector, j| vector.* = row[j * vector_len ..][0..vector_len].*;
    return vectors;
}

/// Calls `function(context, rows, row)` for each group of up to `tile_rows` rows in `first..last`, with `rows` known at compile time.
inline fn eachTile(first: usize, last: usize, context: anytype, comptime function: anytype) void {
    var row = first;
    while (row < last) : (row += tile_rows) switch (@min(tile_rows, last - row)) {
        inline 1...tile_rows => |rows| function(context, rows, row),
        else => unreachable,
    };
}

/// Outputs per packed weight panel.
/// BLAS works on whole panels, and needs them wider than the Zig kernel's tiles to be fast.
const panel_width = if (build_options.blas) 64 else tile_width;
const tiles_per_panel = panel_width / tile_width;
comptime {
    assert(panel_width % tile_width == 0);
}

/// Length of the buffer that `packWeights` needs for rows of `inputs` values.
pub fn packBufferLen(pool: Pool, inputs: usize) usize {
    return pool.threads() * panel_width * inputs;
}

/// Rearranges weights with one row per output, in place, into the layout that `linear` reads.
/// Each group of `panel_width` outputs becomes a panel with one row per input.
/// Outputs past the last full group keep their rows.
pub fn packWeights(pool: Pool, weight: []f32, inputs: usize, buffer: []f32) void {
    assert(buffer.len >= packBufferLen(pool, inputs));
    const Pack = struct {
        weight: []f32,
        inputs: usize,
        buffer: []f32,

        fn run(pack: *const @This(), p: usize, worker: usize) void {
            const panel_len = panel_width * pack.inputs;
            const panel = pack.weight[p * panel_len ..][0..panel_len];
            const rows = pack.buffer[worker * panel_len ..][0..panel_len];
            @memcpy(rows, panel);
            // Transposing 4x4 blocks in registers is several times faster than moving single values.
            const F4 = @Vector(4, f32);
            var k: usize = 0;
            while (k + 4 <= pack.inputs) : (k += 4) {
                var j: usize = 0;
                while (j < panel_width) : (j += 4) {
                    var block: [4]F4 = undefined;
                    for (&block, 0..) |*row, i| row.* = rows[(j + i) * pack.inputs + k ..][0..4].*;
                    const low01 = @shuffle(f32, block[0], block[1], [4]i32{ 0, -1, 1, -2 });
                    const low23 = @shuffle(f32, block[2], block[3], [4]i32{ 0, -1, 1, -2 });
                    const high01 = @shuffle(f32, block[0], block[1], [4]i32{ 2, -3, 3, -4 });
                    const high23 = @shuffle(f32, block[2], block[3], [4]i32{ 2, -3, 3, -4 });
                    const columns = [4]F4{
                        @shuffle(f32, low01, low23, [4]i32{ 0, 1, -1, -2 }),
                        @shuffle(f32, low01, low23, [4]i32{ 2, 3, -3, -4 }),
                        @shuffle(f32, high01, high23, [4]i32{ 0, 1, -1, -2 }),
                        @shuffle(f32, high01, high23, [4]i32{ 2, 3, -3, -4 }),
                    };
                    for (columns, 0..) |column, i| panel[(k + i) * panel_width + j ..][0..4].* = column;
                }
            }
            for (k..pack.inputs) |tail| for (0..panel_width) |j| {
                panel[tail * panel_width + j] = rows[j * pack.inputs + tail];
            };
        }
    };
    const pack: Pack = .{ .weight = weight, .inputs = inputs, .buffer = buffer };
    pool.run(weight.len / (panel_width * inputs), &pack, Pack.run);
}

/// Whether `linear` overwrites its output or adds to it.
pub const Store = enum { replace, add };

/// Computes `x * weight^T + bias` for every row of `x`, and writes it to `out` or adds it.
/// `weight` has one row per output, rearranged by `packWeights`.
pub fn linear(
    pool: Pool,
    x: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    out: []f32,
    in_dim: usize,
    out_dim: usize,
    mode: Store,
) void {
    const rows = x.len / in_dim;
    assert(x.len == rows * in_dim);
    assert(weight.len == out_dim * in_dim);
    assert(out.len == rows * out_dim);

    const gemm: Gemm = .{
        .x = x,
        .weight = weight,
        .bias = bias,
        .out = out,
        .rows = rows,
        .inputs = in_dim,
        .outputs = out_dim,
        .mode = mode,
    };
    // Waking threads costs more than it saves on small products.
    const parallel: Pool = if (rows * in_dim * out_dim < 1 << 22) .serial else pool;
    if (build_options.blas) {
        var split: Gemm.Split = .init(&gemm);
        parallel.run(parallel.threads(), &split, Gemm.Split.work);
    } else {
        parallel.run(gemm.blocks(0), &gemm, Gemm.runBlock);
    }
}

/// A `linear` call split into jobs: blocks of rows and tiles for the Zig kernel, and whole panels for BLAS.
const Gemm = struct {
    x: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    out: []f32,
    rows: usize,
    inputs: usize,
    outputs: usize,
    mode: Store,

    const block_rows = 8 * tile_rows;
    const block_tiles = 4;

    fn panels(gemm: *const Gemm) usize {
        return gemm.outputs / panel_width;
    }

    fn rowBlocks(gemm: *const Gemm) usize {
        return @divCeil(gemm.rows, block_rows);
    }

    /// Number of blocks that cover the panels from `first_panel` on, and the outputs past the last panel.
    fn blocks(gemm: *const Gemm, first_panel: usize) usize {
        const tile_blocks = @divCeil((gemm.panels() - first_panel) * tiles_per_panel, block_tiles);
        const remainder_blocks = @intFromBool(gemm.outputs % panel_width != 0);
        return gemm.rowBlocks() * (tile_blocks + remainder_blocks);
    }

    fn runBlock(gemm: *const Gemm, block: usize, _: usize) void {
        gemm.runBlockFrom(0, block);
    }

    /// Runs one of the blocks counted by `blocks(first_panel)`.
    fn runBlockFrom(gemm: *const Gemm, first_panel: usize, block: usize) void {
        const row_blocks = gemm.rowBlocks();
        const first_row = block % row_blocks * block_rows;
        const last_row = @min(first_row + block_rows, gemm.rows);
        const first_tile = first_panel * tiles_per_panel + block / row_blocks * block_tiles;
        const end_tile = gemm.panels() * tiles_per_panel;
        if (first_tile >= end_tile) return gemm.remainder(first_row, last_row);
        for (first_tile..@min(first_tile + block_tiles, end_tile)) |t| {
            eachTile(first_row, last_row, Tile{ .gemm = gemm, .index = t }, Tile.update);
        }
    }

    /// With BLAS, a few workers send the first panels to BLAS, while the others run the Zig kernel on the rest.
    /// With Accelerate, this keeps the matrix units and the vector units of the other cores busy at the same time.
    /// Workers that run out of their kind of work help with the other kind, so both finish together, even if no BLAS worker runs at all.
    const Split = struct {
        gemm: *const Gemm,
        blas_panels: usize,
        next_panel: std.atomic.Value(usize) = .init(0),
        next_block: std.atomic.Value(usize) = .init(0),

        /// Both were tuned with Accelerate on an Apple M5 Max.
        const blas_workers = 8;
        /// The Zig kernel gets one panel out of this many.
        const zig_share = 4;

        fn init(gemm: *const Gemm) Split {
            return .{ .gemm = gemm, .blas_panels = gemm.panels() - @divCeil(gemm.panels(), zig_share) };
        }

        fn work(split: *Split, _: usize, worker: usize) void {
            if (worker < blas_workers) {
                split.panels();
                split.blocks();
            } else {
                split.blocks();
                split.panels();
            }
        }

        fn panels(split: *Split) void {
            while (true) {
                const panel = split.next_panel.fetchAdd(1, .monotonic);
                if (panel >= split.blas_panels) return;
                split.gemm.blasPanel(panel);
            }
        }

        fn blocks(split: *Split) void {
            const count = split.gemm.blocks(split.blas_panels);
            while (true) {
                const block = split.next_block.fetchAdd(1, .monotonic);
                if (block >= count) return;
                split.gemm.runBlockFrom(split.blas_panels, block);
            }
        }
    };

    fn blasPanel(gemm: *const Gemm, panel: usize) void {
        const column = panel * panel_width;
        const target = gemm.out[column..];
        const beta: f32 = switch (gemm.mode) {
            .replace => 0,
            .add => 1,
        };
        const weights = gemm.weight[column * gemm.inputs ..];
        sgemm(gemm.rows, panel_width, gemm.inputs, gemm.x, gemm.inputs, weights, panel_width, beta, target, gemm.outputs);
        if (gemm.bias) |bias| for (0..gemm.rows) |r| {
            for (target[r * gemm.outputs ..][0..panel_width], bias[column..][0..panel_width]) |*o, value| o.* += value;
        };
    }

    /// The outputs starting at `index * tile_width`, for a group of rows.
    const Tile = struct {
        gemm: *const Gemm,
        index: usize,

        fn update(context: Tile, comptime rows: usize, row: usize) void {
            const gemm = context.gemm;
            const column = context.index * tile_width;
            const b = gemm.weight[column / panel_width * gemm.inputs * panel_width + column % panel_width ..].ptr;
            const sums = tile(rows, gemm.x[row * gemm.inputs ..].ptr, gemm.inputs, b, panel_width, gemm.inputs);

            inline for (sums, 0..) |row_sums, i| {
                const target = gemm.out[(row + i) * gemm.outputs + column ..][0..tile_width];
                inline for (row_sums, 0..) |sum, j| {
                    const slot = target[j * vector_len ..][0..vector_len];
                    var value = sum;
                    if (gemm.bias) |bias| value += bias[column + j * vector_len ..][0..vector_len].*;
                    if (gemm.mode == .add) value += slot.*;
                    slot.* = value;
                }
            }
        }
    };

    /// Computes the outputs past the last full panel, whose weights are still rows.
    fn remainder(gemm: *const Gemm, first_row: usize, last_row: usize) void {
        for (first_row..last_row) |row| {
            const x = gemm.x[row * gemm.inputs ..][0..gemm.inputs];
            for (gemm.panels() * panel_width..gemm.outputs) |o| {
                const slot = &gemm.out[row * gemm.outputs + o];
                var value = dot(x, gemm.weight[o * gemm.inputs ..][0..gemm.inputs]);
                if (gemm.bias) |bias| value += bias[o];
                if (gemm.mode == .add) value += slot.*;
                slot.* = value;
            }
        }
    }
};

/// Rotary embedding tables, shared by every head and layer with the same base.
pub const Rope = struct {
    cos: []const f32,
    sin: []const f32,

    pub fn init(arena: Allocator, seq: usize, head_dim: usize, theta: f32) !Rope {
        const half = head_dim / 2;
        const cos = try arena.alloc(f32, seq * half);
        const sin = try arena.alloc(f32, seq * half);
        const dims: f32 = @floatFromInt(head_dim);
        for (0..half) |i| {
            // Multiplying by the inverse frequency, instead of dividing, rounds like PyTorch.
            const index: f32 = @floatFromInt(2 * i);
            const frequency = 1 / std.math.pow(f32, theta, index / dims);
            for (0..seq) |t| {
                const position: f32 = @floatFromInt(t);
                const angle = position * frequency;
                cos[t * half + i] = @cos(angle);
                sin[t * half + i] = @sin(angle);
            }
        }
        return .{ .cos = cos, .sin = sin };
    }
};

/// Queries that one attention job handles.
const query_block = 4 * tile_rows;

fn paddedKeys(seq: usize) usize {
    return std.mem.alignForward(usize, seq, tile_width);
}

/// Number of values of scratch memory that `attention` needs.
pub fn attentionScratchLen(pool: Pool, seq: usize, dim: usize, heads: usize) usize {
    const keys = paddedKeys(seq);
    const jobs = heads * @divCeil(seq, query_block);
    return keys * dim + pool.workers(jobs) * query_block * keys;
}

/// Multi-head attention over rows laid out as `[q | k | v]`.
/// With a `window`, each position only sees that many neighbors on each side.
/// `scratch` needs `attentionScratchLen(pool, seq, dim, heads)` values.
pub fn attention(
    pool: Pool,
    qkv: []f32,
    out: []f32,
    scratch: []f32,
    seq: usize,
    dim: usize,
    heads: usize,
    window: ?usize,
    rope: ?Rope,
) void {
    assert(dim % heads == 0);
    assert(scratch.len >= attentionScratchLen(pool, seq, dim, heads));
    const keys_len = paddedKeys(seq) * dim;
    const att: Attention = .{
        .qkv = qkv,
        .out = out,
        .keys = scratch[0..keys_len],
        .scores = scratch[keys_len..],
        .seq = seq,
        .dim = dim,
        .heads = heads,
        .window = window,
        .rope = rope,
    };
    pool.run(heads, &att, Attention.prepare);
    pool.run(heads * @divCeil(seq, query_block), &att, Attention.run);
}

const Attention = struct {
    qkv: []f32,
    out: []f32,
    /// Keys of every head, in tiles of `tile_width` keys with one row per dimension.
    keys: []f32,
    /// Scores of one block of queries for each worker.
    scores: []f32,
    seq: usize,
    dim: usize,
    heads: usize,
    window: ?usize,
    rope: ?Rope,

    fn headDim(att: *const Attention) usize {
        return att.dim / att.heads;
    }

    fn stride(att: *const Attention) usize {
        return 3 * att.dim;
    }

    /// Applies rotary embeddings to the queries and keys of `head`, then packs its keys.
    fn prepare(att: *const Attention, head: usize, _: usize) void {
        if (att.rope) |table| att.rotate(head, table);
        att.packKeys(head);
    }

    fn rotate(att: *const Attention, head: usize, table: Rope) void {
        const d = att.headDim();
        const half = d / 2;
        for (0..att.seq) |t| {
            const cos = table.cos[t * half ..][0..half];
            const sin = table.sin[t * half ..][0..half];
            for ([_]usize{ 0, att.dim }) |offset| {
                const x = att.qkv[t * att.stride() + offset + head * d ..][0..d];
                var i: usize = 0;
                while (i < half) : (i += vector_len) {
                    const a = load(x[0..half], i, 0);
                    const b = load(x[half..], i, 0);
                    const c = load(cos, i, 0);
                    const s = load(sin, i, 0);
                    store(x[0..half], i, a * c - b * s);
                    store(x[half..], i, b * c + a * s);
                }
            }
        }
    }

    fn packKeys(att: *const Attention, head: usize) void {
        const d = att.headDim();
        const keys = att.qkv[att.dim + head * d ..];
        const padded = paddedKeys(att.seq);
        const target = att.keys[head * padded * d ..][0 .. padded * d];
        for (0..padded / tile_width) |t| for (0..tile_width) |j| {
            const key = t * tile_width + j;
            for (0..d) |i| {
                target[(t * d + i) * tile_width + j] = if (key < att.seq) keys[key * att.stride() + i] else 0;
            }
        };
    }

    fn run(att: *const Attention, job: usize, worker: usize) void {
        const d = att.headDim();
        const head = job % att.heads;
        const first = job / att.heads * query_block;
        const count = @min(query_block, att.seq - first);
        const lo, _ = span(first, att.seq, att.window);
        _, const hi = span(first + count - 1, att.seq, att.window);
        const padded = paddedKeys(att.seq);
        const scores = att.scores[worker * query_block * padded ..][0 .. query_block * padded];
        const out = att.out[first * att.dim + head * d ..];

        const head_dim: f32 = @floatFromInt(d);

        // Scores cover whole tiles of keys, starting at `base`.
        const base = std.mem.alignBackward(usize, lo, tile_width);
        const width = std.mem.alignForward(usize, hi, tile_width) - base;
        const products: Products = .{
            .a = att.qkv[first * att.stride() + head * d ..],
            .a_stride = att.stride(),
            .b = att.keys[(head * padded + base) * d ..],
            .b_stride = tile_width,
            .b_tile = d * tile_width,
            .len = d,
            .out = scores,
            .out_stride = width,
            .scale = 1 / @sqrt(head_dim),
        };
        products.compute(count, width / tile_width);
        att.normalize(scores, width, first, count, base);

        const probabilities = scores[lo - base ..];
        const values = att.qkv[lo * att.stride() + 2 * att.dim + head * d ..];
        if (d % tile_width == 0) {
            const weighted: Products = .{
                .a = probabilities,
                .a_stride = width,
                .b = values,
                .b_stride = att.stride(),
                .b_tile = tile_width,
                .len = hi - lo,
                .out = out,
                .out_stride = att.dim,
            };
            weighted.compute(count, d / tile_width);
        } else for (0..count) |i| {
            const target = out[i * att.dim ..][0..d];
            @memset(target, 0);
            for (probabilities[i * width ..][0 .. hi - lo], 0..) |p, j| {
                for (target, values[j * att.stride() ..][0..d]) |*o, value| o.* += p * value;
            }
        }
    }

    /// Turns rows of scores into probabilities, with zeros outside each query's span.
    /// Row `i` holds the scores of query `first + i` for the keys from `base` on.
    fn normalize(att: *const Attention, scores: []f32, width: usize, first: usize, count: usize, base: usize) void {
        for (0..count) |i| {
            const row = scores[i * width ..][0..width];
            const lo, const hi = span(first + i, att.seq, att.window);
            softmax(row[lo - base .. hi - base]);
            @memset(row[0 .. lo - base], 0);
            @memset(row[hi - base ..], 0);
        }
    }

    /// Products of rows of `a` with tiles of `b`, scaled and stored in rows of `out`.
    const Products = struct {
        a: []const f32,
        a_stride: usize,
        b: []const f32,
        b_stride: usize,
        /// Distance in `b` between the starts of two tiles.
        b_tile: usize,
        len: usize,
        out: []f32,
        out_stride: usize,
        scale: f32 = 1,

        fn compute(products: *const Products, rows: usize, tiles: usize) void {
            for (0..tiles) |t| eachTile(0, rows, Tile{ .products = products, .index = t }, Tile.update);
        }

        const Tile = struct {
            products: *const Products,
            index: usize,

            fn update(context: Tile, comptime rows: usize, row: usize) void {
                const p = context.products;
                const b = p.b[context.index * p.b_tile ..].ptr;
                const sums = tile(rows, p.a[row * p.a_stride ..].ptr, p.a_stride, b, p.b_stride, p.len);
                const scale: V = @splat(p.scale);
                inline for (sums, 0..) |row_sums, i| {
                    const target = p.out[(row + i) * p.out_stride + context.index * tile_width ..];
                    inline for (row_sums, 0..) |sum, j| target[j * vector_len ..][0..vector_len].* = sum * scale;
                }
            }
        };
    };
};

/// Returns the range of positions that position `t` attends to.
fn span(t: usize, seq: usize, window: ?usize) struct { usize, usize } {
    const w = window orelse return .{ 0, seq };
    return .{ t -| w, @min(seq, t + w + 1) };
}

test "pool runs every job once and never shares a worker number" {
    const pool: Pool = try .init(testing.allocator, testing.io);
    defer pool.deinit(testing.allocator);
    const Check = struct {
        runs: [64]std.atomic.Value(u32) = @splat(.init(0)),
        busy: [64]std.atomic.Value(bool) = @splat(.init(false)),
        overlaps: std.atomic.Value(u32) = .init(0),
        bad_workers: std.atomic.Value(u32) = .init(0),
        workers: usize = 0,

        fn job(check: *@This(), index: usize, worker: usize) void {
            if (worker >= check.workers) _ = check.bad_workers.fetchAdd(1, .monotonic);
            if (check.busy[worker].swap(true, .acquire)) _ = check.overlaps.fetchAdd(1, .monotonic);
            _ = check.runs[index].fetchAdd(1, .monotonic);
            check.busy[worker].store(false, .release);
        }
    };
    for (0..2000) |round| {
        const jobs = round % 64 + 1;
        var check: Check = .{ .workers = pool.workers(jobs) };
        pool.run(jobs, &check, Check.job);
        for (check.runs[0..jobs]) |runs| try testing.expectEqual(1, runs.raw);
        try testing.expectEqual(0, check.overlaps.raw);
        try testing.expectEqual(0, check.bad_workers.raw);
    }
}

test linear {
    const pool: Pool = try .init(testing.allocator, testing.io);
    defer pool.deinit(testing.allocator);
    var out: [4]f32 = undefined;
    linear(pool, &.{ 1, 2, 3, 4 }, &.{ 1, 0, 2, 3 }, &.{ 1, -1 }, &out, 2, 2, .replace);
    try testing.expectEqualSlices(f32, &.{ 2, 7, 4, 17 }, &out);
    linear(pool, &.{ 1, 2, 3, 4 }, &.{ 1, 0, 2, 3 }, null, &out, 2, 2, .add);
    try testing.expectEqualSlices(f32, &.{ 3, 15, 7, 35 }, &out);
}

test "linear matches a naive product across panels, tiles, and workers" {
    const gpa = testing.allocator;
    // Sizes that leave partial tiles, partial blocks, and outputs past the last panel.
    const rows = 53;
    const inputs = 301;
    const outputs = 3 * panel_width + 5;
    var prng: std.Random.DefaultPrng = .init(1);
    const random = prng.random();

    const x = try gpa.alloc(f32, rows * inputs);
    defer gpa.free(x);
    const weight = try gpa.alloc(f32, outputs * inputs);
    defer gpa.free(weight);
    const bias = try gpa.alloc(f32, outputs);
    defer gpa.free(bias);
    for (x) |*v| v.* = random.float(f32) - 0.5;
    for (weight) |*v| v.* = random.float(f32) - 0.5;
    for (bias) |*v| v.* = random.float(f32) - 0.5;

    const expected = try gpa.alloc(f32, rows * outputs);
    defer gpa.free(expected);
    for (0..rows) |r| for (0..outputs) |o| {
        var sum: f64 = bias[o];
        for (0..inputs) |k| sum += @as(f64, x[r * inputs + k]) * weight[o * inputs + k];
        expected[r * outputs + o] = @floatCast(sum);
    };

    const pool: Pool = try .init(gpa, testing.io);
    defer pool.deinit(gpa);
    const buffer = try gpa.alloc(f32, packBufferLen(pool, inputs));
    defer gpa.free(buffer);
    packWeights(pool, weight, inputs, buffer);
    const out = try gpa.alloc(f32, rows * outputs);
    defer gpa.free(out);
    linear(pool, x, weight, bias, out, inputs, outputs, .replace);
    for (expected, out) |want, got| try testing.expectApproxEqAbs(want, got, 1e-4);
    linear(pool, x, weight, bias, out, inputs, outputs, .add);
    for (expected, out) |want, got| try testing.expectApproxEqAbs(2 * want, got, 2e-4);

    // A worker that doesn't use BLAS must still compute every panel when it runs alone.
    if (build_options.blas) {
        @memset(out, 0);
        const gemm: Gemm = .{ .x = x, .weight = weight, .bias = bias, .out = out, .rows = rows, .inputs = inputs, .outputs = outputs, .mode = .add };
        var split: Gemm.Split = .init(&gemm);
        split.blas_panels = 1;
        split.work(0, Gemm.Split.blas_workers);
        for (expected, out) |want, got| try testing.expectApproxEqAbs(want, got, 1e-4);
    }
}

test attention {
    const pool: Pool = try .init(testing.allocator, testing.io);
    defer pool.deinit(testing.allocator);
    var qkv = [_]f32{ 0, 0, 2, 0, 0, 4, 0, 0, 9 };
    var out: [3]f32 = undefined;
    var scratch: [1024]f32 = undefined;

    attention(pool, &qkv, &out, &scratch, 3, 1, 1, 0, null);
    try testing.expectEqualSlices(f32, &.{ 2, 4, 9 }, &out);

    attention(pool, &qkv, &out, &scratch, 3, 1, 1, 1, null);
    try testing.expectApproxEqAbs(3, out[0], 1e-6);
    try testing.expectApproxEqAbs(5, out[1], 1e-6);
    try testing.expectApproxEqAbs(6.5, out[2], 1e-6);
}

test "attention matches a naive computation" {
    const gpa = testing.allocator;
    const pool: Pool = try .init(gpa, testing.io);
    defer pool.deinit(gpa);
    const seq = 61;
    const heads = 2;
    const dim = heads * 64;
    var prng: std.Random.DefaultPrng = .init(2);
    const random = prng.random();

    const qkv = try gpa.alloc(f32, seq * 3 * dim);
    defer gpa.free(qkv);
    for (qkv) |*v| v.* = random.float(f32) - 0.5;
    const out = try gpa.alloc(f32, seq * dim);
    defer gpa.free(out);
    const expected = try gpa.alloc(f32, seq * dim);
    defer gpa.free(expected);
    const scratch = try gpa.alloc(f32, attentionScratchLen(pool, seq, dim, heads));
    defer gpa.free(scratch);

    for ([_]?usize{ null, 7 }) |window| {
        attention(pool, qkv, out, scratch, seq, dim, heads, window, null);
        for (0..heads) |h| for (0..seq) |t| {
            const lo, const hi = span(t, seq, window);
            var weights: [seq]f64 = undefined;
            var total: f64 = 0;
            for (lo..hi) |j| {
                var s: f64 = 0;
                for (0..64) |i| s += @as(f64, qkv[t * 3 * dim + h * 64 + i]) * qkv[j * 3 * dim + dim + h * 64 + i];
                weights[j] = @exp(s / 8);
                total += weights[j];
            }
            for (0..64) |i| {
                var sum: f64 = 0;
                for (lo..hi) |j| sum += weights[j] / total * qkv[j * 3 * dim + 2 * dim + h * 64 + i];
                expected[t * dim + h * 64 + i] = @floatCast(sum);
            }
        };
        for (expected, out) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
    }
}

extern "c" fn erf(f64) f64;

test "vector exp and gelu stay close to libm" {
    var x: f32 = -10;
    while (x < 10) : (x += 0.001) {
        const v: V = @splat(x);
        const wide: f64 = x;
        try testing.expectApproxEqRel(@exp(wide), exp(v)[0], 3e-7);
        try testing.expectApproxEqAbs(0.5 * wide * (1 + erf(wide * std.math.sqrt1_2)), geluVector(v)[0], 1e-6);
    }
    try testing.expectEqual(0, exp(@splat(-100))[0]);
    try testing.expect(std.math.isNan(exp(@splat(std.math.nan(f32)))[0]));
    try testing.expect(std.math.isNan(geluVector(@splat(std.math.nan(f32)))[0]));
}

test softmax {
    var values = [_]f32{ 1, 2, 3, 4, 5, -1e30 };
    softmax(&values);
    const exps = [_]f64{ @exp(1.0), @exp(2.0), @exp(3.0), @exp(4.0), @exp(5.0) };
    var total: f64 = 0;
    for (exps) |e| total += e;
    for (values[0..5], exps) |p, e| try testing.expectApproxEqRel(e / total, p, 1e-6);
    try testing.expectEqual(0, values[5]);
}
