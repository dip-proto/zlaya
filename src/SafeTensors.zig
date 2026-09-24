//! Reader for safetensors checkpoints.
//! Only the header is parsed up front.
//! Tensor data stays where it is until `toF32` converts it.
const SafeTensors = @This();

const std = @import("std");
const builtin = @import("builtin");
const native_endian = builtin.cpu.arch.endian();
const Allocator = std.mem.Allocator;
const testing = std.testing;

arena: std.heap.ArenaAllocator,
tensors: std.StringHashMapUnmanaged(Tensor),
source: Source,

pub const DType = enum {
    BF16,
    F16,
    F32,

    pub fn byteSize(dtype: DType) usize {
        return switch (dtype) {
            .BF16, .F16 => 2,
            .F32 => 4,
        };
    }
};

pub const Tensor = struct {
    dtype: DType,
    shape: []const usize,
    /// Byte range within the data section.
    start: usize,
    end: usize,

    pub fn len(tensor: Tensor) usize {
        return (tensor.end - tensor.start) / tensor.dtype.byteSize();
    }
};

const Source = union(enum) {
    bytes: []const u8,
    file: struct { io: std.Io, file: std.Io.File, offset: u64 },
};

/// `bytes` holds the whole file and must outlive the result.
pub fn init(gpa: Allocator, bytes: []const u8) !SafeTensors {
    if (bytes.len < 8) return error.InvalidHeader;
    const header_len = try headerLength(bytes[0..8], bytes.len);
    const data = bytes[8 + header_len ..];
    return parse(gpa, bytes[8..][0..header_len], .{ .bytes = data }, data.len);
}

/// Reads only the header, which keeps memory use low.
/// `file` must stay open until every tensor has been converted.
pub fn open(gpa: Allocator, io: std.Io, file: std.Io.File) !SafeTensors {
    const size = std.math.cast(usize, try file.length(io)) orelse return error.FileTooBig;
    var prefix: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &prefix, 0) != prefix.len) return error.InvalidHeader;
    const header_len = try headerLength(&prefix, size);

    const header = try gpa.alloc(u8, header_len);
    defer gpa.free(header);
    if (try file.readPositionalAll(io, header, 8) != header.len) return error.InvalidHeader;

    const source: Source = .{ .file = .{ .io = io, .file = file, .offset = 8 + header_len } };
    return parse(gpa, header, source, size - 8 - header_len);
}

pub fn deinit(st: *SafeTensors) void {
    st.arena.deinit();
    st.* = undefined;
}

pub fn tensorCount(st: *const SafeTensors) usize {
    return st.tensors.count();
}

/// Number of float32 values needed to convert every tensor.
pub fn totalLen(st: *const SafeTensors) usize {
    var total: usize = 0;
    var it = st.tensors.valueIterator();
    while (it.next()) |tensor| total += tensor.len();
    return total;
}

/// The shape of the returned tensor is valid until `deinit`.
pub fn get(st: *const SafeTensors, name: []const u8) ?Tensor {
    return st.tensors.get(name);
}

/// Converts `tensor` to a new float32 slice owned by the caller.
pub fn toF32(st: *const SafeTensors, gpa: Allocator, tensor: Tensor) ![]f32 {
    const result = try gpa.alloc(f32, tensor.len());
    errdefer gpa.free(result);
    const bytes = std.mem.sliceAsBytes(result);
    const raw = bytes[bytes.len - (tensor.end - tensor.start) ..];
    switch (st.source) {
        .bytes => |data| @memcpy(raw, data[tensor.start..tensor.end]),
        .file => |source| {
            // wasmtime rejects reads of a few hundred megabytes.
            var done: usize = 0;
            while (done < raw.len) {
                const chunk = raw[done..][0..@min(16 * 1024 * 1024, raw.len - done)];
                if (try source.file.readPositionalAll(source.io, chunk, source.offset + tensor.start + done) != chunk.len) {
                    return error.EndOfStream;
                }
                done += chunk.len;
            }
        },
    }
    widen(tensor.dtype, result);
    return result;
}

fn headerLength(prefix: *const [8]u8, file_size: usize) !usize {
    const header_len = std.mem.readInt(u64, prefix, .little);
    if (header_len == 0 or header_len > 100_000_000 or header_len > file_size - 8) return error.InvalidHeader;
    return @intCast(header_len);
}

fn parse(gpa: Allocator, header: []const u8, source: Source, data_len: usize) !SafeTensors {
    if (header[0] != '{') return error.InvalidHeader;
    var arena_instance: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, header, .{ .allocate = .alloc_always });
    if (root != .object) return error.InvalidHeader;
    var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
    var ranges: std.ArrayList(Range) = .empty;
    var entries = root.object.iterator();
    while (entries.next()) |entry| {
        const name = entry.key_ptr.*;
        const value = entry.value_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) {
            if (value != .object) return error.InvalidMetadata;
            for (value.object.values()) |item| if (item != .string) return error.InvalidMetadata;
            continue;
        }
        const tensor = try parseTensor(arena, value, data_len);
        try ranges.append(arena, .{ .start = tensor.start, .end = tensor.end });
        try tensors.put(arena, name, tensor);
    }

    // Tensors must cover the data section without gaps or overlaps.
    std.mem.sort(Range, ranges.items, {}, Range.lessThan);
    var next: usize = 0;
    for (ranges.items) |range| {
        if (range.start != next) return error.InvalidOffsets;
        next = range.end;
    }
    if (next != data_len) return error.InvalidOffsets;

    return .{ .arena = arena_instance, .tensors = tensors, .source = source };
}

fn parseTensor(arena: Allocator, value: std.json.Value, data_len: usize) !Tensor {
    if (value != .object) return error.InvalidTensor;

    const dtype_name = value.object.get("dtype") orelse return error.InvalidTensor;
    if (dtype_name != .string) return error.InvalidTensor;
    const dtype = std.meta.stringToEnum(DType, dtype_name.string) orelse return error.UnsupportedDType;

    const shape_json = value.object.get("shape") orelse return error.InvalidTensor;
    if (shape_json != .array) return error.InvalidShape;
    const shape = try arena.alloc(usize, shape_json.array.items.len);
    for (shape, shape_json.array.items) |*dim, item| dim.* = try unsigned(item);

    // A zero dimension makes the tensor empty, even if the other dimensions would overflow.
    var elements: usize = 0;
    if (std.mem.findScalar(usize, shape, 0) == null) {
        elements = 1;
        for (shape) |dim| elements = std.math.mul(usize, elements, dim) catch return error.InvalidShape;
    }
    const size = std.math.mul(usize, elements, dtype.byteSize()) catch return error.InvalidShape;

    const offsets = value.object.get("data_offsets") orelse return error.InvalidTensor;
    if (offsets != .array or offsets.array.items.len != 2) return error.InvalidOffsets;
    const start = try unsigned(offsets.array.items[0]);
    const end = try unsigned(offsets.array.items[1]);
    if (start > end or end > data_len or end - start != size) return error.InvalidOffsets;

    return .{ .dtype = dtype, .shape = shape, .start = start, .end = end };
}

const Range = struct {
    start: usize,
    end: usize,

    fn lessThan(_: void, a: Range, b: Range) bool {
        return a.start < b.start or (a.start == b.start and a.end < b.end);
    }
};

fn unsigned(value: std.json.Value) !usize {
    if (value != .integer) return error.InvalidInteger;
    return std.math.cast(usize, value.integer) orelse error.InvalidInteger;
}

/// Turns the raw little-endian values stored at the end of `values` into floats, in place.
/// Working from the front is safe, because each step only writes over values it has already read.
fn widen(dtype: DType, values: []f32) void {
    const bytes = std.mem.sliceAsBytes(values);
    const raw = bytes[bytes.len - values.len * dtype.byteSize() ..];
    switch (dtype) {
        .F32 => if (native_endian == .big) std.mem.byteSwapAllElements(u32, @ptrCast(values)),
        inline .BF16, .F16 => |t| {
            const lanes = 8;
            var i: usize = 0;
            while (i + lanes <= values.len) : (i += lanes) {
                const halves = std.mem.littleToNative(@Vector(lanes, u16), @bitCast(raw[2 * i ..][0 .. 2 * lanes].*));
                values[i..][0..lanes].* = widenHalves(t, lanes, halves);
            }
            for (values[i..], i..) |*value, j| {
                value.* = widenHalves(t, 1, .{std.mem.readInt(u16, raw[2 * j ..][0..2], .little)})[0];
            }
        },
    }
}

fn widenHalves(comptime dtype: DType, comptime lanes: usize, halves: @Vector(lanes, u16)) @Vector(lanes, f32) {
    if (dtype == .F16) return halfToSingle(lanes, halves);
    const bits: @Vector(lanes, u32) = halves;
    return @bitCast(bits << @splat(16));
}

/// Only AArch64 is sure to convert half floats with a single instruction.
/// Elsewhere, `@floatCast` from `f16` can call a function for every value.
const native_half = builtin.cpu.arch.isAARCH64();

fn halfToSingle(comptime lanes: usize, half: @Vector(lanes, u16)) @Vector(lanes, f32) {
    if (!native_half) return widenHalfBits(lanes, half);
    const floats: @Vector(lanes, f16) = @bitCast(half);
    return @floatCast(floats);
}

/// Widens half floats exactly, with integer and single-precision operations only.
fn widenHalfBits(comptime lanes: usize, half: @Vector(lanes, u16)) @Vector(lanes, f32) {
    const U = @Vector(lanes, u32);
    const F = @Vector(lanes, f32);
    const bits: U = half;
    const unscaled: F = @bitCast((bits & @as(U, @splat(0x7fff))) << @splat(13));
    // Multiplying by 2^112 fixes the exponent bias, and turns subnormals into normal numbers.
    const scaled = unscaled * @as(F, @splat(0x1p112));
    const magnitude: U = @bitCast(scaled);
    const infinite_or_nan = scaled >= @as(F, @splat(0x1p16));
    const widened = @select(u32, infinite_or_nan, magnitude | @as(U, @splat(0x7f80_0000)), magnitude);
    return @bitCast(widened | (bits & @as(U, @splat(0x8000))) << @splat(16));
}

fn fixture(gpa: Allocator, header: []const u8, data: []const u8) ![]u8 {
    const header_len = std.mem.toBytes(std.mem.nativeToLittle(u64, header.len));
    return std.mem.concat(gpa, u8, &.{ &header_len, header, data });
}

test "floating point tensor conversion and scalar shape" {
    const bytes = try fixture(testing.allocator,
        \\{"bf":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"half":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[8,12]}}
    , &.{ 0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3e, 0x00, 0xbc, 0x00, 0x00, 0x20, 0x40 });
    defer testing.allocator.free(bytes);
    var st: SafeTensors = try .init(testing.allocator, bytes);
    defer st.deinit();

    try testing.expectEqualSlices(usize, &.{2}, st.get("bf").?.shape);
    inline for (.{ .{ "bf", &.{ 1, -2 } }, .{ "half", &.{ 1.5, -1 } }, .{ "scalar", &.{2.5} } }) |case| {
        const converted = try st.toF32(testing.allocator, st.get(case[0]).?);
        defer testing.allocator.free(converted);
        try testing.expectEqualSlices(f32, case[1], converted);
    }
    try testing.expect(st.get("missing") == null);
}

test "tensors read from a file span several chunks" {
    const io = testing.io;
    var data: [2 * 100_003]u8 = undefined;
    for (0..data.len / 2) |i| {
        const value: f16 = @floatFromInt(i % 2048);
        std.mem.writeInt(u16, data[2 * i ..][0..2], @bitCast(value), .little);
    }
    const bytes = try fixture(testing.allocator,
        \\{"x":{"dtype":"F16","shape":[100003],"data_offsets":[0,200006]}}
    , &data);
    defer testing.allocator.free(bytes);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "x.safetensors", .data = bytes });
    const file = try tmp.dir.openFile(io, "x.safetensors", .{});
    defer file.close(io);
    var st: SafeTensors = try .open(testing.allocator, io, file);
    defer st.deinit();
    const converted = try st.toF32(testing.allocator, st.get("x").?);
    defer testing.allocator.free(converted);
    for (converted, 0..) |value, i| try testing.expectEqual(@as(f32, @floatFromInt(i % 2048)), value);

    try tmp.dir.writeFile(io, .{ .sub_path = "short.safetensors", .data = bytes[0..6] });
    const short = try tmp.dir.openFile(io, "short.safetensors", .{});
    defer short.close(io);
    try testing.expectError(error.InvalidHeader, open(testing.allocator, io, short));
}

test "reject invalid tensor ranges and shape overflow" {
    const headers = [_][]const u8{
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
        ,
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,5]}}
        ,
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}
        ,
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
        ,
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}
        ,
    };
    for (headers) |header| {
        const bytes = try fixture(testing.allocator, header, &.{ 0, 0, 0, 0 });
        defer testing.allocator.free(bytes);
        try testing.expectError(error.InvalidOffsets, init(testing.allocator, bytes));
    }

    const bytes = try fixture(testing.allocator,
        \\{"a":{"dtype":"F32","shape":[4294967295,4294967295,4294967295],"data_offsets":[0,0]}}
    , &.{});
    defer testing.allocator.free(bytes);
    try testing.expectError(error.InvalidShape, init(testing.allocator, bytes));
}

test "empty tensors and truncated headers" {
    const bytes = try fixture(testing.allocator,
        \\{"empty":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]}}
    , &.{});
    defer testing.allocator.free(bytes);
    var st: SafeTensors = try .init(testing.allocator, bytes);
    defer st.deinit();
    try testing.expectEqual(0, st.get("empty").?.len());
    try testing.expectError(error.InvalidHeader, init(testing.allocator, bytes[0..7]));
    try testing.expectError(error.InvalidHeader, init(testing.allocator, bytes[0 .. bytes.len - 1]));
}

test "reject duplicate names, invalid metadata and unsupported storage types" {
    const cases = .{
        .{
            \\{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}
            ,
            error.DuplicateField,
        },
        .{
            \\{"__metadata__":{"format":1}}
            ,
            error.InvalidMetadata,
        },
        .{
            \\{"a":{"dtype":"F64","shape":[0],"data_offsets":[0,0]}}
            ,
            error.UnsupportedDType,
        },
        .{
            \\{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}
            ,
            error.InvalidInteger,
        },
    };
    inline for (cases) |case| {
        const bytes = try fixture(testing.allocator, case[0], &.{});
        defer testing.allocator.free(bytes);
        try testing.expectError(case[1], init(testing.allocator, bytes));
    }
}

test "half floats widen exactly" {
    var half: u16 = 0;
    while (true) : (half += 1) {
        const value: f16 = @bitCast(half);
        const want: f32 = value;
        for ([_]f32{ halfToSingle(1, .{half})[0], widenHalfBits(1, .{half})[0] }) |got| {
            if (std.math.isNan(want)) {
                try testing.expect(std.math.isNan(got));
            } else {
                const want_bits: u32 = @bitCast(want);
                const got_bits: u32 = @bitCast(got);
                try testing.expectEqual(want_bits, got_bits);
            }
        }
        if (half == std.math.maxInt(u16)) break;
    }
}
