//! Turns a request into model inputs, following the upstream Laya prompt format.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const testing = std.testing;
const Tokenizer = @import("Tokenizer.zig");
const QuestionType = @import("Model.zig").QuestionType;

/// Most tokens kept from each option, not counting its mask token.
const max_option_tokens = 48;
/// When the options are too long, they are shortened to leave this many tokens for the question.
const question_reserve = 16;
const min_option_tokens = 4;
const min_question_tokens = 8;

pub const Sequence = struct {
    ids: []u32,
    /// Position of each option's mask token in `ids`.
    markers: []usize,
    qtype: QuestionType,
    /// Answer label of each option.
    labels: [][]const u8,
};

/// A tokenized request state, shared by every question of the request.
pub const State = struct {
    ids: []const u32,
    /// Conversations keep their latest turns when truncated.
    keep_tail: bool,

    pub fn encode(arena: Allocator, tokenizer: *const Tokenizer, state: Value) !State {
        return .{
            .ids = try encodeText(arena, tokenizer, try renderValue(arena, state)),
            .keep_tail = state == .array,
        };
    }
};

/// Lays out one question as `[CLS] <type> instructions [SEP] [MASK] option... [SEP] state [SEP]`.
pub fn build(
    arena: Allocator,
    tokenizer: *const Tokenizer,
    state: State,
    question: Value,
    max_len: usize,
    head_max_len: usize,
) !Sequence {
    if (max_len < 4 or head_max_len < question_reserve) return error.InvalidSequenceLimit;
    const rendered = try render(arena, question);
    const prompt = try std.fmt.allocPrint(arena, "{t} question: {s}", .{ rendered.qtype, rendered.instructions });
    const head = try encodeText(arena, tokenizer, prompt);

    const options = try arena.alloc([]u32, rendered.options.len);
    var total: usize = 0;
    for (rendered.options, options) |option, *ids| {
        const encoded = try encodeText(arena, tokenizer, try std.mem.concat(arena, u8, &.{ " ", option }));
        const kept = encoded[0..@min(max_option_tokens, encoded.len)];
        ids.* = try std.mem.concat(arena, u32, &.{ &.{Tokenizer.mask_id}, kept });
        total += ids.len;
    }
    if (total + question_reserve > head_max_len) {
        const per_option = @max(min_option_tokens, (head_max_len - question_reserve) / options.len);
        total = 0;
        for (options) |*ids| {
            ids.* = ids.*[0..@min(ids.len, per_option)];
            total += ids.len;
        }
    }
    const head_budget = @max(min_question_tokens, head_max_len -| total);

    var ids: std.ArrayList(u32) = .empty;
    var markers: std.ArrayList(usize) = .empty;
    try ids.append(arena, Tokenizer.cls_id);
    try ids.appendSlice(arena, head[0..@min(head.len, head_budget)]);
    try ids.append(arena, Tokenizer.sep_id);
    for (options) |option| {
        try markers.append(arena, ids.items.len);
        try ids.appendSlice(arena, option);
    }
    try ids.append(arena, Tokenizer.sep_id);

    const room = max_len -| (ids.items.len + 1);
    const state_ids = if (state.keep_tail)
        state.ids[state.ids.len -| room..]
    else
        state.ids[0..@min(room, state.ids.len)];
    try ids.appendSlice(arena, state_ids);
    try ids.append(arena, Tokenizer.sep_id);

    if (ids.items.len > max_len) ids.shrinkRetainingCapacity(max_len);
    for (markers.items) |marker| if (marker >= ids.items.len) return error.TooManyOptions;
    return .{
        .ids = try ids.toOwnedSlice(arena),
        .markers = try markers.toOwnedSlice(arena),
        .qtype = rendered.qtype,
        .labels = rendered.labels,
    };
}

const Rendered = struct {
    qtype: QuestionType,
    instructions: []const u8,
    options: [][]const u8,
    labels: [][]const u8,
};

fn isEmpty(value: Value) bool {
    return value == .null or (value == .string and value.string.len == 0);
}

fn render(arena: Allocator, question: Value) !Rendered {
    if (question != .object) return error.InvalidQuestion;
    const object = question.object;
    const type_name = object.get("type") orelse return error.MissingQuestionType;
    if (type_name != .string) return error.InvalidQuestionType;
    const qtype = std.meta.stringToEnum(QuestionType, type_name.string) orelse return error.InvalidQuestionType;
    const instructions = try renderValue(arena, object.get("instructions") orelse return error.MissingInstructions);
    const criteria = object.get("criteria") orelse .null;
    if (qtype != .noul and object.contains("labels")) return error.InvalidLabels;

    var options: std.ArrayList([]const u8) = .empty;
    var labels: std.ArrayList([]const u8) = .empty;
    switch (qtype) {
        .choice => switch (criteria) {
            .object => |descriptions| {
                var it = descriptions.iterator();
                while (it.next()) |entry| {
                    const label = entry.key_ptr.*;
                    const description = entry.value_ptr.*;
                    try labels.append(arena, label);
                    if (isEmpty(description)) {
                        try options.append(arena, label);
                    } else {
                        const text = try renderValue(arena, description);
                        try options.append(arena, try std.fmt.allocPrint(arena, "{s}: {s}", .{ label, text }));
                    }
                }
            },
            .array => |items| for (items.items) |item| {
                if (item != .string) return error.InvalidChoiceLabel;
                for (labels.items) |label| {
                    if (std.mem.eql(u8, label, item.string)) break;
                } else {
                    try labels.append(arena, item.string);
                    try options.append(arena, item.string);
                }
            },
            else => return error.InvalidCriteria,
        },
        .score => {
            if (criteria != .array) return error.InvalidCriteria;
            for (criteria.array.items, 0..) |item, level| {
                try labels.append(arena, try std.fmt.allocPrint(arena, "{d}", .{level}));
                const text = try renderValue(arena, item);
                try options.append(arena, try std.fmt.allocPrint(arena, "level {d}: {s}", .{ level, text }));
            }
        },
        .noul => {
            if (criteria != .null and criteria != .object) return error.InvalidCriteria;
            var descriptions: [2]Value = .{ .null, .null };
            if (criteria == .object) {
                var it = criteria.object.iterator();
                while (it.next()) |entry| {
                    const index: usize = if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, "false"))
                        0
                    else if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, "true"))
                        1
                    else
                        return error.InvalidCriteria;
                    descriptions[index] = entry.value_ptr.*;
                }
            }

            var names: [2][]const u8 = .{ "false", "true" };
            if (object.get("labels")) |custom| {
                if (custom != .null) {
                    if (custom != .object or custom.object.count() != 2) return error.InvalidLabels;
                    for (&names) |*name| {
                        const value = custom.object.get(name.*) orelse return error.InvalidLabels;
                        if (value != .string) return error.InvalidLabels;
                        name.* = try trimLabel(value.string);
                        if (name.len == 0) return error.InvalidLabels;
                    }
                    if (std.mem.eql(u8, names[0], names[1])) return error.InvalidLabels;
                }
            }

            const defaults = [_][]const u8{ "no, the statement does not hold", "yes, the statement holds" };
            for (names, descriptions, defaults) |name, description, default| {
                const text = if (isEmpty(description)) default else try renderValue(arena, description);
                try labels.append(arena, name);
                try options.append(arena, try std.fmt.allocPrint(arena, "{s}: {s}", .{ name, text }));
            }
        },
    }
    if (options.items.len == 0) return error.EmptyCriteria;

    return .{
        .qtype = qtype,
        .instructions = instructions,
        .options = try options.toOwnedSlice(arena),
        .labels = try labels.toOwnedSlice(arena),
    };
}

/// Strips whitespace like Python's `str.strip`.
fn trimLabel(text: []const u8) ![]const u8 {
    const view = std.unicode.Utf8View.init(text) catch return error.InvalidLabels;
    var it = view.iterator();
    var start: ?usize = null;
    var end: usize = 0;
    while (true) {
        const pos = it.i;
        const cp = it.nextCodepoint() orelse break;
        const is_space = switch (cp) {
            0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
            else => false,
        };
        if (!is_space) {
            if (start == null) start = pos;
            end = it.i;
        }
    }
    return text[start orelse end .. end];
}

fn sanitize(arena: Allocator, text: []const u8) ![]const u8 {
    return std.mem.replaceOwned(u8, arena, text, Tokenizer.mask_token, " ");
}

fn encodeText(arena: Allocator, tokenizer: *const Tokenizer, text: []const u8) ![]u32 {
    return tokenizer.encodeRaw(arena, try sanitize(arena, text));
}

/// Renders non-string values the way Python's `json.dumps` spells them.
fn renderValue(arena: Allocator, value: Value) ![]const u8 {
    if (value == .string) return value.string;
    var out: std.Io.Writer.Allocating = .init(arena);
    writeJson(&out.writer, value) catch return error.OutOfMemory;
    return out.written();
}

fn writeJson(w: *std.Io.Writer, value: Value) std.Io.Writer.Error!void {
    switch (value) {
        .array => |array| {
            try w.writeByte('[');
            for (array.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try writeJson(w, item);
            }
            try w.writeByte(']');
        },
        .object => |object| {
            try w.writeByte('{');
            for (object.keys(), object.values(), 0..) |key, item, i| {
                if (i > 0) try w.writeAll(", ");
                try std.json.Stringify.encodeJsonString(key, .{}, w);
                try w.writeAll(": ");
                try writeJson(w, item);
            }
            try w.writeByte('}');
        },
        .float => |number| try writePythonFloat(w, number),
        else => try std.json.Stringify.value(value, .{}, w),
    }
}

/// Writes `number` like Python's `repr`, which uses scientific notation outside [1e-4, 1e16).
fn writePythonFloat(w: *std.Io.Writer, number: f64) std.Io.Writer.Error!void {
    var buffer: [64]u8 = undefined;
    const magnitude = @abs(number);
    if (magnitude == 0 or (magnitude >= 1e-4 and magnitude < 1e16)) {
        const text = std.fmt.bufPrint(&buffer, "{d}", .{number}) catch unreachable;
        try w.writeAll(text);
        if (std.mem.findScalar(u8, text, '.') == null) try w.writeAll(".0");
        return;
    }

    // Python always signs the exponent and pads it to two digits.
    const text = std.fmt.bufPrint(&buffer, "{e}", .{number}) catch unreachable;
    const e = std.mem.findScalar(u8, text, 'e').?;
    try w.writeAll(text[0 .. e + 1]);
    var exponent = text[e + 1 ..];
    if (exponent[0] == '-') {
        try w.writeByte('-');
        exponent = exponent[1..];
    } else {
        try w.writeByte('+');
        if (exponent[0] == '+') exponent = exponent[1..];
    }
    if (exponent.len < 2) try w.writeByte('0');
    try w.writeAll(exponent);
}

test "structured criteria rendering and noul label order" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();
    const question = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"type":"noul","instructions":{"x":"café"},"criteria":{"TRUE":{"value":true},"false":0},"labels":{"true":"yes","false":"no"}}
    , .{});

    const rendered = try render(arena, question);
    try testing.expectEqualStrings("{\"x\": \"café\"}", rendered.instructions);
    try testing.expectEqualStrings("no: 0", rendered.options[0]);
    try testing.expectEqualStrings("yes: {\"value\": true}", rendered.options[1]);
    try testing.expectEqualStrings("no", rendered.labels[0]);
    try testing.expectEqualStrings("yes", rendered.labels[1]);
}

test "choice list deduplicates and invalid questions are rejected" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const question = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"type":"choice","instructions":"pick","criteria":["a","b","a"]}
    , .{});
    const rendered = try render(arena, question);
    try testing.expectEqual(2, rendered.options.len);

    try testing.expectError(error.InvalidQuestion, render(arena, .null));
    const empty = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"type":"score","instructions":"pick","criteria":[]}
    , .{});
    try testing.expectError(error.EmptyCriteria, render(arena, empty));
}

test writePythonFloat {
    const cases = .{
        .{ 1.0, "1.0" },
        .{ -0.0, "-0.0" },
        .{ 1e-5, "1e-05" },
        .{ 1e16, "1e+16" },
        .{ 0.0001, "0.0001" },
    };
    inline for (cases) |case| {
        var buffer: [64]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buffer);
        try writePythonFloat(&w, case[0]);
        try testing.expectEqualStrings(case[1], w.buffered());
    }
}

test "mask sanitation and Unicode label whitespace" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();
    try testing.expectEqualStrings("a b c", try sanitize(arena, "a[MASK]b[MASK]c"));
    try testing.expectEqualStrings("yes", try trimLabel("\u{2003}yes\u{a0}"));
    try testing.expectEqualStrings("", try trimLabel("\u{3000}\n"));
    try testing.expectError(error.InvalidLabels, trimLabel("\xff"));
}
