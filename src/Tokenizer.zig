//! Byte-level BPE tokenizer that gives the same tokens as the Hugging Face
//! tokenizer shipped with ModernBERT.
const Tokenizer = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const unicode = @import("unicode_data.zig");

vocab: std.StringHashMapUnmanaged(u32),
merges: std.AutoHashMapUnmanaged([2]u32, Merge),
/// Token of each byte, or `no_token` when the vocabulary lacks it.
byte_tokens: [256]u32,
added: []const AddedToken,

pub const cls_id: u32 = 50281;
pub const sep_id: u32 = 50282;
pub const mask_id: u32 = 50284;
pub const mask_token = "[MASK]";

const no_token = std.math.maxInt(u32);

const Merge = struct { rank: usize, id: u32 };
const AddedToken = struct { content: []const u8, id: u32, lstrip: bool };

/// Everything is allocated in `arena`, so `json` can be freed afterwards.
pub fn init(arena: Allocator, json: []const u8) !Tokenizer {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{ .allocate = .alloc_always });
    try validate(root);
    const model = root.object.get("model").?.object;

    const vocab_json = model.get("vocab").?.object;
    var vocab: std.StringHashMapUnmanaged(u32) = .empty;
    try vocab.ensureTotalCapacity(arena, @intCast(vocab_json.count()));
    for (vocab_json.keys(), vocab_json.values()) |token, id| {
        vocab.putAssumeCapacity(token, @intCast(id.integer));
    }

    const merges_json = model.get("merges").?.array.items;
    var merges: std.AutoHashMapUnmanaged([2]u32, Merge) = .empty;
    try merges.ensureTotalCapacity(arena, @intCast(merges_json.len));
    var merged: std.ArrayList(u8) = .empty;
    for (merges_json, 0..) |pair, rank| {
        const left = pair.array.items[0].string;
        const right = pair.array.items[1].string;
        merged.clearRetainingCapacity();
        try merged.appendSlice(arena, left);
        try merged.appendSlice(arena, right);
        const key: [2]u32 = .{
            vocab.get(left) orelse return error.InvalidVocabulary,
            vocab.get(right) orelse return error.InvalidVocabulary,
        };
        const id = vocab.get(merged.items) orelse return error.InvalidVocabulary;
        merges.putAssumeCapacity(key, .{ .rank = rank, .id = id });
    }

    // Like GPT-2, the vocabulary spells every byte as a printable code point.
    var byte_tokens: [256]u32 = undefined;
    var next_unprintable: u21 = 256;
    for (&byte_tokens, 0..) |*id, i| {
        const byte: u8 = @intCast(i);
        const cp: u21 = switch (byte) {
            '!'...'~', 0xa1...0xac, 0xae...0xff => byte,
            else => blk: {
                const unprintable = next_unprintable;
                next_unprintable += 1;
                break :blk unprintable;
            },
        };
        var buf: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(cp, &buf);
        id.* = vocab.get(buf[0..len]) orelse no_token;
    }

    const added_json = root.object.get("added_tokens").?.array.items;
    const added = try arena.alloc(AddedToken, added_json.len);
    for (added, added_json) |*token, value| {
        token.* = .{
            .content = value.object.get("content").?.string,
            .id = @intCast(value.object.get("id").?.integer),
            .lstrip = value.object.get("lstrip").?.bool,
        };
    }

    return .{ .vocab = vocab, .merges = merges, .byte_tokens = byte_tokens, .added = added };
}

/// Encodes `text` between [CLS] and [SEP].
pub fn encode(tokenizer: *const Tokenizer, gpa: Allocator, text: []const u8) ![]u32 {
    const ids = try tokenizer.encodeRaw(gpa, text);
    defer gpa.free(ids);
    return std.mem.concat(gpa, u32, &.{ &.{cls_id}, ids, &.{sep_id} });
}

/// Encodes `text` without adding [CLS] and [SEP].
pub fn encodeRaw(tokenizer: *const Tokenizer, gpa: Allocator, text: []const u8) ![]u32 {
    const normalized = try normalize(gpa, text);
    defer gpa.free(normalized);

    // Remember where each added token occurs next, so the text is searched only once per token.
    const upcoming = try gpa.alloc(?Occurrence, tokenizer.added.len);
    defer gpa.free(upcoming);
    for (upcoming, tokenizer.added) |*occurrence, token| occurrence.* = find(normalized, 0, token);

    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    var pos: usize = 0;
    while (pos < normalized.len) {
        var best_len: usize = 0;
        var best_id: u32 = 0;
        var next = normalized.len;
        for (upcoming, tokenizer.added) |*slot, token| {
            if (slot.*) |occurrence| {
                if (occurrence.start < pos) slot.* = find(normalized, pos, token);
            }
            const occurrence = slot.* orelse continue;
            const begin = @max(occurrence.begin, pos);
            const end = occurrence.start + token.content.len;
            if (begin == pos and end - pos > best_len) {
                best_len = end - pos;
                best_id = token.id;
            }
            next = @min(next, begin);
        }

        if (best_len != 0) {
            try ids.append(gpa, best_id);
            pos += best_len;
        } else {
            const end = pos + pieceLength(normalized[pos..next]);
            try tokenizer.bpe(gpa, normalized[pos..end], &ids);
            pos = end;
        }
    }
    return ids.toOwnedSlice(gpa);
}

/// Appends the tokens of `piece` to `ids`.
fn bpe(tokenizer: *const Tokenizer, gpa: Allocator, piece: []const u8, ids: *std.ArrayList(u32)) !void {
    const start = ids.items.len;
    for (piece) |byte| {
        const id = tokenizer.byte_tokens[byte];
        if (id == no_token) return error.InvalidVocabulary;
        try ids.append(gpa, id);
    }

    while (ids.items.len - start > 1) {
        const tokens = ids.items[start..];
        var best: ?Merge = null;
        var best_index: usize = undefined;
        for (tokens[0 .. tokens.len - 1], tokens[1..], 0..) |left, right, i| {
            const candidate = tokenizer.merges.get(.{ left, right }) orelse continue;
            if (best == null or candidate.rank < best.?.rank) {
                best = candidate;
                best_index = i;
            }
        }
        const merge = best orelse break;
        tokens[best_index] = merge.id;
        _ = ids.orderedRemove(start + best_index + 1);
    }
}

const Occurrence = struct {
    start: usize,
    /// Where the match begins once lstrip absorbs the whitespace before it.
    begin: usize,
};

fn find(text: []const u8, from: usize, token: AddedToken) ?Occurrence {
    const start = std.mem.findPos(u8, text, from, token.content) orelse return null;
    if (!token.lstrip) return .{ .start = start, .begin = start };

    var begin = from;
    var it: std.unicode.Utf8Iterator = .{ .bytes = text[0..start], .i = from };
    while (it.nextCodepoint()) |cp| {
        if (category(cp) != .whitespace) begin = it.i;
    }
    return .{ .start = start, .begin = begin };
}

const contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

/// Returns the length of the next piece matched by the GPT-2 pre-tokenization pattern.
fn pieceLength(text: []const u8) usize {
    for (contractions) |contraction| {
        if (std.mem.startsWith(u8, text, contraction)) return contraction.len;
    }

    // A single space joins the piece after it, unless that piece is whitespace too.
    var it: std.unicode.Utf8Iterator = .{ .bytes = text, .i = 0 };
    if (text[0] == ' ' and text.len > 1) {
        it.i = 1;
        if (category(it.peekCodepoint().?) == .whitespace) it.i = 0;
    }
    const start = it.i;

    const kind = category(it.nextCodepoint().?);
    var end = it.i;
    var previous = start;
    while (it.nextCodepoint()) |cp| {
        if (category(cp) != kind) break;
        previous = end;
        end = it.i;
    }

    // Whitespace followed by other text leaves its last character to that text.
    if (kind == .whitespace and end < text.len and previous > start) return previous;
    return end;
}

fn category(cp: u21) unicode.Category {
    const index = std.sort.binarySearch(unicode.Range, &unicode.ranges, cp, compareRange) orelse return .other;
    return unicode.ranges[index].category;
}

fn compareRange(cp: u21, range: unicode.Range) std.math.Order {
    if (cp < range.first) return .lt;
    if (cp > range.last) return .gt;
    return .eq;
}

/// Finds the entry for `cp` in a table sorted by code point.
fn lookup(comptime T: type, table: []const T, cp: u21) ?T {
    const Compare = struct {
        fn byCodepoint(key: u21, entry: T) std.math.Order {
            return std.math.order(key, entry.codepoint);
        }
    };
    const index = std.sort.binarySearch(T, table, cp, Compare.byCodepoint) orelse return null;
    return table[index];
}

fn combiningClass(cp: u21) u8 {
    const entry = lookup(unicode.Combining, &unicode.combining, cp) orelse return 0;
    return entry.class;
}

fn lessCombiningClass(_: void, a: u21, b: u21) bool {
    return combiningClass(a) < combiningClass(b);
}

/// Constants of the Hangul syllable algorithm, named as in section 3.12 of the Unicode standard.
const hangul = struct {
    const s_base = 0xac00;
    const l_base = 0x1100;
    const v_base = 0x1161;
    const t_base = 0x11a7;
    const l_count = 19;
    const v_count = 21;
    const t_count = 28;
    const n_count = v_count * t_count;
    const s_count = l_count * n_count;
};

fn decompose(gpa: Allocator, out: *std.ArrayList(u21), cp: u21) Allocator.Error!void {
    if (cp >= hangul.s_base and cp < hangul.s_base + hangul.s_count) {
        const s = cp - hangul.s_base;
        try out.append(gpa, hangul.l_base + s / hangul.n_count);
        try out.append(gpa, hangul.v_base + (s % hangul.n_count) / hangul.t_count);
        if (s % hangul.t_count != 0) try out.append(gpa, hangul.t_base + s % hangul.t_count);
        return;
    }
    if (lookup(unicode.Decomposition, &unicode.decompositions, cp)) |decomposition| {
        try decompose(gpa, out, decomposition.first);
        if (decomposition.second != 0) try decompose(gpa, out, decomposition.second);
        return;
    }
    try out.append(gpa, cp);
}

fn compose(first: u21, second: u21) ?u21 {
    if (first >= hangul.l_base and first < hangul.l_base + hangul.l_count and
        second >= hangul.v_base and second < hangul.v_base + hangul.v_count)
    {
        const lv = (first - hangul.l_base) * hangul.n_count + (second - hangul.v_base) * hangul.t_count;
        return hangul.s_base + lv;
    }
    if (first >= hangul.s_base and first < hangul.s_base + hangul.s_count and
        (first - hangul.s_base) % hangul.t_count == 0 and
        second > hangul.t_base and second < hangul.t_base + hangul.t_count)
    {
        return first + (second - hangul.t_base);
    }

    const pair: [2]u21 = .{ first, second };
    const index = std.sort.binarySearch(unicode.Decomposition, &unicode.compositions, pair, comparePair) orelse return null;
    return unicode.compositions[index].codepoint;
}

fn comparePair(pair: [2]u21, entry: unicode.Decomposition) std.math.Order {
    return std.math.order(pair[0], entry.first).differ() orelse std.math.order(pair[1], entry.second);
}

/// Returns the NFC normalization of `text`.
fn normalize(gpa: Allocator, text: []const u8) ![]u8 {
    for (text) |c| {
        if (!std.ascii.isAscii(c)) break;
    } else return gpa.dupe(u8, text);

    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(gpa);
    var it = (try std.unicode.Utf8View.init(text)).iterator();
    while (it.nextCodepoint()) |cp| try decompose(gpa, &cps, cp);

    // Sort each run of combining marks by combining class.
    var run_start: usize = 0;
    for (cps.items, 0..) |cp, i| {
        if (combiningClass(cp) != 0) continue;
        std.sort.insertion(u21, cps.items[run_start..i], {}, lessCombiningClass);
        run_start = i + 1;
    }
    std.sort.insertion(u21, cps.items[run_start..], {}, lessCombiningClass);

    if (cps.items.len > 1) {
        var starter: usize = 0;
        var last_class: u8 = 0;
        var len: usize = 1;
        for (cps.items[1..]) |cp| {
            const class = combiningClass(cp);
            if (last_class == 0 or last_class < class) {
                if (compose(cps.items[starter], cp)) |composed| {
                    cps.items[starter] = composed;
                    continue;
                }
            }
            if (class == 0) starter = len;
            last_class = class;
            cps.items[len] = cp;
            len += 1;
        }
        cps.shrinkRetainingCapacity(len);
    }

    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(gpa);
    for (cps.items) |cp| {
        var buf: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(cp, &buf);
        try bytes.appendSlice(gpa, buf[0..len]);
    }
    return bytes.toOwnedSlice(gpa);
}

test normalize {
    const cases = [_][2][]const u8{
        .{ "e\xcc\x81", "é" },
        .{ "\u{1100}\u{1161}\u{11a8}", "\u{ac01}" },
        .{ "\u{ac01}", "\u{ac01}" },
        .{ "", "" },
        .{ "a\u{315}\u{300}", "à\u{315}" },
    };
    for (cases) |case| {
        const normalized = try normalize(testing.allocator, case[0]);
        defer testing.allocator.free(normalized);
        try testing.expectEqualStrings(case[1], normalized);
    }
}

test pieceLength {
    try testing.expectEqual(6, pieceLength(" hello world"));
    try testing.expectEqual(1, pieceLength("  hello"));
    try testing.expectEqual(3, pieceLength("'re here"));
    try testing.expectEqual(6, pieceLength(" café!"));
}

/// Returns the member `name` of an object after checking that it has type `tag`.
fn field(value: std.json.Value, name: []const u8, tag: std.meta.Tag(std.json.Value)) !std.json.Value {
    if (value != .object) return error.InvalidTokenizer;
    const result = value.object.get(name) orelse return error.InvalidTokenizer;
    if (std.meta.activeTag(result) != tag) return error.InvalidTokenizer;
    return result;
}

fn expectString(value: std.json.Value, name: []const u8, expected: []const u8) !void {
    const actual = try field(value, name, .string);
    if (!std.mem.eql(u8, actual.string, expected)) return error.UnsupportedTokenizer;
}

/// Rejects tokenizer settings that this implementation does not handle.
fn validate(root: std.json.Value) !void {
    try expectString(try field(root, "normalizer", .object), "type", "NFC");

    const pre_tokenizer = try field(root, "pre_tokenizer", .object);
    try expectString(pre_tokenizer, "type", "ByteLevel");
    if ((try field(pre_tokenizer, "add_prefix_space", .bool)).bool) return error.UnsupportedTokenizer;
    if (!(try field(pre_tokenizer, "use_regex", .bool)).bool) return error.UnsupportedTokenizer;

    _ = try field(root, "truncation", .null);
    _ = try field(root, "padding", .null);

    const post_processor = try field(root, "post_processor", .object);
    try expectString(post_processor, "type", "TemplateProcessing");
    const single = (try field(post_processor, "single", .array)).array.items;
    if (single.len != 3) return error.UnsupportedTokenizer;
    try expectString(try field(single[0], "SpecialToken", .object), "id", "[CLS]");
    try expectString(try field(single[1], "Sequence", .object), "id", "A");
    try expectString(try field(single[2], "SpecialToken", .object), "id", "[SEP]");

    const model = try field(root, "model", .object);
    try expectString(model, "type", "BPE");
    for ([_][]const u8{ "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix" }) |name| {
        _ = try field(model, name, .null);
    }
    for ([_][]const u8{ "fuse_unk", "byte_fallback", "ignore_merges" }) |name| {
        if ((try field(model, name, .bool)).bool) return error.UnsupportedTokenizer;
    }
    for ((try field(model, "vocab", .object)).object.values()) |id| {
        if (id != .integer or id.integer < 0 or id.integer >= no_token) return error.InvalidVocabulary;
    }
    for ((try field(model, "merges", .array)).array.items) |merge| {
        if (merge != .array or merge.array.items.len != 2) return error.InvalidTokenizer;
        for (merge.array.items) |part| if (part != .string) return error.InvalidTokenizer;
    }

    const added_tokens = (try field(root, "added_tokens", .array)).array.items;
    for (added_tokens) |token| {
        const content = (try field(token, "content", .string)).string;
        if (content.len == 0) return error.UnsupportedTokenizer;
        for (content) |byte| if (!std.ascii.isAscii(byte)) return error.UnsupportedTokenizer;
        _ = try field(token, "lstrip", .bool);
        if ((try field(token, "rstrip", .bool)).bool) return error.UnsupportedTokenizer;
        if ((try field(token, "single_word", .bool)).bool) return error.UnsupportedTokenizer;
        const id = (try field(token, "id", .integer)).integer;
        if (id < 0 or id >= no_token) return error.InvalidVocabulary;
    }

    const specials = [_]struct { []const u8, u32 }{
        .{ "[CLS]", cls_id },
        .{ "[SEP]", sep_id },
        .{ mask_token, mask_id },
    };
    for (specials) |special| {
        const content, const id = special;
        for (added_tokens) |token| {
            const object = token.object;
            if (std.mem.eql(u8, object.get("content").?.string, content) and object.get("id").?.integer == id) break;
        } else return error.UnsupportedTokenizer;
    }
}

test "malformed tokenizer configuration" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    try testing.expectError(error.InvalidTokenizer, init(arena.allocator(), "{}"));
    try testing.expectError(error.InvalidTokenizer, init(arena.allocator(), "[]"));
}
