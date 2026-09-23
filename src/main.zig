const std = @import("std");
const zlaya = @import("zlaya");

const usage = "Usage: zlaya MODEL_DIRECTORY REQUEST.json [--raw]\n";

pub fn main(init: std.process.Init) void {
    run(init) catch |err| std.process.fatal("{t}", .{err});
}

fn run(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print(usage, .{});
        return;
    }
    if (args.len < 3 or args.len > 4) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    const raw = args.len == 4 and std.mem.eql(u8, args[3], "--raw");
    if (args.len == 4 and !raw) return error.InvalidArguments;

    const request_json = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], arena, .limited(16 * 1024 * 1024));
    const request = try std.json.parseFromSliceLeaky(std.json.Value, arena, request_json, .{});

    var engine: zlaya.Engine = try .load(init.gpa, init.io, args[1]);
    defer engine.deinit();
    const response = try engine.predict(arena, request, raw);

    var buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const stdout = &stdout_writer.interface;
    try std.json.Stringify.value(response, .{ .whitespace = .indent_2 }, stdout);
    try stdout.writeByte('\n');
    try stdout.flush();
}
