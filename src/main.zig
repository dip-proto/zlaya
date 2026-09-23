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

    // The progress display has to be gone before the response is written.
    const response = response: {
        const progress = std.Progress.start(init.io, .{});
        defer progress.end();
        // std.Progress can't draw on WASI, in single-threaded builds, or without escape codes.
        const plain = progress.index == .none and try std.Io.File.stderr().isTty(init.io);

        if (plain) std.debug.print("Loading model from {s}...\n", .{args[1]});
        var engine: zlaya.Engine = try .load(init.gpa, init.io, args[1], progress);
        defer engine.deinit();
        if (plain) std.debug.print("Answering questions...\n", .{});
        break :response try engine.predict(arena, request, raw, progress);
    };

    var buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const stdout = &stdout_writer.interface;
    try std.json.Stringify.value(response, .{ .whitespace = .indent_2 }, stdout);
    try stdout.writeByte('\n');
    try stdout.flush();
}
