const std = @import("std");

const Blas = enum { none, system };

pub fn build(b: *std.Build) void {
    var query = b.standardTargetOptionsQueryOnly(.{});
    // Enable SIMD on WebAssembly unless -Dcpu is given.
    // wasmtime, wasmer, and Node support both, and relaxed SIMD adds a fused multiply-add.
    if (query.cpu_arch) |arch| {
        if (arch.isWasm() and query.cpu_model == .determined_by_arch_os) {
            query.cpu_features_add.addFeature(@backingInt(std.Target.wasm.Feature.simd128));
            query.cpu_features_add.addFeature(@backingInt(std.Target.wasm.Feature.relaxed_simd));
        }
    }
    const target = b.resolveTargetQuery(query);
    const optimize = b.standardOptimizeOption(.{});

    // Only Accelerate is known to add speed on top of the Zig kernels.
    const default_blas: Blas = switch (target.result.os.tag) {
        .macos => .system,
        else => .none,
    };
    const blas = b.option(Blas, "blas", "BLAS backend: system or none") orelse default_blas;
    const options = b.addOptions();
    options.addOption(bool, "blas", blas != .none);

    const mod = b.addModule("zlaya", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addOptions("build_options", options);
    switch (blas) {
        .none => {},
        .system => if (target.result.os.tag == .macos) {
            mod.linkFramework("Accelerate", .{});
        } else {
            mod.linkSystemLibrary("blas", .{});
        },
    }

    const exe = b.addExecutable(.{
        .name = "zlaya",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zlaya", .module = mod }},
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    b.step("run", "Run CPU inference").dependOn(&run.step);

    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Run tests").dependOn(&b.addRunArtifact(tests).step);
}
