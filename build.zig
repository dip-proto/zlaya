const std = @import("std");

const Blas = enum { none, system, openblas };

pub fn build(b: *std.Build) void {
    var query = b.standardTargetOptionsQueryOnly(.{});
    // Enable SIMD on WebAssembly unless -Dcpu is given.
    // Every current runtime supports it.
    if (query.cpu_arch) |arch| {
        if (arch.isWasm() and query.cpu_model == .determined_by_arch_os) {
            query.cpu_features_add.addFeature(@backingInt(std.Target.wasm.Feature.simd128));
        }
    }
    const target = b.resolveTargetQuery(query);
    const optimize = b.standardOptimizeOption(.{});

    const default_blas: Blas = switch (target.result.os.tag) {
        .macos => .system,
        .wasi => .openblas,
        else => .none,
    };
    const blas = b.option(Blas, "blas", "BLAS backend: system, openblas (built from source, WASI only) or none") orelse default_blas;
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
        .openblas => {
            if (target.result.os.tag != .wasi) std.process.fatal("-Dblas=openblas is only configured for WASI targets", .{});
            if (b.lazyDependency("openblas", .{})) |dep| mod.linkLibrary(openBlas(b, dep, target));
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

/// Compiles the part of OpenBLAS behind `cblas_sgemm`, with its single-threaded WASM128_GENERIC kernels.
fn openBlas(b: *std.Build, dep: *std.Build.Dependency, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "openblas",
        .linkage = .static,
        // A Debug OpenBLAS makes Debug inference about ten times slower.
        .root_module = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true }),
    });
    const m = lib.root_module;

    // OpenBLAS normally generates config.h with a host tool.
    // This one only defines the macros that the sources below read.
    const headers = b.addWriteFiles();
    _ = headers.add("config.h",
        \\#define ARCH_WASM 1
        \\#define WASM128_GENERIC
        \\#define HAVE_C11 1
        \\#define NEEDBUNDERSCORE 1
        \\#define MAX_CPU_NUMBER 1
        \\#define MAX_PARALLEL_NUMBER 1
        \\#define NO_SYSV_IPC
        \\
    );
    // memory.c includes this header on every Unix-like system, but wasi-libc doesn't have it.
    _ = headers.add("sys/ipc.h", "");
    m.addIncludePath(headers.getDirectory());
    m.addIncludePath(dep.path(""));
    m.addCMacro("_WASI_EMULATED_MMAN", "");
    m.linkSystemLibrary("wasi-emulated-mman", .{});

    const sources = [_]struct { []const u8, []const []const u8 }{
        .{ "interface/gemm.c", &.{ "-DCNAME=cblas_sgemm", "-DCBLAS" } },
        .{ "driver/level3/gemm.c", &.{ "-DCNAME=sgemm_nn", "-DNN" } },
        .{ "driver/level3/gemm.c", &.{ "-DCNAME=sgemm_nt", "-DNT" } },
        .{ "driver/level3/gemm.c", &.{ "-DCNAME=sgemm_tn", "-DTN" } },
        .{ "driver/level3/gemm.c", &.{ "-DCNAME=sgemm_tt", "-DTT" } },
        .{ "kernel/wasm/gemmkernel_wasm128.c", &.{"-DCNAME=sgemm_kernel"} },
        .{ "kernel/generic/gemm_beta.c", &.{"-DCNAME=sgemm_beta"} },
        .{ "kernel/generic/gemm_ncopy_2.c", &.{"-DCNAME=sgemm_oncopy"} },
        .{ "kernel/generic/gemm_tcopy_2.c", &.{"-DCNAME=sgemm_otcopy"} },
        .{ "driver/others/memory.c", &.{} },
        .{ "driver/others/xerbla.c", &.{} },
        .{ "driver/others/openblas_env.c", &.{} },
    };
    for (sources) |source| {
        const path, const flags = source;
        m.addCSourceFile(.{ .file = dep.path(path), .flags = flags });
    }
    return lib;
}
