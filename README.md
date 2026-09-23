<p align="center">
  <img src="assets/logo.svg" alt="zlaya logo" width="400">
</p>

# zlaya

A CPU-only inference engine in Zig for [Laya](https://huggingface.co/convaiinnovations/laya).
It needs no GPU and can run entirely in WebAssembly.

It answers choice, score, and yes/no (`noul`) questions about text or structured JSON.
Inference runs locally using native Zig tokenization, safetensors loading, ModernBERT, and the trained decision and action heads.

## Quick start

Use Zig-nightly.

```sh
zig build -Doptimize=ReleaseFast
sh scripts/download-model.sh models/laya
zig-out/bin/zlaya models/laya examples/triage.json
```

The weights take about 804 MiB on disk and are expanded to float32 in memory.
Allow several GiB of RAM for loading and inference.
After downloading the files, inference requires no network access.

## Requests and answers

Pass a JSON file with `state` and a nonempty `questions` object.
State can be text, an object, or a conversation list.
Questions are evaluated sequentially with the model loaded once.

```json
{
  "state": "Please refund the duplicate charge.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Does the user request a refund?"
    }
  }
}
```

### Question types

- `choice` takes `criteria` as an object mapping labels to descriptions, or a list of label strings.
- `score` takes a list of level descriptions and returns the expected level index, starting at zero.
- `noul` returns the probability that its statement is true.
  It optionally accepts `criteria` descriptions keyed by `false` and `true`, and a `labels` object with those same two keys.

See [examples/triage.json](examples/triage.json) for all three types.

```sh
zig-out/bin/zlaya models/laya examples/triage.json
```

### Output

Results are written as JSON to stdout under `answers`.
Choice and score answers include their full probability distribution.

Every answer includes confidence and the learned action probability.

Numbers are rounded to four decimal places, following the current Python implementation.

Temperature calibration uses the checkpoint's option-count buckets and the upstream clamp to `[0.5, 5]`.

When stderr is a terminal, zlaya shows progress there while loading the model and answering questions.

Native builds draw a live counter, and WebAssembly builds print one line per phase.
Redirecting stderr hides it, but errors are still reported there.

### Context limits and debugging

The checkpoint's default context budget is 512 tokens, with a 192-token question/options budget.
Long text keeps its beginning; long conversation lists keep their end.
An error is returned if an option marker would be truncated away.

Use `--raw` as the final argument to include token IDs, marker positions, option logits, and action logits for verification.

## Build options

### Native backends

On macOS, the default build uses Apple's Accelerate CPU BLAS for matrix multiplication.
The rest of the model executes in Zig.

On other systems, the default uses the portable Zig matrix kernels and libc's error function.

You can select the backend explicitly with `-Dblas`:

```sh
zig build -Doptimize=ReleaseFast -Dblas=none
zig build -Doptimize=ReleaseFast -Dblas=system
```

`none` uses the portable Zig kernels.

`system` links Accelerate on macOS, and elsewhere a system library exposing `cblas_sgemm` as `libblas`.
A third value, `openblas`, compiles OpenBLAS from source; it is the default for WebAssembly and only supported there.

### WebAssembly

```sh
zig build -Dtarget=wasm32-wasi -Doptimize=ReleaseFast
wasmtime run --dir . zig-out/bin/zlaya.wasm models/laya examples/triage.json
```

This produces `zig-out/bin/zlaya.wasm`, a WASI program that takes the same arguments and prints the same answers as the native one.
For matrix multiplication, the build downloads OpenBLAS from source and compiles it to WebAssembly.

Building needs nothing besides Zig, and native builds never fetch OpenBLAS.

Pass `-Dblas=none` to use the portable Zig kernels instead.

WebAssembly SIMD is enabled by default, since every current runtime supports it and it roughly halves inference time.
An explicit `-Dcpu`, such as `-Dcpu=mvp`, takes precedence, and OpenBLAS then uses its scalar code paths.

#### Runtime file access

WASI programs can only open files in directories the runtime exposes to them.
With wasmtime, `--dir .` grants access to the current directory, so keep the model and the request below it and pass relative paths.

The same binary runs under wasmer, where guest paths are absolute:

```sh
wasmer run --volume .:/work zig-out/bin/zlaya.wasm -- /work/models/laya /work/examples/triage.json
```

It also runs under Node's `node:wasi` module, through the small launcher in `scripts`:

```sh
node scripts/zlaya-node.mjs zig-out/bin/zlaya.wasm models/laya examples/triage.json
```

Browsers would need a WASI shim that provides file access, but `wasm32-freestanding` support will be added as an alternative soon.

#### Memory use

On every target, the checkpoint is read tensor by tensor into a single float32 buffer, without ever holding the whole file.
WebAssembly depends on this: wasm32 can address only 4 GiB, and its allocator rounds large blocks up to powers of two.
The WebAssembly memory reaches about 2.1 GiB after loading, and 2.8 GiB with a full 512-token prompt.

## Zig API

Import the `zlaya` module exported by `build.zig`.

### Engine lifecycle

`Engine.load(gpa, io, model_directory, progress)` loads reusable weights and tokenization data; call `deinit()` when finished.
The same allocator provides scratch memory for each prediction.

`engine.predict(arena, request_value, false, progress)` returns a `std.json.Value` response allocated in `arena`.
Use a fresh arena for each request and free it after consuming the response.
The response borrows question IDs and criteria from the request, so keep the request alive as long as the response.

Both functions report to a `std.Progress.Node`, with one item per converted tensor or answered question.
Pass `.none` if you don't need progress.

### Lower-level API

The module also exports `Tokenizer`, `SafeTensors`, `Model`, `QuestionType`, and `sequence` for callers that need raw tokenization or logits.

Like in the standard library, the types with fields are files of their own, so `zlaya.Tokenizer` is `src/Tokenizer.zig`.

`SafeTensors.open` reads only the header of a checkpoint, so keep the file open until the model is loaded.
`Tokenizer.init` allocates everything in the arena it is given.
`Model.init` takes an arena plus a separate allocator for the float32 weights, which `SafeTensors.totalLen` can size exactly.

`Model.forward` takes token IDs, option marker positions, and a `QuestionType`.
It returns a `Model.Output`, whose `deinit(gpa)` frees the logits.
