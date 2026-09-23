#!/bin/sh
# Runs the WebAssembly build with the same arguments as the native binary,
# for example as `scripts/validate-reference.py --binary scripts/zlaya-wasmtime.sh`.
# Only the current directory is exposed, under both "." and its absolute path.
exec wasmtime run --dir . --dir "$PWD::$PWD" "$(dirname "$0")/../zig-out/bin/zlaya.wasm" "$@"
