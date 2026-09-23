// Runs the WebAssembly build with Node's WASI support:
//   node scripts/zlaya-node.mjs zig-out/bin/zlaya.wasm MODEL_DIRECTORY REQUEST.json [--raw]
// The current directory is exposed to the program, so paths must be relative and below it.
import { WASI } from 'node:wasi';
import { readFile } from 'node:fs/promises';

const [wasmPath, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: 'preview1', args: ['zlaya', ...args], preopens: { '.': '.' }, returnOnExit: true });
const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), wasi.getImportObject());
process.exitCode = wasi.start(instance);
