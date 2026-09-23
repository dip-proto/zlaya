# /// script
# dependencies = ["tokenizers==0.22.2"]
# ///
"""Compare native tokenization against Hugging Face using a local tokenizer JSON."""

import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

root = pathlib.Path(__file__).resolve().parent.parent
source = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root / "models/laya/tokenizer/tokenizer.json")
reference = Tokenizer.from_file(str(source))
cases = [
    "",
    "hello world",
    " café!",
    "  hello",
    "\n\t test",
    "Hello, I'm testing 123.",
    "[CLS] [MASK] <|endoftext|>",
    "e\u0301 \u1100\u1161\u11a8",
    "你好世界",
    "مرحبا بالعالم",
    "a" * 50,
    " " * 30,
    "a\n\n b",
    "\u0301a",
    "x\u0085y",
    "foo|||EMAIL_ADDRESS|||bar",
    "📧 test™ ①",
    "foo\u2003[MASK]bar",
]
cases += ["foo" + " " * count + "[MASK]bar" for count in range(1, 50)]
rng = random.Random(42)
alphabet = list("aZ'123 !\n\t\r café世界مرحبا\u0301\u0315\u0085\u00a0\u2003📧")
for _ in range(500):
    cases.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 100))))
(root / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(
    prefix="tokenizer-check-", dir=root / "tmp"
) as directory:
    directory = pathlib.Path(directory)
    shutil.copyfile(source, directory / "tokenizer.json")
    lines = [
        'const std = @import("std");',
        'const Tokenizer = @import("tokenizer");',
        'test "upstream tokenizer parity" {',
        "const a = std.testing.allocator;",
        "var arena: std.heap.ArenaAllocator = .init(a); defer arena.deinit();",
        'const json = try a.dupe(u8, @embedFile("tokenizer.json"));',
        "const t = try Tokenizer.init(arena.allocator(), json); a.free(json);",
    ]
    for case in cases:
        literal = "".join(f"\\x{byte:02x}" for byte in case.encode())
        for special, method in [(True, "encode"), (False, "encodeRaw")]:
            ids = reference.encode(case, add_special_tokens=special).ids
            expected = ",".join(map(str, ids))
            lines.append(
                f'{{ const result = try t.{method}(a, "{literal}"); defer a.free(result); '
                f"try std.testing.expectEqualSlices(u32, &.{{{expected}}}, result); }}"
            )
    lines.append("}")
    harness = directory / "test.zig"
    harness.write_text("\n".join(lines))
    subprocess.run(
        [
            "zig",
            "test",
            "--dep",
            "tokenizer",
            f"-Mroot={harness}",
            f"-Mtokenizer={root / 'src/Tokenizer.zig'}",
        ],
        check=True,
        cwd=root,
    )
print(f"Matched {len(cases)} inputs with and without special tokens.")
