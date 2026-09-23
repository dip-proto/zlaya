# /// script
# dependencies = ["numpy==2.5.3", "torch==2.14.0", "transformers==5.17.0", "safetensors==0.8.0"]
# ///
"""Compare the native CPU engine with Laya's PyTorch reference on local weights.

Run with the reference dependencies installed in your Python environment.
The upstream Laya checkout is read from --upstream; no code is downloaded.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def cases():
    yield "triage", json.loads(Path("examples/triage.json").read_text())
    yield (
        "conversation-tail",
        {
            "state": [
                {"role": "user", "content": "Old message about oranges. " * 200},
                {"role": "user", "content": "The latest answer is blue."},
            ],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What color is in the latest message?",
                    "criteria": ["blue", "orange"],
                }
            },
        },
    )
    yield (
        "single-choice",
        {
            "state": "Only one option exists.",
            "questions": {
                "only": {
                    "type": "choice",
                    "instructions": "Choose the option.",
                    "criteria": {"only": None},
                }
            },
        },
    )
    yield (
        "many-options",
        {
            "state": "The number is eleven.",
            "questions": {
                "number": {
                    "type": "choice",
                    "instructions": "Which number is stated?",
                    "criteria": {str(i): f"The number {i}" for i in range(12)},
                }
            },
        },
    )
    yield (
        "unicode",
        {
            "state": "Café café cafe\u0301. 日本語の文章。مرحبا بالعالم! 한국어 문장. 👩🏽‍💻\r\nÉté.",
            "questions": {
                "present": {
                    "type": "noul",
                    "instructions": "Le texte contient-il du japonais ?",
                    "labels": {"false": "\u2003non\u00a0", "true": " oui "},
                },
                "score": {
                    "type": "score",
                    "instructions": "How many languages appear?",
                    "criteria": ["one", "two", "several"],
                },
            },
        },
    )
    yield (
        "special-tokens",
        {
            "state": "A [MASK] token and [CLS], [SEP], <|endoftext|>.",
            "questions": {
                "masked": {
                    "type": "noul",
                    "instructions": "Does this [MASK] mention tokens?",
                    "criteria": {
                        "false": "No [MASK] tokens",
                        "true": "Yes [MASK] tokens",
                    },
                    "labels": {"false": "absent", "true": "present"},
                }
            },
        },
    )
    yield (
        "structured-numeric",
        {
            "state": {
                "temperature": 1.0,
                "tiny": 1e-5,
                "huge": 1e16,
                "zero": -0.0,
                "enabled": False,
                "items": [None, 2, "café"],
            },
            "questions": {
                "state": {
                    "type": "choice",
                    "instructions": {"task": "Interpret café", "threshold": 0.0001},
                    "criteria": {
                        "zero": 0,
                        "disabled": False,
                        "object": {"n": 1.0},
                        "empty": "",
                    },
                },
                "truth": {
                    "type": "noul",
                    "instructions": "Is enabled true?",
                    "criteria": {"FALSE": 0, "TRUE": {"boolean": True}},
                },
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("tmp/model"))
    parser.add_argument("--upstream", type=Path, default=Path("tmp/laya-upstream"))
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/zlaya"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--case", action="append", help="Run only named cases; may be repeated"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.upstream))
    from laya import common
    from laya.agent import Agent

    config = json.loads((args.model / "rl_agent_config.json").read_text())
    max_len = config.get("max_len", 512)
    tokenizer = AutoTokenizer.from_pretrained(args.model / "tokenizer")
    model = common.build_model(config, str(args.model / "encoder"), pretrained=False)
    model.load_state_dict(
        load_file(str(args.model / "model.safetensors"), device="cpu"), assign=True
    )
    model = model.float().eval()
    maximum = {"logits": 0.0, "act_logits": 0.0, "probabilities": 0.0}
    questions_checked = 0
    Path("tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reference-check-", dir="tmp") as temp:
        for name, request in cases():
            if args.case and name not in args.case:
                continue
            path = Path(temp) / f"{name}.json"
            path.write_text(json.dumps(request, ensure_ascii=False))
            output = subprocess.run(
                [str(args.binary.resolve()), str(args.model), str(path), "--raw"],
                check=True,
                capture_output=True,
                text=True,
            )
            actual = json.loads(output.stdout)["answers"]
            for qid, question in request["questions"].items():
                qtype = common.QTYPES[question["type"]]
                internal = Agent._to_internal(question)
                ids, markers = common.build_sequence(
                    tokenizer,
                    request["state"],
                    internal,
                    max_len=max_len,
                    head_max_len=config.get("head_max_len", 192),
                    truncate_left=isinstance(request["state"], list),
                )
                native = actual[qid]
                assert native["ids"] == ids, f"{name}/{qid}: token IDs differ"
                assert native["markers"] == markers, (
                    f"{name}/{qid}: marker positions differ"
                )
                if name == "conversation-tail":
                    assert len(ids) == max_len
                    assert "latest answer is blue" in tokenizer.decode(ids[-40:])
                with torch.inference_mode():
                    logits, actions = model(
                        torch.tensor([ids]),
                        torch.ones(1, len(ids), dtype=torch.long),
                        torch.tensor([markers]),
                        torch.ones(1, len(markers), dtype=torch.bool),
                        torch.tensor([qtype]),
                    )
                bucket = common.temp_bucket(qtype, len(markers))
                temperature = common.clamp_temperature(
                    config.get("temperature_by_options", {}).get(
                        bucket, config.get("temperature", [1, 1, 1])[qtype]
                    )
                )
                probabilities = torch.softmax(logits[0] / temperature, -1).tolist()
                if question["type"] == "noul":
                    actual_probabilities = [1 - native["noul"], native["noul"]]
                else:
                    actual_probabilities = list(native["probabilities"].values())
                references = {
                    "logits": logits[0].tolist(),
                    "act_logits": actions[0].tolist(),
                    "probabilities": probabilities,
                }
                observed = {
                    "logits": native["logits"],
                    "act_logits": native["act_logits"],
                    "probabilities": actual_probabilities,
                }
                for field in references:
                    expected = np.array(references[field])
                    got = np.array(observed[field])
                    tolerance = 1e-3 if field != "probabilities" else 1e-4
                    np.testing.assert_allclose(
                        got,
                        expected,
                        rtol=1e-4 if field == "act_logits" else 0,
                        atol=tolerance,
                        err_msg=f"{name}/{qid}: {field}",
                    )
                    maximum[field] = max(
                        maximum[field], float(np.max(np.abs(got - expected)))
                    )
                assert native["type"] == question["type"]
                p = np.array(probabilities)
                confidence = (
                    max(p[1], 1 - p[1])
                    if qtype == 2
                    else common.confidence_from_probs(p, len(p))
                )
                np.testing.assert_allclose(
                    native["confidence"], confidence, rtol=0, atol=1e-4
                )
                np.testing.assert_allclose(
                    native["action"]["act_probability"],
                    torch.softmax(actions[0], -1)[0].item(),
                    rtol=0,
                    atol=1e-4,
                )
                if qtype == 0:
                    labels = list(internal["crit"])
                    assert native["choice"] == labels[int(p.argmax())]
                    assert list(native["probabilities"]) == labels
                elif qtype == 1:
                    np.testing.assert_allclose(
                        native["score"],
                        float((np.arange(len(p)) * p).sum()),
                        rtol=0,
                        atol=1e-4,
                    )
                    assert list(native["legend"].values()) == question["criteria"]
                questions_checked += 1
                print(
                    f"PASS {name}/{qid}: {len(ids)} tokens, {len(markers)} options, temperature {temperature:.6g}",
                    flush=True,
                )
    assert questions_checked > 0, "No cases matched"
    print(
        f"Validated {questions_checked} questions; maximum absolute differences: {maximum}"
    )


if __name__ == "__main__":
    main()
