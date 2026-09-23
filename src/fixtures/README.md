# Model numerical fixture

These files contain a randomly initialized tiny Laya model, not pretrained model weights.
The deterministic seed is 714.
The encoder has vocabulary 19, width 8, three layers, two attention heads, intermediate width 10, and local attention distance 1.
The decision head uses the official `DecisionModel` defaults of two layers and two actions, which the Zig test passes to `Model.init`.

Inputs are token IDs `[1, 4, 9, 3, 7, 5, 2]` and option markers `[2, 4, 5]`.
Expected outputs cover all three question types with dropout disabled and float32 CPU inference.
The test compares option logits and action logits with absolute tolerance 0.00002.

Regenerate with `scripts/generate_model_fixture.py` through `uv run` in an environment containing PyTorch, Transformers, and safetensors.
The script expects the official Laya repository checked out in `tmp/laya-upstream`.
The reference revision used was `1e28ac20c0896b1c37a744cd11f740eb98f8b178`.
