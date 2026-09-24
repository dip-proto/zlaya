# Third-party sources

The model architecture, prompt construction, and decision semantics follow the Apache-2.0 licensed [Laya project](https://github.com/NandhaKishorM/laya) by NandhaKishorM and its contributors.
The Zig implementation ports these inference operations and replaces PyTorch execution with CPU kernels.
The reference revision is `1e28ac20c0896b1c37a744cd11f740eb98f8b178`.
The full Apache-2.0 license is included in LICENSE.

The pretrained model remains a separate download from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), also published under Apache-2.0.
No pretrained weights are included in this source distribution.

The generated Unicode tables derive from Python's Unicode 15.1 database.
The Unicode data license is included in UNICODE-LICENSE.txt.
The generator and exact Unicode version are recorded in `scripts/generate-unicode.py`.
