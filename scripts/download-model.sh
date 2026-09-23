#!/bin/sh
set -eu
model_dir=${1:-models/laya}
revision=aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8
mkdir -p "$model_dir/encoder" "$model_dir/tokenizer"
for file in model.safetensors encoder/config.json rl_agent_config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
    curl --fail --location --retry 3 "https://huggingface.co/convaiinnovations/laya/resolve/$revision/$file" -o "$model_dir/$file.part"
    if [ "$file" = model.safetensors ]; then
        expected=891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c
        if command -v sha256sum >/dev/null 2>&1; then
            actual=$(sha256sum "$model_dir/$file.part")
        else
            actual=$(shasum -a 256 "$model_dir/$file.part")
        fi
        if [ "${actual%% *}" != "$expected" ]; then
            echo "Checkpoint SHA-256 mismatch" >&2
            exit 1
        fi
    fi
    mv "$model_dir/$file.part" "$model_dir/$file"
done
