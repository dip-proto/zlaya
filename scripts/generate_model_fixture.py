# /// script
# dependencies = ["numpy==2.5.3", "torch==2.14.0", "transformers==5.17.0", "safetensors==0.8.0"]
# ///
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import ModernBertConfig, ModernBertModel

sys.path.insert(0, "tmp/laya-upstream")
from laya import common as c

torch.manual_seed(714)
config = ModernBertConfig(
    vocab_size=19,
    hidden_size=8,
    intermediate_size=10,
    num_hidden_layers=3,
    num_attention_heads=2,
    max_position_embeddings=32,
    local_attention=2,
    global_attn_every_n_layers=3,
    attention_dropout=0.0,
    embedding_dropout=0.0,
    mlp_dropout=0.0,
    pad_token_id=0,
    cls_token_id=1,
    sep_token_id=2,
    bos_token_id=1,
    eos_token_id=2,
)
config._attn_implementation = "eager"
model = c.DecisionModel(ModernBertModel(config)).eval()
root = Path("src/fixtures")
save_file(model.state_dict(), str(root / "tiny.safetensors"))
(root / "tiny-encoder.json").write_text(config.to_json_string())
ids = torch.tensor([[1, 4, 9, 3, 7, 5, 2]])
markers = torch.tensor([[2, 4, 5]])
results = []
with torch.no_grad():
    for qt in range(3):
        logits, act = model(
            ids,
            torch.ones_like(ids),
            markers,
            torch.ones_like(markers, dtype=torch.bool),
            torch.tensor([qt]),
        )
        results.append(
            {"qtype": qt, "logits": logits[0].tolist(), "act_logits": act[0].tolist()}
        )
(root / "tiny-expected.json").write_text(json.dumps(results, indent=2) + "\n")
