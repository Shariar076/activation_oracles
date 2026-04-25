#!/usr/bin/env python3
"""One-shot diagnostic for mwhanna/qwen3-8b-transcoders layer 18.

Tests whether the encoder is working by checking reconstruction:
  decode(encode(mlp_in)) should approximate mlp_out.

Tries two encode conventions:
  A) relu(x @ W_enc.T + b_enc)              — no input centering
  B) relu((x - b_dec) @ W_enc.T + b_enc)    — dictionary_learning convention

Whichever gives lower reconstruction MSE is the right one.
Also prints L0 (avg active features/token) and explained variance.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LAYER = 18
MODEL_NAME = "Qwen/Qwen3-8B"
PROMPT = "The quick brown fox jumps over the lazy dog. The word is leaf."

device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)

# Load transcoder weights
path = hf_hub_download(repo_id="mwhanna/qwen3-8b-transcoders", filename=f"layer_{LAYER}.safetensors")
T = {}
with safe_open(path, framework="pt") as f:
    for k in f.keys():
        T[k] = f.get_tensor(k).to(device=device, dtype=dtype)
W_enc, W_dec, b_enc, b_dec = T["W_enc"], T["W_dec"], T["b_enc"], T["b_dec"]
print(f"W_enc {tuple(W_enc.shape)} W_dec {tuple(W_dec.shape)} b_enc {tuple(b_enc.shape)} b_dec {tuple(b_dec.shape)}")

# Load model
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True), torch_dtype=dtype,
)
model.eval()

# Capture MLP input AND output
mlp = model.model.layers[LAYER].mlp
captured = {}
def pre_hook(_m, args, kwargs):
    captured["in"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
def post_hook(_m, _args, out):
    captured["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

h1 = mlp.register_forward_pre_hook(pre_hook, with_kwargs=True)
h2 = mlp.register_forward_hook(post_hook)
inputs = tok(PROMPT, return_tensors="pt", add_special_tokens=False).to(device)
model(**inputs)
h1.remove(); h2.remove()

x_in = captured["in"][0].float()    # [L, d]
x_out = captured["out"][0].float()  # [L, d]
print(f"mlp_in shape {tuple(x_in.shape)} | mlp_out shape {tuple(x_out.shape)}")
print(f"mlp_in  norm/tok: mean={x_in.norm(dim=-1).mean():.3f}")
print(f"mlp_out norm/tok: mean={x_out.norm(dim=-1).mean():.3f}")

W_enc32, W_dec32, b_enc32, b_dec32 = W_enc.float(), W_dec.float(), b_enc.float(), b_dec.float()

def reconstruct(encode_fn, x_in, x_out_target, name):
    feats = encode_fn(x_in)             # [L, d_sae]
    l0 = (feats > 0).float().sum(-1).mean().item()
    recon = feats @ W_dec32 + b_dec32   # [L, d]
    err = (recon - x_out_target).pow(2).mean().item()
    var = x_out_target.var().item()
    ev = 1.0 - err / var
    print(f"  [{name}] L0={l0:.1f}/{W_enc32.shape[0]} | MSE={err:.4f} | var(target)={var:.4f} | EV={ev:.3f}")

print("\n=== Reconstructing mlp_OUT from mlp_IN ===")
reconstruct(lambda x: torch.relu(x @ W_enc32.T + b_enc32),               x_in, x_out, "A: no centering")
reconstruct(lambda x: torch.relu((x - b_dec32) @ W_enc32.T + b_enc32),   x_in, x_out, "B: subtract b_dec from input")

print("\n=== Reconstructing mlp_IN from mlp_IN (SAE-style) ===")
reconstruct(lambda x: torch.relu(x @ W_enc32.T + b_enc32),               x_in, x_in, "A: no centering")
reconstruct(lambda x: torch.relu((x - b_dec32) @ W_enc32.T + b_enc32),   x_in, x_in, "B: subtract b_dec from input")
