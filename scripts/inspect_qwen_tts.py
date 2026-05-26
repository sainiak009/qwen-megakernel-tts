#!/usr/bin/env python3
"""
inspect_qwen_tts.py

Compare Qwen3-0.6B (megakernel target) vs Qwen3-TTS-12Hz-0.6B-Base talker decoder.
Shows architecture diffs, vocab sizes, weight names, and the exact remapping needed.

Run without GPU — only fetches configs, not weights.
"""

import json
import urllib.request

# ── Known constants from megakernel source ────────────────────────────────────

MEGAKERNEL_ARCH = {
    "model": "Qwen/Qwen3-0.6B",
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,           # text LM head
    "max_seq_len": 2048,            # hardcoded in model.py
    "rope_theta": 10000.0,          # standard Qwen3 (computed in Python, not kernel)
    "rms_norm_eps": 1e-6,
    "embed_size": 151936,           # embed_tokens rows
    "lm_head_size": 151936,         # output projection rows
    "tied_embeddings": True,
}

TALKER_ARCH = {
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base (talker decoder)",
    "num_hidden_layers": 28,        # SAME
    "hidden_size": 1024,            # SAME
    "num_attention_heads": 16,      # SAME
    "num_key_value_heads": 8,       # SAME
    "head_dim": 128,                # SAME
    "intermediate_size": 3072,      # SAME
    "vocab_size": 3072,             # audio codec tokens  ← DIFFERENT
    "max_seq_len": 32768,           # config max; 4096 is enough for most TTS
    "rope_theta": 1000000.0,        # ← DIFFERENT
    "rms_norm_eps": 1e-6,           # SAME
    "text_vocab_size": 151936,      # text embed (prefill) ← context only
    "embed_size": 3072,             # talker's own embed_tokens (audio codes only)
    "lm_head_size": 3072,           # codec_head rows      ← DIFFERENT
    "tied_embeddings": False,       # codec_head is separate
    "num_code_groups": 16,          # residual codebooks (code_predictor handles 2-16)
    "position_id_per_seconds": 13,
}

# ── Weight name mapping ───────────────────────────────────────────────────────

WEIGHT_MAPPING = """
Megakernel expects          TTS state dict key                       Notes
─────────────────────────────────────────────────────────────────────────────
model.embed_tokens.weight   <talker_prefix>embed_tokens.weight       Audio codes only (3072×1024)
                                                                      Text embed handled separately
                                                                      during prefill.

model.norm.weight           <talker_prefix>model.norm.weight         Final RMSNorm. SAME shape.

lm_head_weight              <talker_prefix>codec_head.weight         Shape [3072,1024]. Must be
(used as lm_head in kernel) (NOT tied to embed_tokens)               padded to [151936,1024] for
                                                                      kernel compatibility. Rows
                                                                      3072..151935 set to zero —
                                                                      argmax still correct as long
                                                                      as ≥1 audio logit > 0.

Per-layer weights (×28):
  input_layernorm.weight    layers.{i}.input_layernorm.weight        SAME shape [1024]
  self_attn.q_proj.weight   layers.{i}.self_attn.q_proj.weight       SAME [2048,1024]
  self_attn.k_proj.weight   layers.{i}.self_attn.k_proj.weight       SAME [1024,1024]
  self_attn.v_proj.weight   layers.{i}.self_attn.v_proj.weight       SAME [1024,1024]
  self_attn.q_norm.weight   layers.{i}.self_attn.q_norm.weight       SAME [128]
  self_attn.k_norm.weight   layers.{i}.self_attn.k_norm.weight       SAME [128]
  self_attn.o_proj.weight   layers.{i}.self_attn.o_proj.weight       SAME [1024,2048]
  post_attn_layernorm.wt    layers.{i}.post_attention_layernorm.weight SAME [1024]
  mlp.gate_proj.weight      layers.{i}.mlp.gate_proj.weight          SAME [3072,1024]
  mlp.up_proj.weight        layers.{i}.mlp.up_proj.weight            SAME [3072,1024]
  mlp.down_proj.weight      layers.{i}.mlp.down_proj.weight          SAME [1024,3072]
"""

DIFFERENCES = [
    ("rope_theta",          10_000.0,  1_000_000.0,  "Recompute Python cos/sin tables. No kernel change."),
    ("max_seq_len",         2048,      4096,          "Reallocate KV cache. No kernel change."),
    ("vocab_size (output)", 151936,    3072,          "Pad codec_head to [151936,1024] with zeros."),
    ("embed_tokens rows",   151936,    3072,          "Swap embed table after text prefill."),
    ("lm_head tied",        True,      False,         "Use codec_head, not embed_tokens."),
]

NOT_MODIFIED = """
The code_predictor (residual codebook generator) runs in Python/PyTorch
entirely outside the CUDA megakernel — it is not touched by this adaptation.
"""


def fetch_hf_config(repo_id: str) -> dict:
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def print_comparison():
    print("=" * 78)
    print("Architecture Comparison: Qwen3-0.6B Megakernel  vs  Qwen3-TTS Talker")
    print("=" * 78)

    fields = [
        "num_hidden_layers", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
        "vocab_size", "max_seq_len", "rope_theta", "rms_norm_eps",
        "embed_size", "lm_head_size", "tied_embeddings",
    ]

    print(f"\n{'Field':<28} {'Megakernel (Qwen3-0.6B)':<28} {'TTS Talker':<28} {'Match'}")
    print("-" * 92)
    for f in fields:
        mk  = MEGAKERNEL_ARCH.get(f, "—")
        tts = TALKER_ARCH.get(f, "—")
        match = "✓" if mk == tts else "✗"
        print(f"  {f:<26} {str(mk):<28} {str(tts):<28} {match}")

    print("\n\nDifferences requiring adaptation:")
    print("-" * 92)
    print(f"  {'Field':<22} {'Megakernel':<18} {'TTS Talker':<18} Fix")
    for field, mk, tts, fix in DIFFERENCES:
        print(f"  {field:<22} {str(mk):<18} {str(tts):<18} {fix}")

    print(WEIGHT_MAPPING)
    print("NOT modified:")
    print(NOT_MODIFIED)

    print("=" * 78)
    print("Layers that are IDENTICAL (no kernel change needed):")
    print("  All 28 transformer layers: RMSNorm, QKV proj, q/k_norm, o_proj, SwiGLU MLP")
    print("  → The megakernel's fused CUDA kernel runs these unchanged.")
    print("=" * 78)


def inspect_live_model(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
    """Load the actual model and print state dict key names with shapes."""
    print(f"\n\nLive inspection of {model_name}")
    print("(requires ~2 GB download on first run)")
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print("transformers not installed; skipping live inspection.")
        return

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"\nConfig model_type: {cfg.model_type}")

    print("\nLoading model (this downloads weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("\nState dict keys and shapes (first 60 keys):")
    sd = model.state_dict()
    for i, (k, v) in enumerate(sd.items()):
        print(f"  {k:<70} {list(v.shape)}")
        if i >= 59:
            print(f"  ... ({len(sd)} total keys)")
            break

    # Find the talker prefix dynamically
    talker_prefix = None
    for k in sd:
        if "layers.0.self_attn.q_proj.weight" in k:
            idx = k.index("layers.0.self_attn.q_proj.weight")
            talker_prefix = k[:idx]
            print(f"\nDetected talker transformer prefix: '{talker_prefix}'")
            break

    # Find codec/audio head
    for k, v in sd.items():
        if "codec_head" in k or ("lm_head" in k and v.shape[0] <= 4096):
            print(f"Audio LM head: {k}  {list(v.shape)}")

    # Find audio embed
    for k, v in sd.items():
        if "embed_tokens" in k and v.shape[0] <= 4096:
            print(f"Audio embed_tokens: {k}  {list(v.shape)}")


if __name__ == "__main__":
    import sys

    print_comparison()

    if "--live" in sys.argv:
        inspect_live_model()
    else:
        print("\nTip: run with --live to load the actual model and inspect weight names.")
