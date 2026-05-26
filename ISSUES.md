# Issues & Fixes Log

---

## Issue #1 — `pip install git+https://...` fails for qwen_megakernel

**Error:**
```
ERROR: git+https://github.com/AlpinDale/qwen_megakernel does not appear to be a Python project:
neither 'setup.py' nor 'pyproject.toml' found.
```

**Cause:** The megakernel repo has no Python package metadata — it's a raw source repo, not a pip-installable package.

**Fix:** Clone manually and add to PYTHONPATH:
```bash
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel && pip install -r requirements.txt && cd ..
export PYTHONPATH="$(pwd)/qwen_megakernel:$PYTHONPATH"
```

---

## Issue #2 — `KeyError: 'qwen3_tts'` — qwen3_tts is not in the transformers registry

**Error:**
```
KeyError: 'qwen3_tts'
ValueError: The checkpoint you are trying to load has model type `qwen3_tts`
but Transformers does not recognize this architecture.
```

**Attempted dead ends:**
- Upgrading transformers to `5.10.0.dev0` via `git+https://github.com/huggingface/transformers.git` — error persisted.
- Adding `trust_remote_code=True` — no effect. `qwen3_tts` is not in the transformers AUTO registry at any version.

**Root cause:** `qwen3_tts` is Alibaba's proprietary model type. It is **not registered in the standard HuggingFace transformers library at all** — not in the stable release, not in `5.10.0.dev0`, not in the git main branch. Alibaba distributes it via their own separate package: `qwen-tts`.

Our original code used `AutoModelForCausalLM.from_pretrained()` from transformers, which only works for model types registered in the transformers AUTO registry. `qwen3_tts` is not in that registry, so no version of transformers will ever load it this way.

**What we changed in the code (model loading):**

*Before (broken) — `scripts/generate_megakernel_tts.py`, `scripts/generate_baseline_tts.py`, `server/qwen_tts_engine.py`:*
```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,   # didn't help — fails at config registry lookup
)
processor = AutoProcessor.from_pretrained(model_id)
```

*After (working):*
```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    model_id,
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

**What we changed (tokenizer removal from engine):**

The original `server/qwen_tts_engine.py` loaded `AutoTokenizer` alongside the model for the megakernel path. Since `Qwen3TTSModel` handles tokenization internally in the HF baseline path (`generate_custom_voice()`), we removed the standalone tokenizer there. However, the megakernel path (`_stream_megakernel`) still needs to tokenize text before calling `decoder.prefill_text(text_ids)`, so we kept `AutoTokenizer.from_pretrained()` specifically for that path.

Important: `AutoTokenizer.from_pretrained()` **does work** with `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — the `qwen3_tts` registry error only fires when loading the model architecture via `AutoModelForCausalLM`. Tokenizer loading reads `tokenizer.json`/`tokenizer_config.json` and does not depend on model type registration.

**What we changed (baseline generation API):**

The HF baseline path in `_stream_hf()` was rewritten to use `Qwen3TTSModel.generate_custom_voice()` instead of `model.generate()` — the `qwen-tts` API is completely different from the standard transformers generation API:
```python
audio_list, sr = model.generate_custom_voice(
    text=text,
    language="English",
    speaker="default",
)
```

**Fix on instance:**
```bash
pip install qwen-tts>=0.1.1
git pull   # get updated engine code
uvicorn server.app:app --host 0.0.0.0 --port 8080
```

**Lesson:** When a model config shows `model_type: qwen3_tts`, always check if the vendor ships their own inference package before reaching for `transformers.AutoModel*`. Also: `AutoTokenizer` is safe to use even when `AutoModelForCausalLM` fails — they are separate registry lookups.

---

## Issue #3 — `'Qwen3TTSModel' object has no attribute 'state_dict'`

**Error:**
```
Megakernel load failed ('Qwen3TTSModel' object has no attribute 'state_dict'); falling back to HF baseline.
```

**Cause:** `Qwen3TTSModel` is not an `nn.Module` subclass — it is a plain Python wrapper class. Calling `.state_dict()` directly on it fails. Our `load_tts_weights()` assumed it was a standard PyTorch model.

**Fix:** Access the inner `nn.Module` via `model.model` before calling `.state_dict()`:

*Before (broken):*
```python
model = Qwen3TTSModel.from_pretrained(model_id, ...)
sd = model.state_dict()   # AttributeError
```

*After (working):*
```python
model = Qwen3TTSModel.from_pretrained(model_id, ...)
inner = model.model if hasattr(model, "model") else model
sd = inner.state_dict()
```

This is consistent with how we already probe `model.model.speech_tokenizer` elsewhere in the same function.

**Lesson:** Vendor wrapper classes (like `Qwen3TTSModel`) are not always `nn.Module` subclasses. Always check before calling PyTorch model methods on them.

---

## Issue #4 — `Could not find talker audio embed_tokens (expected shape [3072, 1024])`

**Error:**
```
Megakernel load failed (Could not find talker audio embed_tokens (expected shape [3072, 1024])); falling back to HF baseline.
```

**Cause:** Our code assumed the talker has two separate embedding tables — one for text tokens [151936, 1024] and one for audio tokens [3072, 1024]. In practice the talker uses a **single shared embed_tokens table** (likely [151936, 1024]) for both text and audio tokens. No separate [3072, 1024] audio embed exists in the state dict.

**Fix:** Added a fallback: if no exact [3072, 1024] embed is found, slice the first `TTS_AUDIO_VOCAB` (3072) rows from the talker's embed table to use as the audio embed. Also added a diagnostic print of all embed keys on load so the actual shapes are visible in logs.

```python
# Fallback: slice first 3072 rows from shared embed table
for k, v in sd.items():
    if "embed_tokens" in k and prefix in k and v.shape[0] >= TTS_AUDIO_VOCAB:
        audio_embed = v[:TTS_AUDIO_VOCAB].contiguous()
        break
```

Similarly, the text prefill path now falls back to using the full talker embed table if no 151936-row embed is found separately.

**Actual state dict revealed** (from diagnostic output on instance):
```
talker.model.codec_embedding.weight   [3072, 1024]    ← audio embed (correct shape)
talker.model.text_embedding.weight    [151936, 2048]  ← text embed (2048-dim, NOT 1024!)
talker.code_predictor.model.codec_embedding.{0-14}.weight  [2048, 1024]  ← codebook generator (not our concern)
```

Two sub-issues discovered:
1. Key names are `codec_embedding` / `text_embedding`, NOT `embed_tokens`
2. `text_embedding` has hidden dim **2048**, but the talker transformer expects **1024** — a projection layer must exist somewhere

**Updated fix:**
- Audio embed: search for `codec_embedding` key with shape [3072, 1024] ✓
- Text embed: find `text_embedding` [151936, 2048], then search for a projection weight [2048→1024] under the talker prefix to bring it down to HIDDEN_SIZE before use
- Added diagnostic print of all non-layer talker keys so projection weight name becomes visible in logs
- Fallback: if no projection found, truncate text_embedding to [:, :1024] as last resort

**Lesson:** Don't assume a TTS model stores separate embedding tables for text and audio, or that embedding dims match the transformer hidden size. Always inspect actual state dict key names and shapes first.

---

## Issue #5 — `'Qwen3TTSModel' object has no attribute 'config'`

**Error:**
```
Megakernel load failed ('Qwen3TTSModel' object has no attribute 'config'); falling back to HF baseline.
```

**Cause:** `Qwen3TTSModel` is a plain wrapper class (not `nn.Module`) — it exposes no `.config`. Our code called `model.config` directly where `model` was the wrapper. `inner = model.model` (the actual `nn.Module`) does have `.config`.

**Fix:** Use `inner.config` instead of `model.config`, with a safe fallback to `model.config` if `inner.config` is absent, and a hardcoded default of `codec_eos_id=2048` if neither is found:

```python
cfg = getattr(inner, "config", None) or getattr(model, "config", None)
if cfg is not None:
    talker_cfg = getattr(cfg, "talker_config", cfg)
    codec_eos_id = getattr(talker_cfg, "codec_eos_token_id",
                   getattr(cfg, "codec_eos_token_id", 2048))
else:
    codec_eos_id = 2048
```

**Also discovered in same session:** No text projection layer exists in the talker's non-layer weights (`norm.weight`, `codec_embedding.weight`, `text_embedding.weight` — that's all). The `text_embedding` is 2048-dim with no 2048→1024 linear anywhere in the talker. This strongly suggests the talker **does not take raw text tokens** — it receives hidden states from the thinker model (which likely has hidden_size=2048). Text prefill by feeding raw token IDs through the talker backbone is architecturally incorrect. A different prefill strategy is needed (run thinker first, pass its output as KV-cache seed).

**Lesson:** Check `.config` on the inner `nn.Module`, not on the vendor wrapper class. When no projection is found between embedding dim and hidden dim, the embedding is not meant to feed that transformer directly.
