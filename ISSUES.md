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
