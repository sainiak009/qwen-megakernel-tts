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

**Also discovered in same session:** The non-layer search only covered `talker.model.*`. The actual projection (`talker.text_projection`) sits one level up under `talker.*`, which is why we missed it. See Issue #8 for the full architecture discovery.

**Lesson:** Check `.config` on the inner `nn.Module`, not on the vendor wrapper class. When no projection is found between embedding dim and hidden dim, search one prefix level higher in the module hierarchy.

---

## Issue #6 — Empty WAV (44 bytes) + `speech_tokenizer.decode()` TypeError — full architecture misunderstanding

**Symptoms:**
1. Server returns HTTP 200 with a 44-byte WAV (header only, zero audio frames)
2. Server logs: `TypeError: encoded must be an encode output, a dict, or a list of dicts.`

**Root causes (three layers, resolved together):**

### Layer 1 — Wrong text prefill
The talker's `text_embedding` is [151936, **2048**] but the transformer backbone expects **1024**-dim input. We were truncating the 2048-dim embedding to 1024 as a last resort, producing garbage hidden states. The first predicted audio token was immediately EOS (id=2150) → zero codes generated → empty WAV.

**Real fix:** The talker has `talker.text_projection` — a 2-layer MLP [2048→2048→1024] — that was missed because the diagnostic only searched `talker.model.*` keys. The correct prefill:
```python
projected = talker.text_projection(talker.model.text_embedding(text_ids))  # [N, 1024]
```

These projected vectors are also stored as `trailing_text_hidden` and added to the kernel input at each decode step (text conditioning is **per-step**, not just prefill).

### Layer 2 — Missing code_predictor step
The speech tokenizer requires **all 16 codebooks** to decode to audio. The talker backbone generates only the **first codebook** autoregressively. After each first-codebook token, `talker.code_predictor.generate()` must be called to produce the 15 residual codebook codes. We were skipping this entirely.

Config revealed: `num_code_groups=16`, so 15 residual codes per step.

### Layer 3 — Wrong speech_tokenizer.decode() call format
The 12Hz speech tokenizer expects: `decode({"audio_codes": tensor[num_codebooks, num_frames]})` — a dict, not a raw tensor. Returns `(wavs: List[np.ndarray], sample_rate: int)`.

**Full per-step flow (now correctly implemented):**
```
1. codec_embed = audio_embed[first_code]               [1024]
2. pred = code_predictor.generate(inputs_embeds=codec_embed, max_new_tokens=15)
3. step_embed = codec_embed + Σ predictor_embed[i](pred.sequences[i])  [1024]
4. step_embed += trailing_text_hidden[decode_step]     [1024]  ← text conditioning
5. next_code = megakernel(_step_with_embed(step_embed))
6. all_codes[step] = cat([first_code, pred.sequences])  [16]
   ↓ (every chunk_codes steps)
7. speech_tokenizer.decode({"audio_codes": codes_tensor[16, chunk]}) → audio
```

**Kernel trick for step 5:** Kernel only accepts `(token_id, embed_table)`. To pass an arbitrary 1024-dim vector without kernel modification: pass `token_id=0` with a 1-row temp table `[1, 1024]` containing the precomputed embedding.

**Fix:** Complete rewrite of `TalkerDecoder` in `generate_megakernel_tts.py` and `_decode_codes_to_audio` + `_stream_megakernel` in `server/qwen_tts_engine.py`.

**Lesson:** Read the vendor's `generate()` source before assuming the architecture. The per-step text conditioning and multi-codebook structure were only visible by reading `modeling_qwen3_tts.py` directly on the instance.

---

## Issue #7 — `qwen_megakernel not installed` — PYTHONPATH not exported at server start

**Error:**
```
Megakernel load failed (qwen_megakernel not installed. Clone from github.com/AlpinDale/qwen_megakernel
and add to PYTHONPATH.); falling back to HF baseline.
```

**Cause:** `qwen_megakernel` has no `setup.py`/`pyproject.toml` and is not pip-installable. It must be on
`PYTHONPATH`. When `uvicorn` is launched from a new shell without `export PYTHONPATH=...`, the import
fails silently and the engine falls back to the HF baseline.

**Fix:** Created `start.sh` at the project root. It:
1. Validates that `../qwen_megakernel` exists and exits with a clear message if not.
2. Exports `PYTHONPATH` prepended with the megakernel directory.
3. Kills any existing listener on the target port.
4. Launches `uvicorn` via `exec`.

```bash
bash start.sh            # default: host=0.0.0.0, port=8080
PORT=9090 bash start.sh  # custom port
```

**Lesson:** Whenever a dependency must be on `PYTHONPATH`, bake that export into the project's startup
script rather than relying on shell state. A missing `PYTHONPATH` entry is invisible from error messages
until the failing import bubbles up.

---

## Issue #8 — `ValueError: tts_model_type: base does not support generate_custom_voice`

**Error:**
```
ValueError: model with tokenizer_type: qwen3_tts_tokenizer_12hz, tts_model_size: 0b6,
tts_model_type: base does not support generate_custom_voice,
Please check Model Card or Readme for more details.
```

**Cause:** The qwen-tts library has three completely separate generation APIs, each locked to a specific
`tts_model_type`:

| API | tts_model_type |
|-----|---------------|
| `generate_custom_voice()` | `custom_voice` |
| `generate_voice_clone()` | `base` |
| `generate_voice_design()` | `voice_design` |

`Qwen3-TTS-12Hz-0.6B-Base` reports `tts_model_type: base`. Our HF baseline path called
`generate_custom_voice()` unconditionally → instant `ValueError`.

**Root cause of confusion:** the model type naming is counterintuitive — "Base" sounds generic but maps to
the voice-cloning variant of the API, not a plain TTS call.

**Fix:** In `_stream_hf()` (`server/qwen_tts_engine.py`), detect `tts_model_type` at runtime and dispatch:

```python
tts_model_type = getattr(getattr(model, "model", None), "tts_model_type", "custom_voice")

if tts_model_type == "base":
    # Base model: generate_voice_clone() with x_vector_only_mode=True.
    # Synthetic noise clip used as reference — extracts a random speaker embedding.
    ref_audio = (np.random.randn(24000).astype(np.float32) * 0.05, 24000)
    audio_list, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        x_vector_only_mode=True,
    )
else:
    audio_list, sr = model.generate_custom_voice(
        text=text,
        language="English",
        speaker="default",
    )
```

`x_vector_only_mode=True` tells the model to use only the speaker embedding (x-vector) from the reference
audio and ignore any reference codes. The synthetic noise clip yields a noisy/random speaker identity,
which is acceptable for the fallback path — the output is intelligible, just not a specific voice.

**Note:** This fix is only needed when the megakernel path fails. Normally `start.sh` ensures
`PYTHONPATH` is set and the megakernel path loads successfully, bypassing the HF baseline entirely.

**Lesson:** Always check the vendor API docs / source for model-type guards before calling a generation
method. The `tts_model_type` attribute lives on `model.model` (the inner `Qwen3TTSForConditionalGeneration`),
not on the `Qwen3TTSModel` wrapper.

---

## Issue #9 — Empty WAV (44 bytes) — wrong prefill embed sequence

**Symptom:** Server returns HTTP 200 with a 44-byte WAV (header only, no audio frames). No error in logs.

**Root cause:** `prefill_text()` was feeding `text_projection(text_embedding(raw_text_ids))` directly into
the kernel — just the raw text token projections one-by-one. The actual talker backbone expects a specific
**8-token composite prefill sequence** that mixes text and codec embeddings. Without this structure, the
backbone never enters "audio generation mode" and predicts EOS (token 2150) as the first audio code,
causing `generate_audio_codes_iter()` to return immediately with zero codes.

**Root cause discovery:** Read `Qwen3TTSForConditionalGeneration.generate()` in
`modeling_qwen3_tts.py` line 2022+. The actual embed sequence passed to `talker.generate()` is:

```
[0] text_proj(<|im_start|>)
[1] text_proj(assistant)
[2] text_proj(\n)
[3] text_proj(tts_pad=151671) + audio_embed(codec_nothink=2155)   ← summed
[4] text_proj(tts_pad=151671) + audio_embed(codec_think_bos=2156) ← summed
[5] text_proj(tts_pad=151671) + audio_embed(codec_think_eos=2157) ← summed
[6] text_proj(tts_bos=151672) + audio_embed(codec_pad=2148)       ← summed
[7] text_proj(text_token[0]) + audio_embed(codec_bos=2149)        ← summed
```

After these 8 tokens, autoregressive decode begins. Per-step text conditioning:
```
trailing_text_hidden[i] = text_proj(text_token[i+1])  for i < len(text_body)-1
trailing_text_hidden[-1] = text_proj(tts_eos=151673)
```

**Additional required change:** input text must be formatted before tokenizing:
```python
# WRONG (raw text):
text_ids = tokenizer.encode(text, add_special_tokens=True)

# CORRECT (chat template):
formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
text_ids  = tokenizer.encode(formatted, add_special_tokens=False)
```
This produces exactly 3 role-prefix tokens + N text-body tokens + 5 trailing tokens.

**Fix:**
- `load_tts_weights()`: extract 7 additional token IDs from config (`codec_pad_id`, `codec_nothink_id`,
  `codec_think_bos_id`, `codec_think_eos_id`, `tts_bos_token_id`, `tts_eos_token_id`, `tts_pad_token_id`).
- `TalkerDecoder`: store the new IDs.
- `prefill_text()`: completely rewritten to build the 8-token composite embed sequence above.
- `_stream_megakernel()` (engine): use `_format_tts_text(text)` and `add_special_tokens=False`.
- `benchmark()` / `main()`: same text format fix.

**Lesson:** The talker backbone does not accept plain text tokens — it requires a chat-formatted sequence
where codec tag embeddings are **summed** with TTS special token projections. This conditioning pattern
is only visible by reading `modeling_qwen3_tts.py` line 2124–2232 in full.
