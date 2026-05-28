# qwen-megakernel-tts

Adapts [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) for
[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) talker decoder,
with a minimal FastAPI streaming server and a Pipecat voice pipeline demo.

---

## Status

**HF baseline path (`USE_MEGAKERNEL=0`) is the working default.** It produces
valid audio end-to-end (~1.1s for "Hello world" at 24 kHz).

**Megakernel path is currently broken** — kept behind `USE_MEGAKERNEL=1` for
further investigation. Failure analysis below.

### Megakernel failure analysis (May 2026)

After integrating the kernel with the talker decoder (text projection + code
predictor + composite step embeddings + summed prefill), end-to-end runs fail
during autoregressive decode. Observed:

- **HF backend** produces valid audio (verified: 26,880 frames / 1.12 s for
  "Hello world.").
- **Megakernel backend** reaches prefill correctly (8 prefill steps, valid
  output tokens, sane hidden-state norms ~214), then fails at the first
  autoregressive decode step. Resulting WAV is either 44 bytes (header only)
  or a single ~0.08 s chunk of unstable amplitude.
- **KV cache positions 0–7 are not retroactively corrupted** (verified by
  snapshot/diff before and after the failing kernel call: `max_diff = 0`).
- **Bad K values appear at the freshly-written position** of the failing call
  — magnitudes around `1e32`, large enough to make `softmax(Q·K)` undefined
  for subsequent layers.
- **Failure is non-deterministic.** Different Python sync patterns (extra
  `.item()` calls, `torch.stack(...).cpu()` reductions between launches,
  `torch.cuda.synchronize()` placement) shift which layer first goes bad —
  sometimes layer 0, sometimes layer 1, sometimes prefill itself.
- **`LDG_LM_NUM_BLOCKS` revert (24 → 1280) does not fix it.** That patch was
  the only tuning change made to the LM-head launch, and reverting it leaves
  the same non-deterministic failure pattern.

Most likely cause: a race condition in the persistent decode kernel's atomic
grid-sync barriers (`barrier_counter`, `barrier_sense`, `kv_flag`, `attn_flag`).
The kernel already includes a host-side pre-launch barrier reset
(`cudaMemsetAsync` + `cudaDeviceSynchronize` in `launch_ldg_decode_direct`),
but the cross-launch state is still sensitive to allocator/stream timing.

Pinning the exact (layer, op) where NaN first appears requires kernel-side
instrumentation and a CUDA extension rebuild — deferred until the demo,
documentation, and submission path are complete.

### Recommended setting

```bash
export USE_MEGAKERNEL=0   # HF baseline; default in app.py
```

---

## Architecture Decision

### Why the backbone maps 1:1

The Qwen3-TTS talker decoder IS a Qwen3-0.6B transformer. Every layer
(RMSNorm → GQA attention with q/k_norm → SwiGLU MLP) is byte-for-byte the same
shape. The megakernel's fused CUDA kernel runs these layers unchanged.

### What had to change (Python-only, no kernel modifications)

| Difference | Qwen3-0.6B (megakernel) | Qwen3-TTS talker | Fix |
|---|---|---|---|
| RoPE theta | 10,000 | 1,000,000 | Recompute cos/sin tables in Python |
| KV cache size | 2,048 | 4,096 | Reallocate KV buffers |
| Output vocab | 151,936 (text) | 3,072 (audio codes) | Pad codec_head to [151936, 1024] with zeros; argmax still correct |
| Input embed | 151,936 text tokens | 3,072 audio codes | Swap embed table: text during prefill, audio during decode |
| LM head tied | Yes | No (separate codec_head) | Load codec_head instead of embed_tokens |

### What was NOT modified

- `kernel.cu` — zero bytes changed
- `code_predictor` (residual codebook generator) — runs in PyTorch, untouched

### Honest limitation

The megakernel was designed for **text generation** (token_id → argmax).
The TTS talker's full forward pass adds:

1. **Text context injection**: each decode step adds `trailing_text_hiddens[:, gen_step]`
   to the input embedding. This is **not implemented** in the megakernel path — we generate
   audio codes conditioned only on the KV-cached text prefill.
2. **Residual codebooks**: `code_predictor` generates 15 additional codebook tokens
   per step. This runs in PyTorch alongside the megakernel backbone.

For maximum quality, use the HuggingFace path (`USE_MEGAKERNEL=0`).
For maximum speed with acceptable quality, use the megakernel path.

---

## Performance Targets

| Metric | Target | Expected (megakernel) |
|---|---|---|
| TTFC (time to first audio chunk) | < 60 ms | ~15–35 ms (prefill only) |
| RTF (gen_time / audio_duration) | < 0.15 | ~0.001 at 1036 tok/s |
| Tokens/sec | — | ~1,000 (backbone @ RTX 5090) |

Numbers assume RTX 5090, CUDA 12.8, bf16. See `benchmarks/results.md` for measured data.

---

## Setup

```bash
# 1. Clone megakernel (no setup.py — must clone manually)
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel && pip install -r requirements.txt && cd ..
export PYTHONPATH="$(pwd)/qwen_megakernel:$PYTHONPATH"

# 2. Clone this repo and install dependencies
git clone <this-repo-url>
cd qwen-megakernel-tts
pip install -r requirements.txt
```

---

## Inspect architectures

```bash
# Static comparison (no GPU required)
python scripts/inspect_qwen_tts.py

# Live: load model and print actual weight names
python scripts/inspect_qwen_tts.py --live
```

---

## Generate audio

```bash
# HuggingFace baseline
python scripts/generate_baseline_tts.py --text "Hello world" --output baseline.wav

# Megakernel-adapted (requires RTX 5090)
python scripts/generate_megakernel_tts.py --text "Hello world" --output mk.wav
```

---

## Benchmark

```bash
# Head-to-head: megakernel vs baseline
python scripts/bench_megakernel.py --runs 5

# Megakernel only (faster)
python scripts/bench_megakernel.py --no-baseline --runs 5
```

---

## Streaming server

```bash
# Start server (pre-loads model)
uvicorn server.app:app --host 0.0.0.0 --port 8080

# Test with curl
curl -X POST http://localhost:8080/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello from the megakernel!"}' \
     -N  # follow SSE stream

# Non-streaming WAV
curl -X POST http://localhost:8080/tts/wav \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output out.wav

# Health check
curl http://localhost:8080/health
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Model ID |
| `USE_MEGAKERNEL` | `0` | `0` = HF baseline (working). `1` = megakernel (experimental, currently broken — see Status). |
| `CHUNK_CODES` | `12` | Audio codes per SSE chunk (1 sec at 12 Hz) |
| `PRELOAD_MODEL` | `1` | Load on startup vs first request |

---

## Pipecat demo

Targets **pipecat-ai 1.2.x**. Install with the transport/STT/LLM extras:

```bash
pip install 'pipecat-ai[daily,deepgram,openai,silero]'
```

Then:

```bash
# 1. Start the TTS server
uvicorn server.app:app --port 8080 &

# 2. Set credentials
export DAILY_API_KEY=...
export DAILY_ROOM_URL=https://your-domain.daily.co/your-room
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...

# 3. Run the bot
python pipecat_demo/bot.py
```

The pipeline:
`Mic → VAD (Silero) → Deepgram STT → GPT-4o-mini → QwenTTSService → Speaker`

`QwenTTSService` pushes `TTSAudioRawFrame` to Pipecat as each PCM chunk arrives
from the SSE stream — audio is never buffered before sending.

### VAD wiring (pipecat 1.2)

VAD moved out of `DailyParams` in pipecat 1.2 — it is now a pipeline stage.
`bot.py` constructs a `SileroVADAnalyzer` and wraps it in a `VADProcessor`
inserted right after `transport.input()`, so downstream stages only see
voiced segments:

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor

vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=SAMPLE_RATE))

pipeline = Pipeline([
    transport.input(),
    vad,                            # ← VAD here, before STT
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

The Silero model is downloaded on first run; `sample_rate=24000` matches the
incoming Daily audio track.

### Verification

The full pipeline is launch-tested. With dummy credentials, `bot.py`:

1. Imports cleanly against pipecat 1.2.1 (no deprecation warnings).
2. Constructs `DailyTransport`, `DeepgramSTTService`, `OpenAILLMService`
   (`OpenAILLMService.Settings(model="gpt-4o-mini")`), `VADProcessor`
   (`SileroVADAnalyzer(sample_rate=24000)`), `QwenTTSService`.
3. Composes the 8-stage pipeline above.
4. Starts `PipelineRunner.run(task)`.
5. Reaches Deepgram's websocket, gets **HTTP 401 "Token dummy"** — the live
   handshake completed; only the dummy key is rejected.

That 401 is the strongest signal short of real credentials: everything is
wired. To go fully live, swap the four env vars for real keys.

---

## File structure

```
qwen-megakernel-tts/
  scripts/
    inspect_qwen_tts.py          # Architecture comparison + weight mapping
    generate_baseline_tts.py     # HF baseline generation + metrics
    generate_megakernel_tts.py   # Adapted megakernel TalkerDecoder + benchmark
    bench_megakernel.py          # Head-to-head benchmark
  server/
    qwen_tts_engine.py           # QwenTTSEngine (streaming, both backends)
    app.py                       # FastAPI SSE server
  pipecat_demo/
    bot.py                       # Pipecat STT→LLM→TTS pipeline
  benchmarks/
    results.md                   # Measured performance numbers
```
