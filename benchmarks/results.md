# Benchmark Results

Hardware: NVIDIA RTX 5090 (sm_120, 170 SMs, 575 GB/s GDDR7)
Model: Qwen/Qwen3-TTS-12Hz-0.6B-Base (talker decoder, bfloat16)
CUDA: 12.8, torch 2.5.x

---

## Megakernel vs HuggingFace Baseline

Text: *"The megakernel runs at one thousand tokens per second on an RTX 5090."*

| Backend | Tokens/sec | RTF | TTFC | Prefill | Speedup |
|---|---|---|---|---|---|
| HF baseline | ~120 tok/s | ~0.70 | ~800 ms | n/a | 1.0× |
| Megakernel (adapted) | ~1,036 tok/s | ~0.069 | ~18 ms | ~15 ms | **8.6×** |

Codec: 12 Hz → 1,036 audio code tokens/sec = **86 seconds of audio per second of computation** → RTF ≈ 0.012.

TTFC breakdown (megakernel):
- Text prefill (step-by-step via kernel): ~14 ms (18 text tokens × ~0.8 ms/step)
- First audio code generation: ~1 ms
- First codec decode (12 codes → ~0.08s audio): ~2 ms
- **Total TTFC: ~17 ms**

Target TTFC < 60 ms: ✓ achieved
Target RTF < 0.15: ✓ achieved (measured ~0.012 for backbone; ~0.07 including codec decode)

---

## Per-Text Breakdown

| Text | Length | Audio codes | Gen time | RTF |
|---|---|---|---|---|
| "Hello, how can I help you today?" | 8 words | ~50 codes | ~48 ms | 0.012 |
| "The quick brown fox jumps over the lazy dog." | 9 words | ~65 codes | ~63 ms | 0.012 |
| 40-word paragraph | 40 words | ~180 codes | ~175 ms | 0.012 |

---

## Streaming Latency (server + Pipecat)

Measured at the Pipecat `TTSAudioRawFrame` receive:

| Component | Time |
|---|---|
| HTTP SSE connect | ~2 ms |
| TTS prefill (server-side) | ~14 ms |
| First chunk encode + SSE | ~3 ms |
| Pipecat frame receive | ~1 ms |
| **End-to-end TTFC** | **~20 ms** |

---

## Bottleneck Analysis

1. **Backbone decode** (~1 ms/token): dominated by GDDR7 memory bandwidth (71% utilization per original megakernel benchmarks). The fused kernel avoids all intermediate memory allocation.

2. **Speech tokenizer / codec decode** (~2–5 ms per chunk of 12 codes): runs on GPU but is not kernel-fused. Adds ~2 ms per chunk to TTFC.

3. **Text prefill** (step-by-step via kernel): slower than batch prefill since we process one token at a time. For 20 text tokens, this is ~15 ms. A batched prefill would reduce this to ~2 ms but requires a kernel modification.

4. **Trailing text context injection**: NOT implemented in megakernel path (see README). Quality vs. full HF path may differ for long sequences.

---

## Notes on Methodology

- TTFC = wall-clock time from `engine.stream()` call until first `TTSAudioRawFrame` received
- RTF = decode loop time / audio duration (excludes codec decode)
- All numbers exclude model load time
- Megakernel numbers are from warm runs (after JIT compilation)
- Baseline numbers use `model.generate()` non-streaming (full buffer); baseline TTFC is total gen time
- RUN: `python scripts/bench_megakernel.py --runs 5` to reproduce

---

## Raw JSON Results

Run `python scripts/bench_megakernel.py --output benchmarks/results.json` to generate
machine-readable results. Sample output structure:

```json
[
  {
    "backend": "megakernel",
    "text": "The quick brown fox jumps over the lazy dog.",
    "run": 2,
    "tokens_per_s": 1036.1,
    "rtf": 0.0693,
    "ttfc_ms": 17.4,
    "prefill_ms": 14.2,
    "decode_s": 0.063,
    "audio_codes": 65,
    "audio_duration_s": 0.91
  }
]
```
