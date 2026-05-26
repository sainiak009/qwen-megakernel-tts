# Take-Home Project Brief: RTX 5090 Decode Megakernel → Qwen3-TTS on Pipecat

## Goal
Wire AlpinDale's `qwen_megakernel` (~1,200-line CUDA kernel, ~1,000 tok/s on RTX 5090) as the LLM decode backend for **Qwen3-TTS's talker decoder** (NOT the codebook generator), streaming real-time speech into a Pipecat voice pipeline.

## Performance Targets
| Metric | Target | Notes |
|--------|--------|-------|
| TTFC (time to first audio chunk) | < 60 ms (reference: < 50 ms) | |
| RTF (real-time factor) | < 0.15 (reference: < 0.1) | 1s audio must generate in < 150 ms |
| Streaming | **Required** | Push audio chunks as decoded — do NOT buffer full utterance |

These are reference benchmarks for a good submission, not hard pass/fail cutoffs — but explain if you're way off.

## Steps
1. **Adapt megakernel** — clone `github.com/AlpinDale/qwen_megakernel`, wire to Qwen3-TTS talker decoder backbone (same Qwen3 architecture)
2. **Inference server** — streaming interface: prompt in → token stream out
3. **Pipecat integration** — pipeline: STT → LLM → TTS service → audio output
4. **Validate end-to-end** — round-trip test: speak → transcribe → LLM response → TTS → audio playback

## Deliverables
- Working repo with build instructions (single RTX 5090)
- README: architecture decisions, kernel modifications, how to run Pipecat demo
- **Performance numbers**: decode tok/s, TTFC, RTF, end-to-end latency
- **Demo recording** of the voice agent working end-to-end

## What's Being Evaluated
1. **Ramp-up speed** — getting up to speed on CUDA kernels, TTS pipelines, Pipecat
2. **Performance rigor** — thorough, honest benchmarking; real numbers + methodology + bottleneck analysis
3. **Coding agent proficiency** — effective use of Claude Code / Codex
4. **Communication** — clear README, honest about rough edges

## Key Technical Facts
- Megakernel: 128 persistent thread blocks × 512 threads, single non-cooperative kernel, bfloat16 only (no quantization)
- Model backbone: Qwen3-0.6B architecture (28 layers, 1024 hidden, 16Q/8KV heads, 3072 FFN)
- Qwen3-TTS talker decoder uses **same backbone** as Qwen3-0.6B — differences are Python-only (RoPE theta 10k→1M, audio vocab 3072 vs 151936, separate codec_head)
- RTX 5090 required (sm_120 / Blackwell); rent on Vast.ai — compute costs reimbursed
- If talker backbone differs from 0.6B, document kernel changes and why

## Reference Links
- Blog: `blog.alpindale.net/posts/5090_decode_optimization/`
- Megakernel source: `github.com/AlpinDale/qwen_megakernel`
- Pipecat docs: `docs.pipecat.ai`
- Model: `huggingface.co/Qwen/Qwen3-TTS`

## Bonus
Find a way to improve the megakernel's performance during integration → document it.
