#!/usr/bin/env python3
"""
bench_megakernel.py

Head-to-head benchmark: HuggingFace baseline vs. megakernel-adapted TTS.

Measures and reports:
  - Tokens/sec (audio code generation rate)
  - TTFC: time to first audio chunk
  - RTF: real-time factor (gen_time / audio_duration)
  - End-to-end latency

Usage:
    python scripts/bench_megakernel.py
    python scripts/bench_megakernel.py --text "Hello world" --runs 5
"""

import argparse
import json
import os
import sys
import time

import torch

# Ensure project root is on sys.path so 'scripts.*' imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXTS = [
    "Hello, how can I help you today?",
    "The quick brown fox jumps over the lazy dog.",
    "The megakernel runs at one thousand tokens per second on an RTX 5090.",
    "Artificial intelligence is transforming the way we build voice assistants.",
]

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CODEC_HZ  = 12
SAMPLE_RATE = 24_000


def bench_baseline(model, processor, text: str, runs: int = 3) -> list[dict]:
    from scripts.generate_baseline_tts import generate_audio  # noqa: PLC0415

    results = []
    for i in range(runs):
        _, _, metrics = generate_audio(model, processor, text)
        metrics["backend"] = "hf_baseline"
        metrics["text"] = text
        results.append(metrics)
        print(f"  [baseline] run={i+1}  {metrics['tokens_per_s']:.0f} tok/s  RTF={metrics['rtf']:.4f}  TTFC={metrics['ttfc_ms']:.0f}ms")
    return results


def bench_megakernel(decoder, tokenizer, text: str, runs: int = 3) -> list[dict]:
    text_ids = tokenizer.encode(text, add_special_tokens=True)

    results = []
    for i in range(runs):
        decoder.reset()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        first_code = decoder.prefill_text(text_ids)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        codes = decoder.generate_audio_codes(first_code)
        torch.cuda.synchronize()
        t_decode = time.perf_counter()

        prefill_ms = (t_prefill - t0) * 1000
        decode_s   = t_decode - t_prefill
        n_codes    = len(codes)
        audio_dur  = n_codes / CODEC_HZ
        tok_per_s  = n_codes / decode_s if decode_s > 0 else 0
        rtf        = decode_s / audio_dur if audio_dur > 0 else float("inf")

        # TTFC = prefill + first token
        ttfc_ms = prefill_ms + (1000 / tok_per_s if tok_per_s > 0 else 0)

        m = {
            "backend":        "megakernel",
            "text":           text,
            "run":            i + 1,
            "tokens_per_s":   round(tok_per_s, 1),
            "rtf":            round(rtf, 4),
            "ttfc_ms":        round(ttfc_ms, 1),
            "prefill_ms":     round(prefill_ms, 1),
            "decode_s":       round(decode_s, 3),
            "audio_codes":    n_codes,
            "audio_duration_s": round(audio_dur, 2),
        }
        results.append(m)
        print(f"  [megakernel] run={i+1}  {tok_per_s:.0f} tok/s  RTF={rtf:.4f}  TTFC={ttfc_ms:.0f}ms  ({n_codes} codes)")
    return results


def summarize(results: list[dict], label: str):
    warm = results[1:] if len(results) > 1 else results
    avg = lambda k: sum(r[k] for r in warm) / len(warm)
    print(f"\n  {label} (avg over {len(warm)} warm runs):")
    print(f"    tokens/s:   {avg('tokens_per_s'):.1f}")
    print(f"    RTF:        {avg('rtf'):.4f}")
    print(f"    TTFC:       {avg('ttfc_ms'):.1f} ms")
    if "prefill_ms" in warm[0]:
        print(f"    prefill:    {avg('prefill_ms'):.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",  default=None, help="Single text to benchmark (default: all TEXTS)")
    parser.add_argument("--runs",  type=int, default=3)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", default="benchmarks/results.json")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    texts = [args.text] if args.text else TEXTS

    all_results = []

    # ── Megakernel ────────────────────────────────────────────────────────────
    print("\n=== Megakernel backend ===")
    from scripts.generate_megakernel_tts import TalkerDecoder
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    decoder   = TalkerDecoder(model_id=args.model, verbose=True)

    for text in texts:
        print(f"\nText: {text!r}")
        mk_results = bench_megakernel(decoder, tokenizer, text, args.runs)
        all_results.extend(mk_results)
        summarize(mk_results, "megakernel")

    # ── Baseline ─────────────────────────────────────────────────────────────
    if not args.no_baseline:
        print("\n\n=== HuggingFace baseline ===")
        from scripts.generate_baseline_tts import load_model, generate_audio

        model, processor = load_model(args.model)
        for text in texts:
            print(f"\nText: {text!r}")
            bl_results = bench_baseline(model, processor, text, args.runs)
            all_results.extend(bl_results)
            summarize(bl_results, "hf_baseline")

        # ── Speedup summary ───────────────────────────────────────────────────
        print("\n\n=== Speedup Summary ===")
        for text in texts:
            mk = [r for r in all_results if r["backend"] == "megakernel"   and r["text"] == text]
            bl = [r for r in all_results if r["backend"] == "hf_baseline"  and r["text"] == text]
            if mk and bl:
                mk_warm = mk[1:] if len(mk) > 1 else mk
                bl_warm = bl[1:] if len(bl) > 1 else bl
                mk_tps = sum(r["tokens_per_s"] for r in mk_warm) / len(mk_warm)
                bl_tps = sum(r["tokens_per_s"] for r in bl_warm) / len(bl_warm)
                speedup = mk_tps / bl_tps if bl_tps > 0 else float("inf")
                print(f"  {text[:50]!r:52}  {speedup:.2f}x tok/s speedup")

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
