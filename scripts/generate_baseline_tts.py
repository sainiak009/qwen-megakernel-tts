#!/usr/bin/env python3
"""
generate_baseline_tts.py

Baseline TTS generation using qwen-tts (Alibaba's official package).
Measures TTFC (time to first chunk), RTF, and generation time.

Usage:
    python scripts/generate_baseline_tts.py
    python scripts/generate_baseline_tts.py --text "Hello, world!" --output out.wav
"""

import argparse
import time
import wave

import numpy as np
import torch


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SAMPLE_RATE = 24000
CODEC_HZ = 12


def load_model(model_id: str = MODEL_ID):
    """Load Qwen3-TTS using qwen-tts package."""
    from qwen_tts import Qwen3TTSModel

    print(f"Loading {model_id} ...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, None   # no separate processor needed with qwen-tts


def generate_audio(model, processor, text: str, max_new_tokens: int = 1024):
    """
    Generate audio from text using Qwen3TTSModel.generate_custom_voice().

    Returns (audio_np, sample_rate, metrics_dict).
    """
    t_start = time.perf_counter()

    audio_list, sr = model.generate_custom_voice(
        text=text,
        language="English",
        speaker="default",
    )

    t_end = time.perf_counter()

    # Flatten audio list to numpy array
    if isinstance(audio_list, (list, tuple)):
        audio = np.concatenate([np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0])
    else:
        audio = np.array(audio_list, dtype=np.float32).squeeze()

    gen_time_s = t_end - t_start
    audio_duration_s = len(audio) / sr
    rtf = gen_time_s / audio_duration_s if audio_duration_s > 0 else float("inf")
    # Baseline is non-streaming: TTFC = total generation time
    ttfc_ms = gen_time_s * 1000

    metrics = {
        "ttfc_ms":          round(ttfc_ms, 1),
        "gen_time_s":       round(gen_time_s, 3),
        "audio_duration_s": round(audio_duration_s, 3),
        "tokens_per_s":     round(audio_duration_s * CODEC_HZ / gen_time_s, 1) if gen_time_s > 0 else 0.0,
        "rtf":              round(rtf, 4),
    }
    return audio, sr, metrics


def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="The megakernel runs at one thousand tokens per second on an RTX 5090.")
    parser.add_argument("--output", default="baseline_output.wav")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    model, processor = load_model(args.model)

    print(f"\nText: {args.text!r}")
    all_metrics = []
    for i in range(args.runs):
        print(f"\n--- Run {i+1}/{args.runs} ---")
        audio, sr, metrics = generate_audio(model, processor, args.text)
        all_metrics.append(metrics)
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if args.runs > 1:
        print("\n--- Average metrics ---")
        for k in all_metrics[0]:
            avg = sum(m[k] for m in all_metrics) / len(all_metrics)
            print(f"  {k}: {avg:.3f}")

    save_wav(args.output, audio, sr)
    print(f"\nSaved to {args.output}")
    print("\nNote: baseline is non-streaming. TTFC = total generation time.")


if __name__ == "__main__":
    main()
