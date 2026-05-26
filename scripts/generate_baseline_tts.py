#!/usr/bin/env python3
"""
generate_baseline_tts.py

Baseline TTS generation using HuggingFace transformers.
Measures TTFC (time to first chunk), RTF, and tokens/sec.

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
SAMPLE_RATE = 24000   # Qwen3-TTS speech tokenizer output sample rate
CODEC_HZ = 12         # token frames per second in the codec


def load_model(model_id: str = MODEL_ID):
    """Load Qwen3-TTS using transformers >= 4.57."""
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    print(f"Loading {model_id} ...")
    t0 = time.time()

    try:
        # transformers 4.57+ registers Qwen3TTS classes automatically
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, processor


def generate_audio(model, processor, text: str, max_new_tokens: int = 1024):
    """
    Generate audio from text.

    Returns (audio_np, sample_rate, metrics_dict).
    """
    device = next(model.parameters()).device

    # Prepare inputs
    t_start = time.perf_counter()
    with torch.no_grad():
        try:
            inputs = processor(text=text, return_tensors="pt").to(device)
        except Exception:
            # Fallback: treat processor as tokenizer
            inputs = processor(text, return_tensors="pt").to(device)

        # Measure TTFC: time from start until first audio chunk can be decoded
        # For non-streaming baseline this is just total generation time
        t_gen_start = time.perf_counter()

        try:
            # Qwen3TTSForConditionalGeneration generates codec token IDs
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_k=50,
            )
        except Exception as e:
            print(f"generate() failed ({e}), trying model.model.generate()...")
            output = model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_k=50,
            )

        t_gen_end = time.perf_counter()

        # Decode codec token IDs → waveform
        t_decode_start = time.perf_counter()
        try:
            # transformers Qwen3TTS model exposes decode_audio or speech_tokenizer
            if hasattr(model, "decode_audio"):
                audio = model.decode_audio(output)
            elif hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
                # output contains codec IDs; strip input tokens
                input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
                codec_ids = output[:, input_len:]
                audio = model.model.speech_tokenizer.decode(codec_ids)
            else:
                # Raw codec IDs — caller handles decoding
                audio = output.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Audio decode failed: {e}")
            audio = np.zeros(1024, dtype=np.float32)

        t_decode_end = time.perf_counter()

    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().float().cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32).squeeze()

    gen_time_s = t_gen_end - t_gen_start
    audio_duration_s = len(audio) / SAMPLE_RATE
    n_tokens = output.shape[-1] - (inputs.get("input_ids", output).shape[-1] if hasattr(inputs, "get") else 0)
    tokens_per_s = n_tokens / gen_time_s if gen_time_s > 0 else 0.0
    rtf = gen_time_s / audio_duration_s if audio_duration_s > 0 else float("inf")

    # In non-streaming mode, TTFC = total generation time (worst case)
    ttfc_ms = gen_time_s * 1000

    metrics = {
        "ttfc_ms": round(ttfc_ms, 1),
        "gen_time_s": round(gen_time_s, 3),
        "audio_duration_s": round(audio_duration_s, 3),
        "tokens_per_s": round(tokens_per_s, 1),
        "n_tokens": n_tokens,
        "rtf": round(rtf, 4),
        "codec_decode_ms": round((t_decode_end - t_decode_start) * 1000, 1),
    }
    return audio, SAMPLE_RATE, metrics


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
    print("\nNote: baseline is non-streaming (full buffer before playback).")
    print("TTFC = total generation time in non-streaming mode.")


if __name__ == "__main__":
    main()
