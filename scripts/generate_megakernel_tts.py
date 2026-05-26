#!/usr/bin/env python3
"""
generate_megakernel_tts.py

Adapts AlpinDale/qwen_megakernel for Qwen3-TTS-12Hz-0.6B-Base talker decoder.

Architecture adaptation (see inspect_qwen_tts.py for full comparison):
  ─ Backbone: 28 transformer layers are IDENTICAL → zero kernel changes
  ─ RoPE: theta 10_000 → 1_000_000, recomputed in Python
  ─ KV cache: max_seq_len 2048 → 4096, reallocated
  ─ Audio LM head: codec_head [3072,1024] padded to [151936,1024] with zeros
      → kernel's argmax returns correct audio code ID (0..3071) as long
        as at least one audio logit > 0, which holds for a trained model
  ─ Embedding swap: text embed during prefill, audio embed during decode

NOT modified:
  ─ kernel.cu / csrc — not a single byte changed
  ─ code_predictor (codebook generator) — runs in PyTorch, untouched

Usage (requires RTX 5090 + CUDA 12.8 + qwen_megakernel installed):
    python scripts/generate_megakernel_tts.py
    python scripts/generate_megakernel_tts.py --text "Hello!" --output mk_out.wav
"""

import argparse
import math
import struct
import time
import wave

import numpy as np
import torch

# ── Megakernel constants (matching kernel.cu) ────────────────────────────────
NUM_LAYERS        = 28
NUM_KV_HEADS      = 8
NUM_Q_HEADS       = 16
HEAD_DIM          = 128
HIDDEN_SIZE       = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE            = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE           = NUM_KV_HEADS * HEAD_DIM  # 1024
KERNEL_VOCAB_SIZE = 151936   # hardcoded in kernel.cu — we pad our LM head to this
TTS_AUDIO_VOCAB   = 3072     # Qwen3-TTS talker audio codebook size
TTS_ROPE_THETA    = 1_000_000.0
TTS_MAX_SEQ_LEN   = 4096     # increased from 2048

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SAMPLE_RATE = 24_000
CODEC_HZ    = 12


def _compute_rope_tables(max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute RoPE cos/sin tables for given theta (TTS uses 1e6, base uses 1e4)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    """Pack the 11-tensor-per-layer flat list into the struct blob the kernel expects."""
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _find_talker_prefix(state_dict: dict) -> str:
    """Dynamically detect the key prefix for talker transformer layers."""
    target = "layers.0.self_attn.q_proj.weight"
    for key in state_dict:
        if key.endswith(target):
            return key[: -len(target)]
    raise RuntimeError(
        f"Cannot find '{target}' in state_dict. Keys: {list(state_dict.keys())[:10]}"
    )


def load_tts_weights(model_id: str = MODEL_ID, verbose: bool = True):
    """
    Load Qwen3-TTS talker weights and remap to megakernel layout.

    Returns weights dict compatible with the megakernel Decoder constructor,
    plus the speech_tokenizer for audio decoding.
    """
    from qwen_tts import Qwen3TTSModel

    if verbose:
        print(f"Loading {model_id} ...")
    t0 = time.time()

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.bfloat16,
    )

    if verbose:
        print(f"  Loaded in {time.time() - t0:.1f}s")

    # Qwen3TTSModel is a wrapper class, not nn.Module — get the inner model
    inner = model.model if hasattr(model, "model") else model
    sd = inner.state_dict()
    prefix = _find_talker_prefix(sd)
    if verbose:
        print(f"  Talker transformer prefix: '{prefix}'")

    # ── 1. Per-layer backbone weights (identical layout to Qwen3-0.6B) ──────
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"{prefix}layers.{i}."
        layer_weights.extend([
            sd[p + "input_layernorm.weight"].contiguous(),
            sd[p + "self_attn.q_proj.weight"].contiguous(),
            sd[p + "self_attn.k_proj.weight"].contiguous(),
            sd[p + "self_attn.v_proj.weight"].contiguous(),
            sd[p + "self_attn.q_norm.weight"].contiguous(),
            sd[p + "self_attn.k_norm.weight"].contiguous(),
            sd[p + "self_attn.o_proj.weight"].contiguous(),
            sd[p + "post_attention_layernorm.weight"].contiguous(),
            sd[p + "mlp.gate_proj.weight"].contiguous(),
            sd[p + "mlp.up_proj.weight"].contiguous(),
            sd[p + "mlp.down_proj.weight"].contiguous(),
        ])

    # ── 2. Final norm ─────────────────────────────────────────────────────────
    # prefix is e.g. 'talker.model.' (everything before 'layers.0.*')
    # The final LayerNorm sits at '{prefix}norm.weight'
    norm_key = f"{prefix}norm.weight"
    if norm_key not in sd:
        candidates = [k for k in sd if k.endswith("norm.weight") and "layers" not in k]
        if not candidates:
            raise RuntimeError(f"Cannot find final norm. Tried '{norm_key}'. All norm keys: {[k for k in sd if 'norm' in k][:8]}")
        norm_key = candidates[0]
        if verbose:
            print(f"  Final norm (fallback): {norm_key}")
    final_norm = sd[norm_key].contiguous()

    # ── Diagnostic: print all embed-related keys so we know what's actually there
    if verbose:
        embed_keys = [(k, list(v.shape)) for k, v in sd.items() if "embed" in k.lower()]
        print(f"  Embed keys in state dict: {embed_keys}")

    # ── 3. Audio embed_tokens (talker's own embed, vocab=3072) ───────────────
    # Try exact match first; fall back to first 3072 rows of a larger embed under talker prefix
    audio_embed = None
    for k, v in sd.items():
        if "embed_tokens" in k and v.shape[0] == TTS_AUDIO_VOCAB:
            audio_embed = v.contiguous()
            if verbose:
                print(f"  Audio embed_tokens: {k}  {list(v.shape)}")
            break
    if audio_embed is None:
        # Talker may share one embed table for both text and audio tokens.
        # Slice first TTS_AUDIO_VOCAB rows as the audio embed.
        for k, v in sd.items():
            if "embed_tokens" in k and prefix in k and v.shape[0] >= TTS_AUDIO_VOCAB:
                audio_embed = v[:TTS_AUDIO_VOCAB].contiguous()
                if verbose:
                    print(f"  Audio embed_tokens (sliced [:3072] from {k}  {list(v.shape)})")
                break
    if audio_embed is None:
        raise RuntimeError(
            f"Could not find talker audio embed_tokens. "
            f"Embed keys found: {[(k, list(v.shape)) for k, v in sd.items() if 'embed' in k.lower()]}"
        )

    # ── 4. Text embed_tokens (151936×1024, for prefill via step()) ───────────
    text_embed = None
    for k, v in sd.items():
        if "embed_tokens" in k and v.shape[0] == KERNEL_VOCAB_SIZE:
            text_embed = v.contiguous()
            if verbose:
                print(f"  Text embed_tokens:  {k}  {list(v.shape)}")
            break
    if text_embed is None:
        # Use the talker's embed table for text prefill as well (same weights)
        for k, v in sd.items():
            if "embed_tokens" in k and prefix in k:
                text_embed = v.contiguous()
                if verbose:
                    print(f"  Text embed_tokens (using full {k}  {list(v.shape)} for prefill)")
                break
    if text_embed is None:
        if verbose:
            print("  Warning: could not find embed_tokens for text prefill; using audio embed")
        text_embed = audio_embed

    # ── 5. Audio LM head (codec_head) padded to kernel's VOCAB_SIZE ──────────
    audio_lm_head = None
    for k, v in sd.items():
        if "codec_head" in k and len(v.shape) == 2:
            audio_lm_head = v.contiguous()
            if verbose:
                print(f"  codec_head:         {k}  {list(v.shape)}")
            break
    if audio_lm_head is None:
        # Fallback: find any linear weight with correct output dim
        for k, v in sd.items():
            if len(v.shape) == 2 and v.shape[0] == TTS_AUDIO_VOCAB and v.shape[1] == HIDDEN_SIZE:
                audio_lm_head = v.contiguous()
                if verbose:
                    print(f"  lm_head (fallback): {k}  {list(v.shape)}")
                break
    if audio_lm_head is None:
        raise RuntimeError("Could not find codec_head weight [3072, 1024]")

    # Pad to [KERNEL_VOCAB_SIZE, HIDDEN_SIZE] — zeros for rows 3072..151935
    # Argmax picks correct audio code as long as ≥1 audio logit > 0
    padded_lm_head = torch.zeros(
        KERNEL_VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    padded_lm_head[:TTS_AUDIO_VOCAB] = audio_lm_head
    padded_lm_head = padded_lm_head.contiguous()

    # ── 6. Speech tokenizer for audio decoding ───────────────────────────────
    speech_tokenizer = None
    if hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
        speech_tokenizer = model.model.speech_tokenizer
    elif hasattr(model, "speech_tokenizer"):
        speech_tokenizer = model.speech_tokenizer

    # ── 7. EOS token ID for audio generation ─────────────────────────────────
    codec_eos_id = getattr(model.config, "codec_eos_token_id",
                   getattr(getattr(model.config, "talker_config", model.config), "codec_eos_token_id", 2048))

    del model
    torch.cuda.empty_cache()

    weights = {
        "audio_embed": audio_embed,
        "text_embed": text_embed,
        "layer_weights": layer_weights,
        "final_norm_weight": final_norm,
        "padded_lm_head": padded_lm_head,
        "codec_eos_id": codec_eos_id,
        "speech_tokenizer": speech_tokenizer,
    }
    if verbose:
        print(f"  codec_eos_id = {codec_eos_id}")
        print("  All weights loaded and remapped.")
    return weights


class TalkerDecoder:
    """
    Megakernel-backed Qwen3-TTS talker decoder.

    Backbone: 28 transformer layers run via megakernel CUDA op (unchanged).
    Embedding: swappable — text embed for prefill, audio embed for decode.
    LM head:   padded codec_head [3072→151936], argmax returns audio code ID.
    RoPE:      recomputed with theta=1_000_000.
    KV cache:  4096 positions.
    """

    def __init__(self, weights: dict | None = None, model_id: str = MODEL_ID, verbose: bool = True):
        try:
            import qwen_megakernel  # triggers JIT build of CUDA extension
            self._decode      = torch.ops.qwen_megakernel_C.decode
            self._gen_nosync  = torch.ops.qwen_megakernel_C.generate_nosync
        except ImportError as e:
            raise RuntimeError(
                "qwen_megakernel not installed. "
                "Run: pip install git+https://github.com/AlpinDale/qwen_megakernel"
            ) from e

        if weights is None:
            weights = load_tts_weights(model_id, verbose=verbose)
        self._weights = weights

        self._audio_embed  = weights["audio_embed"]
        self._text_embed   = weights["text_embed"]
        self._lm_head      = weights["padded_lm_head"]
        self._final_norm   = weights["final_norm_weight"]
        self.codec_eos_id  = weights["codec_eos_id"]
        self.speech_tokenizer = weights["speech_tokenizer"]

        # Start with audio embed (will swap during prefill if text_embed differs)
        self._embed_weight = self._audio_embed

        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position    = 0

        # RoPE tables with TTS theta
        self._cos_table, self._sin_table = _compute_rope_tables(TTS_MAX_SEQ_LEN, TTS_ROPE_THETA)

        # KV cache (larger than stock megakernel's 2048)
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32  = dict(dtype=torch.float32, device="cuda")
        self._k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, TTS_MAX_SEQ_LEN, HEAD_DIM, **bf16)
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
        self._hidden    = torch.empty(HIDDEN_SIZE, **bf16)
        self._act       = torch.empty(HIDDEN_SIZE, **f32)
        self._res       = torch.empty(HIDDEN_SIZE, **f32)
        self._q         = torch.empty(Q_SIZE,      **f32)
        self._k         = torch.empty(KV_SIZE,     **f32)
        self._v         = torch.empty(KV_SIZE,     **f32)
        self._attn_out  = torch.empty(Q_SIZE,      **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096,         **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _step(self, token_id: int) -> int:
        """Single-token decode. Returns next token id."""
        self._decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm,
            self._lm_head,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            TTS_MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def prefill_text(self, text_token_ids: list[int]) -> int:
        """
        Run text tokens through the backbone using text embeddings.
        Returns the first audio code token ID predicted after prefill.
        """
        # Use text embeddings for prefill phase
        self._embed_weight = self._text_embed
        for tid in text_token_ids[:-1]:
            self._step(tid)
        first_code = self._step(text_token_ids[-1])
        # Switch to audio embeddings for generation phase
        self._embed_weight = self._audio_embed
        return first_code

    def generate_audio_codes(
        self,
        first_code: int,
        max_tokens: int = 1024,
    ) -> list[int]:
        """
        Autoregressively generate audio code IDs using the megakernel.
        The code_predictor (residual codebooks) is intentionally NOT called here
        — this generates first-codebook tokens only.
        """
        codes = []
        token = first_code
        for _ in range(max_tokens):
            if token == self.codec_eos_id:
                break
            codes.append(token)
            token = self._step(token)
        return codes

    def decode_to_audio(self, codec_ids: list[int]) -> np.ndarray:
        """Convert first-codebook codes to waveform via speech_tokenizer."""
        if self.speech_tokenizer is None:
            raise RuntimeError("speech_tokenizer not available")
        ids_t = torch.tensor(codec_ids, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            audio = self.speech_tokenizer.decode(ids_t)
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().float().cpu().numpy()
        return audio


def benchmark(model_id: str = MODEL_ID, text: str = None, runs: int = 3):
    if text is None:
        text = "The megakernel runs at one thousand tokens per second on an RTX 5090."

    print("\n" + "=" * 60)
    print("Megakernel TTS Benchmark")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    text_ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"Text tokens: {len(text_ids)}")

    decoder = TalkerDecoder(model_id=model_id, verbose=True)

    all_metrics = []
    for run_i in range(runs):
        decoder.reset()
        torch.cuda.synchronize()

        t_prefill_start = time.perf_counter()
        first_code = decoder.prefill_text(text_ids)
        torch.cuda.synchronize()
        t_prefill_end = time.perf_counter()

        t_decode_start = time.perf_counter()
        audio_codes = decoder.generate_audio_codes(first_code)
        torch.cuda.synchronize()
        t_decode_end = time.perf_counter()

        prefill_ms = (t_prefill_end - t_prefill_start) * 1000
        decode_s   = t_decode_end - t_decode_start
        n_audio    = len(audio_codes)
        audio_dur_s = n_audio / CODEC_HZ
        tok_per_s  = n_audio / decode_s if decode_s > 0 else 0
        rtf        = decode_s / audio_dur_s if audio_dur_s > 0 else float("inf")
        ttfc_ms    = prefill_ms + (1 / tok_per_s * 1000 if tok_per_s > 0 else 0)

        metrics = {
            "run": run_i + 1,
            "prefill_ms": round(prefill_ms, 1),
            "decode_s": round(decode_s, 3),
            "audio_codes": n_audio,
            "audio_duration_s": round(audio_dur_s, 2),
            "tokens_per_s": round(tok_per_s, 1),
            "rtf": round(rtf, 4),
            "ttfc_ms": round(ttfc_ms, 1),
        }
        all_metrics.append(metrics)
        print(f"\nRun {run_i+1}: {tok_per_s:.0f} tok/s  |  RTF={rtf:.4f}  |  TTFC={ttfc_ms:.0f}ms  |  {n_audio} audio codes")

    if runs > 1:
        print("\n--- Averages (excluding first run JIT warmup) ---")
        warm = all_metrics[1:]
        for k in warm[0]:
            if k == "run":
                continue
            avg = sum(m[k] for m in warm) / len(warm)
            print(f"  {k}: {avg:.3f}")

    return all_metrics, decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="The megakernel runs at one thousand tokens per second on an RTX 5090.")
    parser.add_argument("--output", default="megakernel_output.wav")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Text: {args.text!r}")
    all_metrics, decoder = benchmark(args.model, args.text, args.runs)

    # Generate audio for output
    decoder.reset()
    text_ids = tokenizer.encode(args.text, add_special_tokens=True)
    first_code = decoder.prefill_text(text_ids)
    audio_codes = decoder.generate_audio_codes(first_code)

    if decoder.speech_tokenizer is not None:
        audio = decoder.decode_to_audio(audio_codes)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(args.output, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        print(f"\nSaved to {args.output}")
    else:
        print(f"\nSpeech tokenizer unavailable; audio codes saved as text ({len(audio_codes)} codes).")
        with open(args.output.replace(".wav", "_codes.txt"), "w") as f:
            f.write(" ".join(map(str, audio_codes)))


if __name__ == "__main__":
    main()
