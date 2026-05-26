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
  ─ Arbitrary-embed trick: pass token_id=0 with a 1-row temp table to feed
      a precomputed 1024-dim vector to the kernel without kernel modifications

Per-step computation (fully matches qwen-tts generate() flow):
  1. codec_embed = audio_embed[first_code]           [1024]
  2. residual_codes = code_predictor.generate(codec_embed)  → 15 residual codes
  3. step_embed = codec_embed + Σ predictor_embeds[i](residual_codes[i])
  4. step_embed += trailing_text_hidden[decode_step]   # text conditioning
  5. megakernel(_step_with_embed(step_embed)) → next first_code

NOT modified:
  ─ kernel.cu / csrc — not a single byte changed
  ─ code_predictor weights — called as-is from qwen-tts

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
KERNEL_VOCAB_SIZE = 3072     # kernel LDG_VOCAB_SIZE patched to 3072 (TTS audio vocab);
                             # argmax now scans 6.3 MB instead of 311 MB — saves ~180 µs/step
TTS_AUDIO_VOCAB   = 3072     # Qwen3-TTS talker audio codebook size (== KERNEL_VOCAB_SIZE)
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

    Returns a weights dict with:
      - Backbone tensors (layer_weights, final_norm, padded_lm_head, audio_embed)
        copied to contiguous CUDA tensors for the kernel.
      - talker_module: the nn.Module for text_projection + code_predictor access.
        Thinker and other heavy modules are released to free VRAM.
      - All special token IDs needed to reproduce the exact prefill embed sequence.
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
    norm_key = f"{prefix}norm.weight"
    if norm_key not in sd:
        candidates = [k for k in sd if k.endswith("norm.weight") and "layers" not in k
                      and k.startswith(prefix)]
        if not candidates:
            raise RuntimeError(f"Cannot find final norm. Tried '{norm_key}'.")
        norm_key = candidates[0]
        if verbose:
            print(f"  Final norm (fallback): {norm_key}")
    final_norm = sd[norm_key].contiguous()

    # ── 3. Audio embed: talker.model.codec_embedding.weight [3072, 1024] ─────
    audio_embed = None
    for k, v in sd.items():
        if "codec_embedding" in k and prefix in k and v.shape == (TTS_AUDIO_VOCAB, HIDDEN_SIZE):
            audio_embed = v.contiguous()
            if verbose:
                print(f"  Audio embed: {k}  {list(v.shape)}")
            break
    if audio_embed is None:
        raise RuntimeError(
            f"Could not find audio embed [3072, 1024]. "
            f"Talker keys: {[(k, list(v.shape)) for k,v in sd.items() if k.startswith(prefix) and 'layers.' not in k]}"
        )

    # ── 4. Audio LM head (codec_head) padded to kernel's VOCAB_SIZE ──────────
    audio_lm_head = None
    for k, v in sd.items():
        if "codec_head" in k and len(v.shape) == 2:
            audio_lm_head = v.contiguous()
            if verbose:
                print(f"  codec_head: {k}  {list(v.shape)}")
            break
    if audio_lm_head is None:
        raise RuntimeError("Could not find codec_head weight [3072, 1024]")

    # KERNEL_VOCAB_SIZE == TTS_AUDIO_VOCAB == 3072 — no zero-padding needed;
    # kernel LDG_VOCAB_SIZE patched to 3072 so argmax only reads these rows.
    padded_lm_head = audio_lm_head[:KERNEL_VOCAB_SIZE].contiguous()

    # ── 5. Config values ──────────────────────────────────────────────────────
    cfg = getattr(inner, "config", None) or getattr(model, "config", None)
    talker_cfg = getattr(cfg, "talker_config", cfg) if cfg else None

    def _tc(attr, default):
        return getattr(talker_cfg, attr, default) if talker_cfg else default

    def _cc(attr, default):
        return getattr(cfg, attr, default) if cfg else default

    codec_eos_id       = _tc("codec_eos_token_id", 2150)
    codec_bos_id       = _tc("codec_bos_id",       2149)
    codec_pad_id       = _tc("codec_pad_id",       2148)
    codec_nothink_id   = _tc("codec_nothink_id",   2155)
    codec_think_bos_id = _tc("codec_think_bos_id", 2156)
    codec_think_eos_id = _tc("codec_think_eos_id", 2157)
    num_code_groups    = _tc("num_code_groups",       16)
    tts_bos_token_id   = _cc("tts_bos_token_id",  151672)
    tts_eos_token_id   = _cc("tts_eos_token_id",  151673)
    tts_pad_token_id   = _cc("tts_pad_token_id",  151671)

    if verbose:
        print(f"  codec_eos_id={codec_eos_id}  codec_bos_id={codec_bos_id}  num_code_groups={num_code_groups}")

    # ── 6. Keep talker module alive; release rest to free VRAM ───────────────
    # talker_module carries: text_projection, code_predictor, model.text_embedding
    # speech_tokenizer lives on inner, not on talker directly
    talker_module    = getattr(inner, "talker", None)
    speech_tokenizer = getattr(inner, "speech_tokenizer", None) or getattr(model, "speech_tokenizer", None)

    del model
    del inner
    torch.cuda.empty_cache()

    if verbose:
        print("  All weights loaded and remapped.")

    return {
        "audio_embed":         audio_embed,
        "layer_weights":       layer_weights,
        "final_norm_weight":   final_norm,
        "padded_lm_head":      padded_lm_head,
        "codec_eos_id":        codec_eos_id,
        "codec_bos_id":        codec_bos_id,
        "codec_pad_id":        codec_pad_id,
        "codec_nothink_id":    codec_nothink_id,
        "codec_think_bos_id":  codec_think_bos_id,
        "codec_think_eos_id":  codec_think_eos_id,
        "num_code_groups":     num_code_groups,
        "tts_bos_token_id":    tts_bos_token_id,
        "tts_eos_token_id":    tts_eos_token_id,
        "tts_pad_token_id":    tts_pad_token_id,
        "talker_module":       talker_module,
        "speech_tokenizer":    speech_tokenizer,
    }


class TalkerDecoder:
    """
    Megakernel-backed Qwen3-TTS talker decoder.

    Backbone: 28 transformer layers via megakernel CUDA op (unchanged).
    Per-step: codec_embed + code_predictor residuals + trailing_text_hidden
              → summed 1024-dim vector → kernel via 1-row temp embed table trick.
    LM head:  padded codec_head [3072→151936], argmax = audio code ID.
    RoPE:     recomputed with theta=1_000_000 (vs 10_000 in base model).
    KV cache: 4096 positions (vs 2048 in stock kernel).
    """

    def __init__(self, weights: dict | None = None, model_id: str = MODEL_ID, verbose: bool = True):
        try:
            import qwen_megakernel
            self._decode     = torch.ops.qwen_megakernel_C.decode
            self._gen_nosync = torch.ops.qwen_megakernel_C.generate_nosync
        except ImportError as e:
            raise RuntimeError(
                "qwen_megakernel not installed. Clone from github.com/AlpinDale/qwen_megakernel "
                "and add to PYTHONPATH."
            ) from e

        if weights is None:
            weights = load_tts_weights(model_id, verbose=verbose)

        # ── Megakernel tensors ────────────────────────────────────────────────
        self._audio_embed  = weights["audio_embed"]
        self._embed_weight = self._audio_embed   # always audio during decode
        self._lm_head      = weights["padded_lm_head"]
        self._final_norm   = weights["final_norm_weight"]

        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position   = 0

        self._cos_table, self._sin_table = _compute_rope_tables(TTS_MAX_SEQ_LEN, TTS_ROPE_THETA)

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32  = dict(dtype=torch.float32,  device="cuda")
        self._k_cache   = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, TTS_MAX_SEQ_LEN, HEAD_DIM, **bf16)
        self._v_cache   = torch.zeros_like(self._k_cache)
        self._hidden    = torch.empty(HIDDEN_SIZE,       **bf16)
        self._act       = torch.empty(HIDDEN_SIZE,       **f32)
        self._res       = torch.empty(HIDDEN_SIZE,       **f32)
        self._q         = torch.empty(Q_SIZE,            **f32)
        self._k         = torch.empty(KV_SIZE,           **f32)
        self._v         = torch.empty(KV_SIZE,           **f32)
        self._attn_out  = torch.empty(Q_SIZE,            **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE,       **f32)
        self._bmax_vals = torch.empty(4096,              **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1,    dtype=torch.int32, device="cuda")

        # ── qwen-tts components (kept alive from talker_module) ───────────────
        talker = weights.get("talker_module")
        self._text_projection = getattr(talker, "text_projection", None) if talker else None
        self._code_predictor  = getattr(talker, "code_predictor",  None) if talker else None
        # text_embedding lives at talker.model.text_embedding (2048-dim)
        self._text_embed_raw  = (
            talker.model.text_embedding.weight
            if talker and hasattr(talker, "model") and hasattr(talker.model, "text_embedding")
            else None
        )

        self.codec_eos_id          = weights["codec_eos_id"]
        self.codec_bos_id          = weights.get("codec_bos_id",        2149)
        self._codec_pad_id         = weights.get("codec_pad_id",        2148)
        self._codec_nothink_id     = weights.get("codec_nothink_id",    2155)
        self._codec_think_bos_id   = weights.get("codec_think_bos_id",  2156)
        self._codec_think_eos_id   = weights.get("codec_think_eos_id",  2157)
        self._tts_bos_id           = weights.get("tts_bos_token_id",  151672)
        self._tts_eos_id           = weights.get("tts_eos_token_id",  151673)
        self._tts_pad_id           = weights.get("tts_pad_token_id",  151671)
        self._num_residual         = weights.get("num_code_groups", 16) - 1   # 15
        self.speech_tokenizer      = weights["speech_tokenizer"]

        # Per-generation state — reset in reset()
        self._trailing_text_hidden: torch.Tensor | None = None
        self._past_hidden: torch.Tensor | None = None   # backbone norm_out from previous step
        self._decode_step = 0

    # ── Core kernel calls ─────────────────────────────────────────────────────

    def reset(self):
        self._position = 0
        self._decode_step = 0
        self._trailing_text_hidden = None
        self._past_hidden = None
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _step(self, token_id: int) -> int:
        """Single-token decode using embed table lookup (for direct benchmarking)."""
        self._decode(
            self._out_token, token_id, self._embed_weight,
            self._layer_weights_packed, self._final_norm, self._lm_head,
            self._cos_table, self._sin_table, self._k_cache, self._v_cache,
            self._hidden, self._act, self._res, self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, TTS_MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1
        self._past_hidden = self._norm_out.to(torch.bfloat16).unsqueeze(0).unsqueeze(0).clone()
        return self._out_token.item()

    def _step_with_embed(self, embed: torch.Tensor) -> int:
        """
        Single-token decode with a precomputed 1024-dim embedding vector.

        Trick: pass token_id=0 and a 1-row embed table containing our vector.
        The kernel does embed = table[token_id] = table[0] = our embed.
        No kernel modifications needed.
        """
        temp_table = embed.to(torch.bfloat16).unsqueeze(0).contiguous()
        self._decode(
            self._out_token, 0, temp_table,
            self._layer_weights_packed, self._final_norm, self._lm_head,
            self._cos_table, self._sin_table, self._k_cache, self._v_cache,
            self._hidden, self._act, self._res, self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, TTS_MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1
        self._past_hidden = self._norm_out.to(torch.bfloat16).unsqueeze(0).unsqueeze(0).clone()
        return self._out_token.item()

    # ── Text prefill ──────────────────────────────────────────────────────────

    def _proj(self, ids_t: torch.Tensor) -> torch.Tensor:
        """text_projection(text_embedding(ids_t)) → [N, 1024]."""
        raw = self._text_embed_raw[ids_t]            # [N, 2048]
        return self._text_projection(raw.unsqueeze(0)).squeeze(0)  # [N, 1024]

    def prefill_text(self, input_ids: list[int]) -> int:
        """
        Prefill the backbone with the exact embed sequence used by
        Qwen3TTSForConditionalGeneration.generate() (non-ICL streaming mode).

        input_ids must be the tokenized form of:
            "<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"
        encoded with add_special_tokens=False.

        Structure expected:
            input_ids[:3]   = role prefix  [<|im_start|>, assistant, \\n]
            input_ids[3:-5] = text body    (≥ 1 token)
            input_ids[-5:]  = trailing     [<|im_end|>, \\n, <|im_start|>, assistant, \\n]

        Prefill sequence fed to the kernel (8 tokens):
            [0-2]  text_proj(role_prefix)                    — 3 tokens
            [3]    tts_pad_e + codec_nothink_e               — summed embed
            [4]    tts_pad_e + codec_think_bos_e             — summed embed
            [5]    tts_pad_e + codec_think_eos_e             — summed embed
            [6]    tts_bos_e + codec_pad_e                   — summed embed
            [7]    text_proj(text[0]) + codec_bos_e          — summed embed

        trailing_text_hidden used per-step during decode:
            [i]    text_proj(text[i+1])  for i < len(text_body)-1
            [-1]   tts_eos_embed

        Returns the first predicted audio code ID.
        """
        dev = self._text_embed_raw.device
        ids_t = torch.tensor(input_ids, dtype=torch.long, device=dev)

        with torch.inference_mode():
            # ── Special TTS embeddings (projected) ───────────────────────────
            tts_ids = torch.tensor(
                [self._tts_bos_id, self._tts_eos_id, self._tts_pad_id],
                dtype=torch.long, device=dev
            )
            tts_e = self._proj(tts_ids)          # [3, 1024]
            tts_bos_e = tts_e[0]
            tts_eos_e = tts_e[1]
            tts_pad_e = tts_e[2]

            # ── Codec tag + pad/bos embeddings ───────────────────────────────
            codec_ids = torch.tensor(
                [self._codec_nothink_id, self._codec_think_bos_id,
                 self._codec_think_eos_id, self._codec_pad_id, self.codec_bos_id],
                dtype=torch.long, device=dev
            )
            codec_e = self._audio_embed[codec_ids]   # [5, 1024]

            # ── Role prefix (3 tokens) ────────────────────────────────────────
            role_e = self._proj(ids_t[:3])           # [3, 1024]

            # ── Combined codec prefix (4 summed tokens) ───────────────────────
            combined = torch.stack([
                tts_pad_e + codec_e[0],   # tts_pad + codec_nothink
                tts_pad_e + codec_e[1],   # tts_pad + codec_think_bos
                tts_pad_e + codec_e[2],   # tts_pad + codec_think_eos
                tts_bos_e + codec_e[3],   # tts_bos + codec_pad
            ])                            # [4, 1024]

            # ── First text token + codec_bos (1 summed token) ─────────────────
            first_text_e = self._proj(ids_t[3:4]).squeeze(0)   # [1024]
            first_e = first_text_e + codec_e[4]                # [1024]

            # ── trailing_text_hidden: text[1:-5] tokens + tts_eos ─────────────
            text_rest_ids = ids_t[4:-5]
            if text_rest_ids.numel() > 0:
                rest_e = self._proj(text_rest_ids)              # [M, 1024]
                trailing = torch.cat([rest_e, tts_eos_e.unsqueeze(0)], dim=0)
            else:
                trailing = tts_eos_e.unsqueeze(0)              # [1, 1024]

        self._trailing_text_hidden = trailing.detach()
        self._decode_step = 0

        # Feed 7 prefill tokens (ignore returned code IDs — only updating KV cache)
        for e in role_e:
            self._step_with_embed(e)
        for e in combined:
            self._step_with_embed(e)
        # Return the first audio code from the final prefill token
        return self._step_with_embed(first_e)

    # ── Audio code generation ─────────────────────────────────────────────────

    def _compute_step_embed(self, first_code: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the full 1024-dim input embedding for one decode step and
        collect all 16 codebook codes.

        Returns (step_embed [1024], all_codes [16]) where all_codes[0] = first_code
        and all_codes[1:] = residual codes from code_predictor.
        """
        dev = self._audio_embed.device
        last_id_hidden = self._audio_embed[first_code].unsqueeze(0).unsqueeze(0)  # [1,1,1024] bf16

        if self._code_predictor is not None and self._past_hidden is not None:
            # Real model passes cat([past_hidden, last_id_hidden]) → shape [1,2,1024]
            # This triggers the prefill branch in code_predictor.forward (shape[1] > 1)
            # and sets generation_steps = shape[1] - 2 = 0 for the first residual codebook.
            pred_input = torch.cat([self._past_hidden.to(dev), last_id_hidden], dim=1)  # [1,2,1024]
            with torch.inference_mode():
                pred = self._code_predictor.generate(
                    inputs_embeds=pred_input,
                    max_new_tokens=self._num_residual,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
                embed_fns = self._code_predictor.get_input_embeddings()
                residual_sum = torch.zeros(HIDDEN_SIZE, device=dev, dtype=torch.float32)
                for i in range(self._num_residual):
                    r_embed = embed_fns[i](
                        pred.sequences[..., i:i+1].to(dev)
                    ).float().squeeze()
                    residual_sum += r_embed

            residual_codes = pred.sequences[0].cpu()   # [15]
            all_codes = torch.cat([torch.tensor([first_code]), residual_codes])  # [16]
            step_embed = last_id_hidden.squeeze().float() + residual_sum
        else:
            all_codes  = torch.tensor([first_code])
            step_embed = last_id_hidden.squeeze().float()

        # Per-step text conditioning from trailing_text_hidden
        if self._trailing_text_hidden is not None:
            idx = min(self._decode_step, self._trailing_text_hidden.shape[0] - 1)
            step_embed = step_embed + self._trailing_text_hidden[idx].float()

        self._decode_step += 1
        return step_embed.to(torch.bfloat16), all_codes

    def generate_audio_codes_iter(self, first_code: int, max_tokens: int = 1024):
        """
        Generator: yields one all_codes tensor [16] per step until EOS or max_tokens.
        Also advances the megakernel backbone one step per yield.
        """
        token = first_code
        for _ in range(max_tokens):
            if token == self.codec_eos_id:
                return
            step_embed, all_codes = self._compute_step_embed(token)
            yield all_codes
            token = self._step_with_embed(step_embed)

    def generate_audio_codes(self, first_code: int, max_tokens: int = 1024) -> list:
        """Collect all codec code tensors (each [16]) into a list."""
        return list(self.generate_audio_codes_iter(first_code, max_tokens))

    def decode_to_audio(self, all_codec_codes: list) -> np.ndarray:
        """
        Decode a list of all_codes tensors (each [num_codebooks]) to waveform.

        Calls speech_tokenizer.decode({"audio_codes": [num_codebooks, num_frames]}).
        """
        if self.speech_tokenizer is None:
            raise RuntimeError("speech_tokenizer not available")
        if not all_codec_codes:
            return np.zeros(0, dtype=np.float32)

        # Stack: [num_steps, num_codebooks] → transpose → [num_codebooks, num_steps]
        codes_t = torch.stack(all_codec_codes, dim=0).T.long()
        if codes_t.device.type != "cuda":
            codes_t = codes_t.cuda()

        with torch.inference_mode():
            wavs, _ = self.speech_tokenizer.decode({"audio_codes": codes_t})
        return np.array(wavs[0], dtype=np.float32)


def _format_tts_text(text: str) -> str:
    """Wrap plain text in the chat template expected by the talker backbone."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def benchmark(model_id: str = MODEL_ID, text: str = None, runs: int = 3):
    if text is None:
        text = "The megakernel runs at one thousand tokens per second on an RTX 5090."

    print("\n" + "=" * 60)
    print("Megakernel TTS Benchmark")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    text_ids = tokenizer.encode(_format_tts_text(text), add_special_tokens=False)
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
    text_ids = tokenizer.encode(_format_tts_text(args.text), add_special_tokens=False)
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
