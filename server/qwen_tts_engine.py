"""
qwen_tts_engine.py

QwenTTSEngine: text → streaming PCM audio.

Two backends (controlled by use_megakernel flag):
  • megakernel  — TalkerDecoder from generate_megakernel_tts.py
                  backbone runs via fused CUDA kernel; ~1000 tok/s on RTX 5090
  • hf_baseline — HuggingFace model.generate(); correct but slower

The engine:
  1. Tokenizes input text
  2. Runs prefill (text tokens → KV cache)
  3. Autoregressively generates audio code IDs
  4. Every `chunk_codes` codes, decodes to PCM via speech_tokenizer and yields bytes

Audio format: mono PCM-16, sample_rate=24000 Hz (Qwen3-TTS output).
"""

import asyncio
import io
import os
import struct
import sys
import time
import wave
from typing import AsyncGenerator

import numpy as np
import torch

# Ensure project root on path for 'scripts.*' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID    = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SAMPLE_RATE = 24_000
CODEC_HZ    = 12      # codec frames per second
NUM_CHANNELS = 1


def _pcm16_bytes(audio_np: np.ndarray) -> bytes:
    """Convert float32 [-1,1] numpy array to PCM-16 bytes."""
    pcm = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


def _wrap_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM bytes in a WAV container (for one-shot output)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class QwenTTSEngine:
    """
    Streaming TTS engine backed by the megakernel-adapted talker decoder.

    Usage:
        engine = QwenTTSEngine()
        async for pcm_chunk in engine.stream("Hello world"):
            # pcm_chunk: bytes of 16-bit mono PCM at 24kHz
            send_to_pipecat(pcm_chunk)
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        use_megakernel: bool = True,
        chunk_codes: int = 12,   # decode + yield every N audio codes (1 sec / CODEC_HZ)
        max_audio_tokens: int = 1024,
        verbose: bool = True,
    ):
        self.model_id = model_id
        self.use_megakernel = use_megakernel
        self.chunk_codes = chunk_codes
        self.max_audio_tokens = max_audio_tokens
        self.verbose = verbose

        self._decoder = None         # TalkerDecoder (megakernel) or None (HF path)
        self._hf_model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Load model weights. Call once before streaming."""
        if self._loaded:
            return

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.use_megakernel:
            try:
                from scripts.generate_megakernel_tts import TalkerDecoder
                self._decoder = TalkerDecoder(model_id=self.model_id, verbose=self.verbose)
                if self.verbose:
                    print("Megakernel backend loaded.")
            except Exception as e:
                print(f"Megakernel load failed ({e}); falling back to HF baseline.")
                self.use_megakernel = False

        if not self.use_megakernel:
            from transformers import AutoModelForCausalLM
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self._hf_model.eval()
            if self.verbose:
                print("HF baseline backend loaded.")

        self._loaded = True

    def _decode_codes_to_audio(self, codes: list[int]) -> np.ndarray | None:
        """Convert a list of first-codebook audio code IDs to waveform."""
        if self._decoder and self._decoder.speech_tokenizer:
            ids_t = torch.tensor(codes, dtype=torch.long, device="cuda").unsqueeze(0)
            with torch.no_grad():
                audio = self._decoder.speech_tokenizer.decode(ids_t)
        elif self._hf_model:
            st = None
            if hasattr(self._hf_model, "model") and hasattr(self._hf_model.model, "speech_tokenizer"):
                st = self._hf_model.model.speech_tokenizer
            elif hasattr(self._hf_model, "speech_tokenizer"):
                st = self._hf_model.speech_tokenizer
            if st is None:
                return None
            ids_t = torch.tensor(codes, dtype=torch.long, device="cuda").unsqueeze(0)
            with torch.no_grad():
                audio = st.decode(ids_t)
        else:
            return None

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().float().cpu().numpy()
        return audio

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Async generator. Yields PCM-16 mono bytes as audio codes are produced.
        First yield happens after `chunk_codes` audio codes (≈1 sec at 12 Hz).
        """
        if not self._loaded:
            self.load()

        if self.use_megakernel:
            async for chunk in self._stream_megakernel(text):
                yield chunk
        else:
            async for chunk in self._stream_hf(text):
                yield chunk

    async def _stream_megakernel(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Megakernel path: step through audio codes in a background thread,
        yielding PCM chunks as they arrive via an asyncio.Queue.
        """
        import threading

        loop = asyncio.get_running_loop()
        decoder = self._decoder
        tokenizer = self._tokenizer
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def thread_fn():
            try:
                text_ids = tokenizer.encode(text, add_special_tokens=True)
                decoder.reset()
                torch.cuda.synchronize()

                first_code = decoder.prefill_text(text_ids)
                torch.cuda.synchronize()

                buffer = []
                token = first_code
                for _ in range(self.max_audio_tokens):
                    if token == decoder.codec_eos_id:
                        break
                    buffer.append(token)
                    token = decoder._step(token)

                    if len(buffer) >= self.chunk_codes:
                        audio = self._decode_codes_to_audio(buffer)
                        buffer = []
                        if audio is not None:
                            loop.call_soon_threadsafe(queue.put_nowait, _pcm16_bytes(audio))

                if buffer:
                    audio = self._decode_codes_to_audio(buffer)
                    if audio is not None:
                        loop.call_soon_threadsafe(queue.put_nowait, _pcm16_bytes(audio))
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Start background thread — don't await it (that would block until done)
        t = threading.Thread(target=thread_fn, daemon=True)
        t.start()

        # Drain the queue concurrently while the thread generates
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def _stream_hf(self, text: str) -> AsyncGenerator[bytes, None]:
        """HuggingFace path (non-streaming fallback): generate all, then yield in chunks."""
        loop = asyncio.get_event_loop()
        model = self._hf_model
        tokenizer = self._tokenizer
        chunk_codes = self.chunk_codes

        def _run():
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_audio_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                )
            # Strip input tokens
            n_in = inputs["input_ids"].shape[-1]
            codec_ids = output[0, n_in:].cpu().tolist()
            return codec_ids

        codec_ids = await loop.run_in_executor(None, _run)

        # Yield chunks
        for i in range(0, len(codec_ids), chunk_codes):
            chunk = codec_ids[i : i + chunk_codes]
            audio = self._decode_codes_to_audio(chunk)
            if audio is not None:
                yield _pcm16_bytes(audio)

    def generate_wav(self, text: str) -> bytes:
        """Non-streaming: return complete WAV file bytes."""
        if not self._loaded:
            self.load()

        all_pcm = b""

        async def _collect():
            nonlocal all_pcm
            async for chunk in self.stream(text):
                all_pcm += chunk

        asyncio.run(_collect())
        return _wrap_wav(all_pcm)
