"""
bot.py — Pipecat voice agent using the Qwen3-TTS megakernel streaming server.

Pipeline:
  DailyTransport (mic in)
  → DeepgramSTTService (speech-to-text)
  → OpenAILLMService (language model)
  → QwenTTSService (our streaming TTS via megakernel server)
  → DailyTransport (speaker out)

Prerequisites:
  pip install pipecat-ai[daily,deepgram,openai]
  uvicorn server.app:app --port 8080  (in another terminal)

Environment variables:
  DAILY_API_KEY, DAILY_ROOM_URL
  DEEPGRAM_API_KEY
  OPENAI_API_KEY
  TTS_SERVER_URL  (default: http://localhost:8080)

Usage:
  python pipecat_demo/bot.py
"""

import asyncio
import base64
import json
import os
import sys
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TTSService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://localhost:8080")
SAMPLE_RATE = 24_000


class QwenTTSService(TTSService):
    """
    Pipecat TTS service that streams audio from the Qwen megakernel server.

    Connects to the /tts SSE endpoint, decodes base64 PCM-16 chunks,
    and yields TTSAudioRawFrame objects to the pipeline as they arrive.
    This ensures Pipecat receives audio frame-by-frame, not buffered.
    """

    def __init__(self, server_url: str = TTS_SERVER_URL, **kwargs):
        super().__init__(**kwargs)
        self._server_url = server_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()
        logger.info(f"QwenTTSService connected to {self._server_url}")

    async def stop(self, frame):
        if self._session:
            await self._session.close()
        await super().stop(frame)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """
        Stream TTS audio from the megakernel server.

        Yields TTSStartedFrame, one TTSAudioRawFrame per PCM chunk, TTSStoppedFrame.
        """
        logger.debug(f"TTS request: {text!r}")

        if not self._session:
            self._session = aiohttp.ClientSession()

        yield TTSStartedFrame()

        ttfc_logged = False
        try:
            async with self._session.post(
                f"{self._server_url}/tts",
                json={"text": text},
                headers={"Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=60, connect=10),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"TTS server error {resp.status}: {body}")
                    yield TTSStoppedFrame()
                    return

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue

                    payload = json.loads(line[5:].strip())
                    event_type = payload.get("type")

                    if event_type == "chunk":
                        pcm_bytes = base64.b64decode(payload["pcm_b64"])
                        sr = payload.get("sample_rate", SAMPLE_RATE)

                        if not ttfc_logged:
                            ttfc_ms = payload.get("ttfc_ms", "?")
                            logger.info(f"TTFC: {ttfc_ms} ms")
                            ttfc_logged = True

                        yield TTSAudioRawFrame(
                            audio=pcm_bytes,
                            sample_rate=sr,
                            num_channels=1,
                        )

                    elif event_type == "done":
                        elapsed = payload.get("elapsed_ms", "?")
                        logger.info(f"TTS done in {elapsed} ms")
                        break

                    elif event_type == "error":
                        logger.error(f"TTS server error: {payload.get('message')}")
                        break

        except Exception as e:
            logger.exception(f"TTS stream error: {e}")

        yield TTSStoppedFrame()

    def language_to_service_language(self, language):
        return "English"


async def main():
    daily_api_key = os.environ.get("DAILY_API_KEY", "")
    daily_room_url = os.environ.get("DAILY_ROOM_URL", "")

    if not daily_room_url:
        logger.warning("DAILY_ROOM_URL not set; using a test room URL placeholder.")
        daily_room_url = "https://your-domain.daily.co/your-room"

    transport = DailyTransport(
        daily_room_url,
        None,
        "Qwen TTS Bot",
        DailyParams(
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            transcription_enabled=False,
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ.get("DEEPGRAM_API_KEY", ""))

    llm = OpenAILLMService(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    tts = QwenTTSService(server_url=TTS_SERVER_URL)

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=SAMPLE_RATE))

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. Keep responses concise "
                    "and conversational — ideally one or two sentences."
                ),
            }
        ]
    )
    context_aggregator = LLMContextAggregatorPair(context=context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(t, participant):
        await tts.say("Hello! I'm your voice assistant, powered by the Qwen megakernel. How can I help?")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(t, participant, reason):
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
