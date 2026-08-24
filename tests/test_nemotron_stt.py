"""LiveKit adapter for native streaming Nemotron."""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from livekit import rtc
from livekit.agents.stt import SpeechEventType

from local_voice_ai.nemotron_stt import NemotronSTT


def test_native_adapter_advertises_real_streaming() -> None:
    recognizer = NemotronSTT(base_url="http://127.0.0.1:8000/v1")

    assert recognizer.capabilities.streaming is True
    assert recognizer.capabilities.interim_results is True
    assert recognizer.realtime_url == "ws://127.0.0.1:8000/v1/realtime"
    assert recognizer.session_update()["session"] == {
        "sample_rate": 16_000,
        "language": "en",
        "automatic_punctuation": True,
        "endpointing_ms": 300,
    }


@pytest.mark.asyncio
async def test_microphone_frames_produce_interim_and_final_transcripts() -> None:
    received: dict[str, object] = {}

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        received["update"] = await socket.receive_json()
        audio = await socket.receive()
        received["audio_type"] = audio.type
        received["audio_bytes"] = len(audio.data)
        commit = await socket.receive()
        received["commit"] = json.loads(commit.data)
        await socket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "turn-1",
                "delta": "Hello",
            }
        )
        await socket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "turn-1",
                "transcript": "Hello Jetson.",
            }
        )
        await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    async with TestServer(app) as server:
        recognizer = NemotronSTT(base_url=str(server.make_url("/v1")))
        stream = recognizer.stream()
        frame = rtc.AudioFrame.create(sample_rate=16_000, num_channels=1, samples_per_channel=800)
        stream.push_frame(frame)
        stream.end_input()
        events = [event async for event in stream]
        await recognizer.aclose()

    assert received["update"] == recognizer.session_update()
    assert received["audio_type"].name == "BINARY"
    assert received["audio_bytes"] == 1600
    assert received["commit"] == {"type": "input_audio_buffer.commit"}
    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.RECOGNITION_USAGE,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "Hello"
    assert events[2].alternatives[0].text == "Hello Jetson."
