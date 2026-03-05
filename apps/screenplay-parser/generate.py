"""
TTS generation for screenplay content via higgs-tts gRPC API.

Generates audio for the first scene only. Converts scene content (heading,
action, dialogue) to plain text and streams it through the higgs-tts
InferenceEngine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__, "0.1.0")

# Defaults aligned with Higgs TTS recommended input format
DEFAULT_SPEAKER_DESCRIPTION = (
    "Male, American accent, modern speaking rate, moderate-pitch, "
    "friendly tone, and very clear audio."
)
DEFAULT_SCENE_DESCRIPTION = "Audio is recorded from a quiet room."


@dataclass
class GenerationOptions:
    """Optional overrides for Higgs TTS generation. All fields optional."""

    speaker_description: str | None = None
    scene_description: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    ras_win_len: int | None = None
    ras_win_max_num_repeat: int | None = None
    force_audio_gen: bool | None = None


def first_scene_to_text(content: dict) -> str:
    """Convert the first scene from stored screenplay content JSON to plain text.

    Renders heading, location/time, and a single content item (first dialogue
    block if any, else first action line) for minimal TTS generation.
    """
    scenes = content.get("scenes") or []
    if not scenes:
        return ""

    scene = scenes[0]
    parts: list[str] = []

    heading = (scene.get("heading") or "").strip()
    if heading:
        parts.append(heading)

    location = (scene.get("location") or "").strip()
    time_of_day = (scene.get("time_of_day") or "").strip()
    loc_tod = [s for s in (location, time_of_day) if s]
    if loc_tod:
        parts.append(" ".join(loc_tod))

    # Single item: prefer first dialogue block, else first action line
    dialogue_blocks = scene.get("dialogue_blocks") or []
    action_lines = scene.get("action_lines") or []
    if dialogue_blocks:
        block = dialogue_blocks[0]
        char = block.get("character") or "UNKNOWN"
        is_voice_over = block.get("is_voice_over") or False
        speech = (block.get("speech") or "").strip()
        prefix = f"{char} (V.O.): " if is_voice_over else f"{char}: "
        parts.append(prefix + speech)
        parentheticals = block.get("parentheticals") or []
        if parentheticals:
            parts.append(" (" + "; ".join(parentheticals) + ")")
    elif action_lines:
        t = (action_lines[0] or "").strip()
        if t:
            parts.append(t)

    return "\n\n".join(p for p in parts if p)


async def generate_audio_for_text(
    text: str,
    grpc_address: str | None = None,
    options: GenerationOptions | None = None,
) -> tuple[bytes, int, str]:
    """Call higgs-tts gRPC Generate and return (audio_bytes, sampling_rate, format).

    Uses Higgs TTS recommended defaults: descriptive speaker, temperature 1.0,
    max_tokens 2048, seed for reproducibility. Pass options to override.
    """
    address = grpc_address or os.getenv("HIGGS_TTS_GRPC_ADDRESS", "localhost:50051")
    opts = options or GenerationOptions()

    with tracer.start_as_current_span("higgs_tts_generate") as span:
        span.set_attribute("grpc_address", address)
        span.set_attribute("text_length", len(text))

        try:
            import grpc

            from inference_grpc import inference_engine_pb2 as pb2
            from inference_grpc import inference_engine_pb2_grpc as pb2_grpc
        except ImportError as e:
            logger.error("Import error: %s", e)
            raise RuntimeError(
                "Cannot import inference_grpc (grpcio + proto stubs). "
                "Ensure grpcio and inference_grpc are available."
            ) from e

        channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )

        stub = pb2_grpc.InferenceEngineStub(channel)
        logger.info("Calling higgs-tts at %s", address)

        speaker = pb2.Speaker(
            uuid="",
            description=opts.speaker_description or DEFAULT_SPEAKER_DESCRIPTION,
        )
        scene_desc = opts.scene_description or DEFAULT_SCENE_DESCRIPTION

        sampling_params = pb2.SamplingParams(
            temperature=opts.temperature if opts.temperature is not None else 1.0,
            top_p=opts.top_p if opts.top_p is not None else 0.95,
            top_k=opts.top_k if opts.top_k is not None else 50,
            ras_win_len=opts.ras_win_len if opts.ras_win_len is not None else 7,
            ras_win_max_num_repeat=(
                opts.ras_win_max_num_repeat if opts.ras_win_max_num_repeat is not None else 2
            ),
            force_audio_gen=(
                opts.force_audio_gen if opts.force_audio_gen is not None else False
            ),
        )
        if opts.max_tokens is not None:
            sampling_params.max_tokens = opts.max_tokens
        else:
            sampling_params.max_tokens = 2048
        if opts.seed is not None:
            sampling_params.seed = opts.seed
        else:
            sampling_params.seed = 123

        request = pb2.GenerateRequest(
            request_id=str(uuid.uuid4()),
            inputs=[
                pb2.GenerationInput(
                    messages=[
                        pb2.InputMessage(text=text, speaker=speaker),
                    ],
                    scene_description=scene_desc,
                ),
            ],
            sampling_params=sampling_params,
            stream=False,
        )

        audio_data = b""
        sampling_rate = 0
        audio_format = "flac"

        # Timeout in seconds (default 1h). Set HIGGS_TTS_GRPC_TIMEOUT to override, or 0 for no limit.
        grpc_timeout_s = int(os.getenv("HIGGS_TTS_GRPC_TIMEOUT", "3600"))

        def _do_grpc_call() -> tuple[bytes, int, str]:
            nonlocal audio_data, sampling_rate, audio_format
            try:
                kwargs = {} if grpc_timeout_s <= 0 else {"timeout": grpc_timeout_s}
                for response in stub.Generate(request, **kwargs):
                    resp_type = response.WhichOneof("response")
                    if resp_type == "complete":
                        complete = response.complete
                        if complete.audio_data:
                            audio_data = bytes(complete.audio_data)
                            sampling_rate = complete.sampling_rate or 0
                            audio_format = complete.audio_format or audio_format
            finally:
                channel.close()
            return (audio_data, sampling_rate, audio_format)

        # Run blocking gRPC in thread pool to avoid blocking the async event loop
        audio_data, sampling_rate, audio_format = await asyncio.to_thread(
            _do_grpc_call
        )

        span.set_attribute("audio_bytes", len(audio_data))
        span.set_attribute("sampling_rate", sampling_rate)

        return (audio_data, sampling_rate, audio_format)
