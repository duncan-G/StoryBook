"""
Async gRPC servicer for the InferenceEngine service.

Delegates generation to a synchronous :class:`AudioEngine`, running the
blocking call in a thread pool so the gRPC async event loop stays
responsive.

The servicer receives high-level :class:`GenerationInput` messages from
clients and uses :class:`InputProcessor` to normalize prompts, build
system messages, and assemble Chat objects before passing them to the
engine.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import grpc
import numpy as np
import soundfile as sf

from inference_grpc import inference_engine_pb2 as pb2
from inference_grpc import inference_engine_pb2_grpc

from src.data_models.generation_input import GenerationInput
from src.data_models.message import Message
from src.data_models.message_content import TextContent
from src.data_models.response import Response
from src.data_models.speaker import Speaker
from src.generation.engine import AudioEngine

logger = logging.getLogger(__name__)


class InferenceEngineServicer(inference_engine_pb2_grpc.InferenceEngineServicer):
    """Async gRPC servicer implementing the ``InferenceEngine`` service.

    RPCs implemented:
        - **Generate** — audio generation via :class:`AudioEngine`
    """

    def __init__(self, engine: AudioEngine) -> None:
        self.engine = engine
        logger.info("InferenceEngineServicer (async) initialized")

    # ------------------------------------------------------------------
    # Generate  (server-streaming RPC)
    # ------------------------------------------------------------------

    async def Generate(
        self,
        request: pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[pb2.GenerateResponse, None]:
        """Handle generation requests.

        Converts proto ``GenerationInput`` messages into internal
        :class:`GenerationInput` objects, then runs ``engine.generate``
        in a thread pool and yields one ``GenerateComplete`` per input.
        """
        request_id = request.request_id
        logger.debug("Generate request %s received.", request_id)

        try:
            params = _sampling_params_from_proto(request.sampling_params)
            gen_inputs = [_proto_input_to_internal(inp) for inp in request.inputs]

            for gen_input in gen_inputs:
                response: Response = await asyncio.to_thread(
                    self.engine.generate,
                    chat=gen_input,
                    **params,
                )
                yield _build_response(response)

        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Error in Generate for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


# ======================================================================
# Proto → internal conversion helpers
# ======================================================================

def _proto_input_to_internal(proto_input: Any) -> GenerationInput:
    """Convert a proto ``GenerationInput`` to an internal
    :class:`GenerationInput`."""
    messages = []
    for m in proto_input.messages:
        speaker = None
        if m.HasField("speaker") and m.speaker.description:
            speaker = Speaker(
                description=m.speaker.description,
                audio_url=m.speaker.audio_url or None,
                uuid=uuid.UUID(m.speaker.uuid) if m.speaker.uuid else uuid.uuid4(),
            )
        messages.append(
            Message(
                role="user",
                content=TextContent(text=m.text),
                speaker=speaker,
            )
        )
    return GenerationInput(
        messages=messages,
        system_prompt=(
            proto_input.system_prompt
            if proto_input.HasField("system_prompt")
            else None
        ),
        scene_description=(
            proto_input.scene_description
            if proto_input.HasField("scene_description")
            else None
        ),
    )


def _sampling_params_from_proto(params: pb2.SamplingParams) -> dict[str, Any]:
    """Convert a protobuf ``SamplingParams`` into kwargs for
    :meth:`AudioEngine.generate`."""
    return {
        "max_new_tokens": (
            params.max_tokens if params.HasField("max_tokens") else 2048
        ),
        "temperature": (
            params.temperature if params.HasField("temperature") else 0.7
        ),
        "top_k": params.top_k or None,
        "top_p": params.top_p if params.top_p != 0.0 else 0.95,
        "force_audio_gen": params.force_audio_gen,
        "ras_win_len": (
            params.ras_win_len if params.HasField("ras_win_len") else 7
        ),
        "ras_win_max_num_repeat": params.ras_win_max_num_repeat or 2,
        "seed": params.seed if params.HasField("seed") else None,
    }


# ======================================================================
# Internal → proto response helpers
# ======================================================================


AUDIO_FORMAT = "flac"


def _audio_to_bytes(audio: np.ndarray, sampling_rate: int) -> bytes:
    """Encode audio as FLAC bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sampling_rate, format="FLAC")
    buf.seek(0)
    return buf.read()


def _build_response(response: Response) -> pb2.GenerateResponse:
    """Build a ``GenerateComplete`` from an engine :class:`Response`."""
    usage = response.usage or {}

    audio_bytes = b""
    if response.audio is not None and response.sampling_rate:
        audio_bytes = _audio_to_bytes(response.audio, response.sampling_rate)

    output_ids: list[int] = []
    if response.generated_text_tokens is not None:
        output_ids = response.generated_text_tokens.tolist()

    return pb2.GenerateResponse(
        complete=pb2.GenerateComplete(
            output_ids=output_ids,
            finish_reason="stop",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            audio_data=audio_bytes,
            sampling_rate=response.sampling_rate or 0,
            audio_format=AUDIO_FORMAT,
        ),
    )
