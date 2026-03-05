import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from opentelemetry.trace import Status, StatusCode

from .response import Response
from src.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ModelOutputProcessor:
    """Encapsulates conversion of model outputs into `AudioResponse` objects."""

    def __init__(self, tokenizer, audio_tokenizer, audio_codebook_size: int):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_codebook_size = audio_codebook_size

    def process(
        self,
        outputs: Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
        prompt_token_count: int,
    ) -> Response:
        with tracer.start_as_current_span("output_processor.process") as span:
            span.set_attribute("cognivault.prompt_tokens", prompt_token_count)
            try:
                text_outputs, audio_outputs = outputs

                generated_text_tokens = text_outputs[0].cpu().numpy()[prompt_token_count:]
                generated_text = self.tokenizer.decode(generated_text_tokens)
                generated_audio_tokens = (
                    audio_outputs[0].cpu().numpy() if len(audio_outputs) > 0 else None
                )

                span.set_attribute(
                    "cognivault.generated_text_tokens",
                    int(generated_text_tokens.shape[0]),
                )
                if generated_audio_tokens is not None:
                    span.set_attribute(
                        "cognivault.generated_audio_token_count",
                        int(generated_audio_tokens.shape[1]),
                    )
                span.set_attribute("cognivault.audio_chunk_count", len(audio_outputs))

                audio_waveform = self._decode_audio_outputs(audio_outputs)
                response = Response(
                    audio=audio_waveform,
                    generated_audio_tokens=generated_audio_tokens,
                    sampling_rate=self.audio_tokenizer.sampling_rate,
                    generated_text=generated_text,
                    generated_text_tokens=generated_text_tokens,
                    usage=self._build_usage(
                        prompt_token_count, generated_text_tokens, generated_audio_tokens
                    ),
                )

                logger.info(
                    "Output processed",
                    extra={
                        "prompt_token_count": prompt_token_count,
                        "text_token_count": int(generated_text_tokens.shape[0]),
                        "audio_chunk_count": len(audio_outputs),
                    },
                )
                return response
            except Exception as exc:
                logger.exception("Failed to process model outputs")
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def _decode_audio_outputs(
        self, audio_outputs: Sequence[torch.Tensor]
    ) -> Optional[np.ndarray]:
        if len(audio_outputs) == 0:
            return None

        with tracer.start_as_current_span("output_processor.decode_audio") as span:
            wav_chunks = []
            for idx, output_audio in enumerate(audio_outputs):
                vq_code = (
                    self._revert_delay_pattern(output_audio)
                    .clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                )
                wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                wav_chunks.append(wv_numpy)
                span.add_event(
                    "decoded_audio_chunk",
                    {
                        "chunk_index": idx,
                        "chunk_sample_count": int(wv_numpy.shape[-1]),
                    },
                )
            result = np.concatenate(wav_chunks)
            span.set_attribute("cognivault.audio_waveform_samples", result.shape[-1])
            return result

    @staticmethod
    def _revert_delay_pattern(data: torch.Tensor) -> torch.Tensor:
        """Convert samples encoded with delay pattern back to the original form."""
        assert len(data.shape) == 2
        out_l = []
        num_codebooks = data.shape[0]
        for i in range(num_codebooks):
            out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
        return torch.cat(out_l, dim=0)

    @staticmethod
    def _build_usage(
        prompt_token_count: int,
        generated_text_tokens: np.ndarray,
        generated_audio_tokens: Optional[np.ndarray],
    ) -> dict:
        audio_token_count = (
            generated_audio_tokens.shape[1] if generated_audio_tokens is not None else 0
        )
        completion_tokens = generated_text_tokens.shape[0] + audio_token_count
        return {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_token_count + completion_tokens,
            "cached_tokens": 0,
        }

