import json
import logging
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, WhisperProcessor
from transformers.cache_utils import StaticCache

from src.data_models.model_output_processor import ModelOutputProcessor
from src.data_models.generation_input import GenerationInput
from src.data_collator.higgs_audio_data_collator import HiggsAudioDataCollator
from src.input_processor import InputProcessor
from src.audio_tokenizer.higgs_audio_tokenizer import HiggsAudioTokenizer
from src.audio_model.model import HiggsAudioModel

logger = logging.getLogger(__name__)

class AudioEngine:
    """
    AudioEngine is a class that provides a high-level interface for generating audio from a chat history.
    It uses the HiggsAudioModel and AudioTokenizer to generate audio from a ChatML sample.

    Args:
        model_name_or_path: The modelId for a model hosted on Hugging Face Hub or a local path to the tokenizer weights.
        tokenizer_name_or_path: The modelId for a tokenizer hosted on Hugging Face Hub or a local path to the tokenizer weights.
        model_cache_dir: The directory to cache the model and tokenizer.
        torch_dtype: The torch dtype to use for the model.
        device: The device to use for the model.
        kv_cache_lengths: The lengths of the key-value caches to use.
    """
    def __init__(
        self,
        model_name_or_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        tokenizer_name_or_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_name_or_path: str = "bosonai/higgs-audio-v2-tokenizer",
        model_cache_dir: Optional[str] = "./.models",
        torch_dtype: Optional[Union[torch.dtype, str]] = None,
        device: Optional[str] = None,
        kv_cache_lengths: List[int] = [1024, 4096, 8192]
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.model_cache_dir = model_cache_dir
        self.torch_dtype = torch_dtype if torch_dtype is not None else torch.float16
        self.device = self._get_device(device)
        self.tokenizer = self._load_text_tokenizer()
        self.model = self._load_model()
        self.audio_tokenizer = self._load_audio_tokenizer()
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.input_processor = self._create_input_processor()
        self.data_collator = self._create_data_collator()
        self.output_processor = self._create_output_processor()

        # Store cache config for lazy creation
        self.cache_config = deepcopy(self.model.config.text_config)
        self.cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers is not None:
            self.cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        self.kv_cache_lengths = sorted(kv_cache_lengths)
        # Create KV caches lazily to save memory - only create when needed
        self.kv_caches = {}

        logger.info(
            "AudioEngine initialized",
            extra={
                "torch_dtype": str(self.torch_dtype),
                "device": str(self.device),
                "kv_cache_lengths": self.kv_cache_lengths,
            },
        )

    @torch.inference_mode()
    def generate(
        self,
        chat: GenerationInput, 
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate audio from a chatML sample.
        Args:
            chat: A chat history.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_k: The top k to use for the generation.
            top_p: The top p to use for the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
            seed: The seed to use for the generation.
        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        total_start = perf_counter()

        with torch.no_grad():
            try:
                input_start = perf_counter()
                input = self.input_processor.process_input(chat)
                batch = self.data_collator([input])
                input_duration_ms = (perf_counter() - input_start) * 1000
                prompt_token_count = int(batch.input_ids.shape[-1])

                self._prepare_kv_caches()

                generate_start = perf_counter()
                outputs = self.model.generate(
                    **asdict(batch),
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    tokenizer=self.tokenizer,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    past_key_values_buckets=self.kv_caches,
                    ras_win_len=ras_win_len,
                    ras_win_max_num_repeat=ras_win_max_num_repeat,
                    seed=seed,
                )
                generate_duration_ms = (perf_counter() - generate_start) * 1000

                output_start = perf_counter()
                response = self.output_processor.process(
                    outputs=outputs,
                    prompt_token_count=prompt_token_count,
                )
                output_duration_ms = (perf_counter() - output_start) * 1000

                # Calculate metrics
                total_duration_ms = (perf_counter() - total_start) * 1000
                text_token_count = (
                    int(response.generated_text_tokens.shape[0])
                    if response.generated_text_tokens is not None
                    else 0
                )
                audio_token_count = (
                    int(response.generated_audio_tokens.shape[1])
                    if response.generated_audio_tokens is not None
                    else 0
                )
                total_generated_tokens = text_token_count + audio_token_count
                
                # Calculate generation speed (tokens per second)
                generation_speed = (
                    total_generated_tokens / (generate_duration_ms / 1000)
                    if generate_duration_ms > 0
                    else 0
                )
                
                # Calculate audio duration if available
                audio_duration_sec = None
                if response.audio is not None and response.sampling_rate is not None:
                    audio_duration_sec = response.audio.shape[-1] / response.sampling_rate

                logger.info(
                    "Audio generation completed",
                    extra={
                        "prompt_tokens": prompt_token_count,
                        "generated_text_tokens": text_token_count,
                        "generated_audio_tokens": audio_token_count,
                        "total_generated_tokens": total_generated_tokens,
                        "has_audio": response.audio is not None,
                        "audio_duration_sec": round(audio_duration_sec, 3) if audio_duration_sec is not None else None,
                        "sampling_rate": response.sampling_rate,
                        "timing_ms": {
                            "input_prep": round(input_duration_ms, 2),
                            "model_generate": round(generate_duration_ms, 2),
                            "output_processing": round(output_duration_ms, 2),
                            "total": round(total_duration_ms, 2),
                        },
                        "generation_speed_tokens_per_sec": round(generation_speed, 2),
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
                return response
            except Exception as exc:
                total_duration_ms = (perf_counter() - total_start) * 1000
                logger.exception(
                    "Audio generation failed",
                    extra={
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "force_audio_gen": force_audio_gen,
                        "ras_win_len": ras_win_len,
                        "seed": seed,
                        "duration_ms": round(total_duration_ms, 2),
                    },
                )
                raise
            finally:
                self._cleanup_after_generation()
 
            
    def _load_model(self):
        return HiggsAudioModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            cache_dir=self.model_cache_dir
        ).to(self.device)

    def _prepare_kv_caches(self):
        """Prepare KV caches, creating them lazily if they don't exist."""
        for length in self.kv_cache_lengths:
            if length not in self.kv_caches:
                # Create cache lazily when first needed
                self.kv_caches[length] = StaticCache(
                    config=self.cache_config,
                    max_batch_size=1,
                    max_cache_len=length,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
            else:
                self.kv_caches[length].reset()
    
    def _cleanup_after_generation(self):
        """Clean up model state and CUDA graphs after generation to prevent memory leaks."""
        # Reset model's past_key_values_bucket tracking
        if hasattr(self.model, 'current_past_key_values_bucket'):
            self.model.current_past_key_values_bucket = None
        
        # Clear CUDA graph runners to free memory
        # These accumulate during generation and can consume significant memory
        if hasattr(self.model, 'decode_graph_runners'):
            self.model.decode_graph_runners.clear()
        
        # Optionally: Clear KV caches to free memory (uncomment if memory is tight)
        # Note: They will be recreated on next generation, which adds a small overhead
        # self.kv_caches.clear()
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_device(self, device: Optional[str] = None):
        """Get the device to use for the model.
        Determines whether to use CPU or GPU for the model.
        Note on MacOS, we can specify MPS for the device to use the Apple Silicon GPU,
        because CUDA is not available on MacOS.
        Currently not supporting this.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_text_tokenizer(self):
        """Create the tokenizer for the model.
        https://github.com/huggingface/transformers/blob/cac0a28c83cf87b7a05495de3177099c635ba852/src/transformers/models/auto/tokenization_auto.py#L902
        AutoTokenizer part of huggingface's Auto classes that will instantiate the appropriate tokenizer based on the model config.
        The tokenizer is what will convert text into token ids.
        tokenizer_name_or_path should contain the location of the tokenizer weights and other metadata.
        cache_dir is the directory to cache the tokenizer weights.
        """
        return AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, cache_dir=self.model_cache_dir)

    def _load_audio_tokenizer(self) -> HiggsAudioTokenizer:
        """
        Load the Higgs audio tokenizer weights from disk or HF Hub.
        https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/tree/main
        """
        model = HiggsAudioTokenizer.load(
            tokenizer_name_or_path=self.audio_tokenizer_name_or_path,
            device=self.device
        )

        logger.info("Audio tokenizer ready", extra={"tokenizer_path": str(self.audio_tokenizer_name_or_path)})
        return model


    def _create_input_processor(self):
        return InputProcessor(
            text_tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            device=self.device
        )

    def _create_data_collator(self)->HiggsAudioDataCollator:
        return HiggsAudioDataCollator(
            whisper_processor=WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", cache_dir=self.model_cache_dir),
            audio_in_token_id=100,
            audio_out_token_id=101,
            pad_token_id=100,
            audio_stream_bos_id=100,
            audio_stream_eos_id=101,
            round_to=8,
            pad_left=False,
            encode_whisper_embed=True,
            return_audio_in_tokens=True,
            audio_num_codebooks=None,
            use_delay_pattern=False,
        )

    def _create_output_processor(self) -> ModelOutputProcessor:
        return ModelOutputProcessor(
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            audio_codebook_size=self.audio_codebook_size,
        )