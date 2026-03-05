# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
import numpy as np
from transformers import AutoModel
import torchaudio
import json
import librosa
import pyloudnorm as pyln
from huggingface_hub import snapshot_download

from vector_quantize_pytorch import ResidualFSQ
from libs.rvq.vq import ResidualVectorQuantizer
from libs.xcodec import Encoder, Decoder
from libs.dac.model import dac as dac2
from .encoded_result import EncodedResult
from .higgs_audio_feature_extractor import HiggsAudioFeatureExtractor

class HiggsAudioTokenizer(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2],  #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        n_q: int = 8,
        codebook_dim: int = None,
        normalize: bool = False,
        causal: bool = False,
        semantic_teacher: str = "hubert_base_general",
        last_layer_semantic: bool = True,
        merge_mode: str = "concat",
        downsample_mode: str = "step_down",
        semantic_mode: str = "classic",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        # ========== Store basic configuration ==========
        self.sample_rate = sample_rate
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.semantic_teacher = semantic_teacher
        self.last_layer_semantic = last_layer_semantic
        self.downsample_mode = downsample_mode
        self.device = device
        
        # Calculate derived parameters
        self.hop_length = np.prod(ratios)
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 50 Hz

        # ========== Acoustic codec components ==========
        self.acoustic_encoder = dac2.Encoder(64, ratios, D)
        self.acoustic_decoder = dac2.Decoder(D, 1024, ratios)
        
        # ========== Semantic teacher model setup ==========
        # Get semantic model parameters
        semantic_model_name, self.semantic_sample_rate, self.semantic_dim, self.encoder_semantic_dim = (
            self._get_semantic_model_parameters(semantic_teacher)
        )
        
        # Overwrite semantic model sr if provided
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate
        
        # Load the semantic model (handle trust_remote_code for hubert_base_general)
        if semantic_teacher == "hubert_base_general":
            self.semantic_model = AutoModel.from_pretrained(semantic_model_name, trust_remote_code=True)
        else:
            self.semantic_model = AutoModel.from_pretrained(semantic_model_name)
        
        self.semantic_model.eval()
        # Freeze semantic model parameters
        for param in self.semantic_model.parameters():
            param.requires_grad = False
        
        # Calculate semantic downsampling factor
        self.semantic_downsample_factor = int(
            self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320
        )

        # ========== Semantic codec components ==========
        self.semantic_encoder = Encoder(
            input_channels=self.semantic_dim,
            encode_channels=self.encoder_semantic_dim,
        )
        self.semantic_decoder = Decoder(
            code_dim=self.encoder_semantic_dim,
            output_channels=self.semantic_dim,
            decode_channels=self.semantic_dim,
        )

        # ========== Quantizer setup ==========
        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        
        if isinstance(bins, int):  # RVQ
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        else:  # RFSQ
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        # ========== Fusion layers ==========
        self.fc_prior = nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)
        self.fc_post1 = nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)
        self.fc_post2 = nn.Linear(self.quantizer_dim, D)

        # ========== Downsampling and feature extraction ==========
        if downsample_mode == "avg":
            self.semantic_pooling = nn.AvgPool1d(
                kernel_size=self.semantic_downsample_factor, stride=self.semantic_downsample_factor
            )

        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(sampling_rate=self.sample_rate)
        
        # Move all model components to the specified device
        self.to(device)

    @property
    def tps(self):
        return self.frame_rate

    @property
    def sampling_rate(self):
        return self.sample_rate

    @property
    def num_codebooks(self):
        return self.n_q

    @property
    def codebook_size(self):
        return self.quantizer_dim

    def get_last_layer(self):
        return self.semantic_decoder.conv2.weight

    @staticmethod
    def load(
        tokenizer_name_or_path: str, 
        device: str = "cuda"
    ) -> "HiggsAudioTokenizer":
        """
        Loads a pre-trained HiggsAudioTokenizer from a local directory or HuggingFace Hub.

        This method handles legacy checkpoint compatibility by remapping old state_dict keys 
        (e.g., 'encoder' -> 'acoustic_encoder', decoder_2 -> acoustic_decoder, encoder_semantic -> semantic_encoder, decoder_semantic -> semantic_decoder)
        and sanitizing configuration typos (semantic_techer -> semantic_teacher).

        Args:
            tokenizer_name_or_path (str): Local path or HF Hub ID (e.g., "user/repo").
                                          Must contain 'config.json' and 'model.pth'.
            device (str): Device to load the model onto ("cuda", "cpu", "mps").

        Returns:
            HiggsAudioTokenizer: The initialized model in evaluation mode.

        Raises:
            FileNotFoundError: If config.json or model.pth are missing.
            RuntimeError: If critical model components (acoustic encoder/decoder) fail to load.
        """
        # ---------------------------------------------------------
        # 1. Resolve Path (Local vs. Hub)
        # ---------------------------------------------------------
        if os.path.exists(tokenizer_name_or_path):
            tokenizer_path = tokenizer_name_or_path
        else:
            # Not local? Assume it's a Hub ID and download a snapshot.
            try:
                tokenizer_path = snapshot_download(tokenizer_name_or_path)
            except Exception as e:
                raise ValueError(f"Could not find local path or download from Hub: {tokenizer_name_or_path}") from e

        config_path = os.path.join(tokenizer_path, "config.json")
        model_path = os.path.join(tokenizer_path, "model.pth")

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Directory must contain 'config.json' and 'model.pth'. Found: {os.listdir(tokenizer_path)}")

        # ---------------------------------------------------------
        # 2. Load and Sanitize Configuration
        # ---------------------------------------------------------
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Backward Compatibility: Fix known typos in older config files
        if "semantic_techer" in config:
            config["semantic_teacher"] = config.pop("semantic_techer")

        # Initialize the model structure
        model = HiggsAudioTokenizer(**config, device=device)

        # ---------------------------------------------------------
        # 3. Load and Remap Checkpoint Weights
        # ---------------------------------------------------------
        raw_state_dict = torch.load(model_path, map_location=device)
        clean_state_dict = {}

        # Mappings for legacy keys: { old_prefix: new_prefix }
        legacy_key_map = {
            "encoder.": "acoustic_encoder.",
            "decoder_2.": "acoustic_decoder.",
            "encoder_semantic.": "semantic_encoder.",
            "decoder_semantic.": "semantic_decoder."
        }

        for key, value in raw_state_dict.items():
            new_key = key
            # Check if key needs remapping
            for old_prefix, new_prefix in legacy_key_map.items():
                if key.startswith(old_prefix) and not key.startswith(new_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break 
            clean_state_dict[new_key] = value

        # ---------------------------------------------------------
        # 4. Load Weights & Validate Critical Components
        # ---------------------------------------------------------
        # strict=False allows us to load inference weights even if optimizer states are missing
        missing_keys, _ = model.load_state_dict(clean_state_dict, strict=False)

        # Define critical prefixes that MUST be present for the model to function
        critical_prefixes = ["acoustic_encoder.", "acoustic_decoder."]
        
        # Verify that all keys starting with critical prefixes were actually loaded
        loaded_keys = set(clean_state_dict.keys())
        model_keys = set(model.state_dict().keys())

        for prefix in critical_prefixes:
            # Find all expected keys for this prefix
            expected = {k for k in model_keys if k.startswith(prefix)}
            # Find which ones are missing from the loaded dict
            missing = expected - loaded_keys
            
            if missing:
                # Format the error to show the first few missing keys for debugging
                example_missing = list(missing)[:5]
                raise RuntimeError(
                    f"Critical failure: '{prefix}' weights incomplete. "
                    f"Missing {len(missing)} keys, including: {example_missing}..."
                )

        # ---------------------------------------------------------
        # 5. Finalize
        # ---------------------------------------------------------
        # Warn about non-critical missing keys (excluding optimizer/scheduler internals)
        if missing_keys:
            ignored_substrings = ['optimizer', 'scheduler', 'amp', 'global_step']
            real_missing = [k for k in missing_keys if not any(sub in k for sub in ignored_substrings)]
            if real_missing:
                import warnings
                warnings.warn(f"Non-critical weights were not loaded: {real_missing[:5]}... (Total: {len(real_missing)})")

        # Ensure model is on the correct device (in case state_dict loading affected device placement)
        model.to(device)
        model.eval()
        return model

    def forward(self, raw_waveform: torch.Tensor, bw: int):
        ssl_semantic_vectors = self._extract_semantic_features(raw_waveform).detach()

        semantic_latents = self.semantic_encoder(ssl_semantic_vectors.transpose(1, 2))
        acoustic_latents = self.acoustic_encoder(raw_waveform)

        combined_latents = torch.cat([acoustic_latents, semantic_latents], dim=1)

        fused_latents = self.fc_prior(combined_latents.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            fused_latents = fused_latents.transpose(1, 2)
            quantized_latents, code_indices, bandwidth, commit_loss = self.quantizer(fused_latents, self.frame_rate, bw)
            quantized_latents = quantized_latents.transpose(1, 2)
        else:
            quantized_latents, code_indices = self.quantizer(fused_latents)
            commit_loss = torch.tensor(0.0)

        quantized_semantic_latents = self.fc_post1(quantized_latents).transpose(1, 2)
        quantized_acoustic_latents = self.fc_post2(quantized_latents).transpose(1, 2)

        reconstructed_waveform = self.acoustic_decoder(quantized_acoustic_latents)

        decoded_semantic_vectors = self.semantic_decoder(quantized_semantic_latents)
        semantic_recon_loss = F.mse_loss(ssl_semantic_vectors.transpose(1, 2).detach(), decoded_semantic_vectors)

        return reconstructed_waveform, commit_loss, semantic_recon_loss, None

    def encode(
        self,
        audio_source: Union[str, np.ndarray],
        sampling_rate: Optional[int] = None,
        loudness_normalize: bool = False,
        loudness_threshold: float = -23.0,
    ) -> torch.Tensor:
        """
        Convert an audio source (path or waveform) into discrete tokenizer codes.

        Args:
            audio_source: File path or numpy waveform to tokenize.
            sampling_rate: Original sampling rate when passing a raw waveform; ignored for file paths.
            loudness_normalize: Whether to loudness-normalize before tokenization.
            loudness_threshold: Target LUFS level when `loudness_normalize` is True.
        """
        # Load waveform from disk or use the provided numpy array
        if isinstance(audio_source, str):
            waveform, audio_sampling_rate = librosa.load(audio_source, mono=True, sr=None)
        else:
            waveform = audio_source
            if sampling_rate is None:
                raise ValueError("Sampling rate must be provided when passing a raw waveform.")
            audio_sampling_rate = sampling_rate

        # Optionally normalize the integrated loudness
        if loudness_normalize:
            meter = pyln.Meter(audio_sampling_rate)
            loudness = meter.integrated_loudness(waveform)
            waveform = pyln.normalize.loudness(waveform, loudness, loudness_threshold)

        # Resample to the tokenizer's expected sampling rate
        if audio_sampling_rate != self.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=audio_sampling_rate, target_sr=self.sampling_rate)

        # Extract features from the audio waveform
        audio_features_result = self.audio_tokenizer_feature_extractor(
            raw_audio=waveform, sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate, return_tensors="pt"
        )
        audio_features_tensor = audio_features_result["input_values"].to(self.device)

        with torch.no_grad():
            # Quantized Result: The full output containing loss, latents, and codes
            quantized_output = self._xcodec_encode(audio_features_tensor)

            # Discrete Codes: The final list of integers (indices from the Codebook)
            discrete_codes = quantized_output.audio_codes[0]

        return discrete_codes

    def decode(self, code_indices: torch.Tensor) -> torch.Tensor:
        code_indices = code_indices.to(self.device)

        if self.quantizer_type == "RVQ":
            code_indices = code_indices.permute(1, 0, 2)
            quantized_latents = self.quantizer.decode(code_indices)
            quantized_latents = quantized_latents.transpose(1, 2)
        else:
            code_indices = code_indices.permute(0, 2, 1)
            quantized_latents = self.quantizer.get_output_from_indices(code_indices)
        quantized_acoustic_latents = self.fc_post2(quantized_latents).transpose(1, 2)

        reconstructed_waveform = self.acoustic_decoder(quantized_acoustic_latents)
        return reconstructed_waveform.detach().cpu().numpy()

    @torch.no_grad()
    def _extract_semantic_features(self, waveform_input: torch.Tensor) -> torch.Tensor:
        """
        Extracts semantic representations from the frozen Self-Supervised Learning (SSL) model.
        These features guide the X-Codec to ensure Semantic Integrity.
        """
        # Resample input to match the SSL model's expected sample rate
        resampled_input = torchaudio.functional.resample(
            waveform_input, self.sample_rate, self.semantic_sample_rate
        )

        semantic_vectors = None

        # Handling different SSL Backbones (HuBERT, WavLM, etc.)
        if (
            self.semantic_teacher == "hubert_base"
            or self.semantic_teacher == "hubert_base_general"
            or self.semantic_teacher == "wavlm_base_plus"
        ):
            # Standard SSL models usually expect mono input
            mono_input = resampled_input[:, 0, :]
            mono_input = F.pad(mono_input, (160, 160))
            
            # Extract hidden states (semantic information)
            outputs = self.semantic_model(mono_input, output_hidden_states=True).hidden_states
            stacked_outputs = torch.stack(outputs, dim=1)

            # Average layers to get a robust semantic representation
            semantic_vectors = stacked_outputs.mean(1)

        elif self.semantic_teacher == "w2v_bert2":
            semantic_vectors = self.semantic_model(resampled_input)

        elif self.semantic_teacher.startswith("whisper"):
            if self.last_layer_semantic:
                semantic_vectors = self.semantic_model(resampled_input, avg_layers=False)
            else:
                semantic_vectors = self.semantic_model(resampled_input, avg_layers=True)

        elif self.semantic_teacher.startswith("mert_music"):
            if self.last_layer_semantic:
                semantic_vectors = self.semantic_model(resampled_input, avg_layers=False)
            else:
                semantic_vectors = self.semantic_model(resampled_input, avg_layers=True)

        elif self.semantic_teacher.startswith("qwen_audio_omni"):
            semantic_vectors = self.semantic_model(resampled_input)

        # Post-Processing / Downsampling
        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                semantic_vectors = semantic_vectors[:, :: self.semantic_downsample_factor, :]

        elif self.downsample_mode == "avg":
            # Pool features if using average downsampling
            semantic_vectors = self.semantic_pooling(semantic_vectors.transpose(1, 2)).transpose(1, 2)
        
        return semantic_vectors

    def _calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    @staticmethod
    def _get_semantic_model_parameters(semantic_teacher: str) -> tuple[str, int, int, int]:
        """
        Get semantic model parameters for a given semantic teacher.

        Args:
            semantic_teacher: The semantic teacher model identifier.

        Returns:
            A tuple of (semantic_model_name, semantic_sample_rate, semantic_dim, semantic_encoder_dim).
        """
        if semantic_teacher == "hubert_base":
            semantic_model_name = "facebook/hubert-base-ls960"
            semantic_sample_rate = 16000
            semantic_dim = 768
            semantic_encoder_dim = 768
        elif semantic_teacher == "wavlm_base_plus":
            semantic_model_name = "microsoft/wavlm-base-plus"
            semantic_sample_rate = 16000
            semantic_dim = 768
            semantic_encoder_dim = 768
        elif semantic_teacher == "hubert_base_general":
            semantic_model_name = "bosonai/hubert_base"
            semantic_sample_rate = 16000
            semantic_dim = 768
            semantic_encoder_dim = 768
        else:
            raise ValueError(f"Unsupported semantic_teacher: {semantic_teacher}")

        return semantic_model_name, semantic_sample_rate, semantic_dim, semantic_encoder_dim

    def _xcodec_encode(
        self, raw_waveform: torch.Tensor, target_bandwidth: Optional[int] = None
    ) -> EncodedResult:
        """
        Executes the X-Codec inference pipeline:
        1) Extraction: Get semantic vectors from the frozen SSL model (WavLM/HuBERT).
        2) Encoding: Process parallel Semantic and Acoustic streams.
        3) Alignment: Temporally align semantic and acoustic features.
        4) Fusion: Merge streams and project meaningful features.
        5) Quantization: Convert fused features into discrete acoustic tokens.

        Args:
            raw_waveform: Audio tensor shaped [batch, channels, samples].
            target_bandwidth: Optional bitrate limit for the Residual Vector Quantizer (RVQ).
        """
        # 1. Semantic Stream: Extract raw features from the frozen SSL model
        # The paper refers to this as the "Semantic Encoder" pathway using a frozen model
        ssl_semantic_vectors = self._extract_semantic_features(raw_waveform).detach()

        # Encode the semantic vectors into the model's internal latent space
        # (Transposing to match channel-last/first expectations of the sub-modules)
        semantic_latents = self.semantic_encoder(ssl_semantic_vectors.transpose(1, 2))

        # 2. Acoustic Stream: Extract low-level acoustic features
        acoustic_latents = self.acoustic_encoder(raw_waveform)

        # 3. Temporal Alignment
        # The semantic and acoustic streams may have different downsampling rates.
        # This block ensures the acoustic features match the temporal dimension of the semantic features.
        if acoustic_latents.shape[2] != semantic_latents.shape[2]:
            pad_size = 160 * self.semantic_downsample_factor
            # Re-encode with padding if alignment fails (Handling edge cases in temporal resolution)
            acoustic_latents = self.acoustic_encoder(
                F.pad(raw_waveform[:, 0, :], (pad_size, pad_size)).unsqueeze(0)
            )

        # Truncate to ensure exact length match
        if acoustic_latents.shape[2] != semantic_latents.shape[2]:
            min_length = min(acoustic_latents.shape[2], semantic_latents.shape[2])
            acoustic_latents = acoustic_latents[:, :, :min_length]
            semantic_latents = semantic_latents[:, :, :min_length]

        # 4. Fusion
        # Concatenate acoustic and semantic features along the channel dimension
        combined_latents = torch.cat([acoustic_latents, semantic_latents], dim=1)

        # Project fused features (The paper describes "guiding" acoustic quantization with semantic data)
        # Assuming self.fc_prior acts as the Fusion Layer described in paper
        fused_latents = self.fc_prior(combined_latents.transpose(1, 2))

        # 5. Residual Vector Quantization (RVQ)
        if self.quantizer_type == "RVQ":
            fused_latents = fused_latents.transpose(1, 2)
            quantized_latents, code_indices, bandwidth, commit_loss = self.quantizer(
                fused_latents, self.frame_rate, target_bandwidth
            )
            # Permute to [batch, codes, time]
            code_indices = code_indices.permute(1, 0, 2)
        else:
            quantized_latents, code_indices = self.quantizer(fused_latents)
            code_indices = code_indices.permute(0, 2, 1)

        return EncodedResult(code_indices)
