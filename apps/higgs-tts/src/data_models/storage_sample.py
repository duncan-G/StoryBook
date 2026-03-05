import torch
from dataclasses import dataclass


@dataclass
class HiggsAudioModelInputStorageSample:
    input_tokens: torch.LongTensor
    label_tokens: torch.LongTensor
    audio_bytes_cache_dir_index: int
    audio_codes_cache_dir_index: int
    audio_bytes_indices: torch.LongTensor
    audio_codes_indices: torch.LongTensor
    speaker_indices: torch.LongTensor
    file_index: int
    original_sample_index: int
