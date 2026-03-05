import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class HiggsAudioBatchModelInput:
    input_ids: torch.LongTensor  # shape (bsz, seq_len).
    attention_mask: torch.Tensor  # shape (bsz, seq_len).
    audio_features: Optional[torch.Tensor]  # shape (num_audio_in, feature_dim, max_mel_seq_len).
    audio_feature_attention_mask: Optional[torch.Tensor]  # shape (num_audio_in, max_mel_seq_len).
    audio_out_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: Optional[torch.LongTensor]  # shape (num_audio_out,)
    # The audio_out_ids_start_group_loc has the same length as audio_out_ids_start. It is used to recover group location in a batch for an audio segment
    # Currently, we concatenante audio segments along dim 0 to handle variadic audio segment length. However, in the alignment stage, we need the location information
    # For example,
    #  audio_out_ids_start = [0, 2, 4, 8]; and the first two audio segments come from the same sample in a batch, and other two come from different samples.
    #  This is a batch of 3 samples, then we will have the group location as:
    #  audio_out_ids_start_group_loc = [0, 0, 1, 2]
    audio_out_ids_start_group_loc: Optional[
        torch.LongTensor
    ]  # shape (num_audio_out,), specify which a sample's group location in the batch
    audio_in_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: Optional[torch.LongTensor]  # shape (num_audio_in,)
    label_ids: Optional[torch.LongTensor]  # shape (bsz, seq_len)
    label_audio_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    reward: Optional[float] = None # Optional reward value
