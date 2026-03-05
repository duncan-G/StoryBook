import math
from typing import List, Optional, Tuple

import librosa
import torch
import torch.nn.functional as F
from transformers.models.whisper.processing_whisper import WhisperProcessor

from src.data_models.model_batch_input import HiggsAudioBatchModelInput
from src.data_models.model_input import HiggsAudioModelInput
from src.audio_tokenizer.utils import build_delay_pattern_mask


def _ceil_to_nearest(n: int, round_to: int) -> int:
    """Round `n` up to the nearest multiple of `round_to`."""
    return (n + round_to - 1) // round_to * round_to


class HiggsAudioDataCollator:
    """Collate `ModelInput` samples into a batched `ModelBatchInput`.

    This collator:

      * Pads text input IDs and label IDs to a common length.
      * Associates each `<|audio_bos|><|AUDIO|><|audio_eos|>` span with:
          - Audio codebooks (discrete audio tokens)
          - Optional Whisper features (if `encode_whisper_embed=True`)
      * Handles long audio by:
          - Splitting the waveform into fixed-duration chunks.
          - Duplicating the corresponding audio token triplet for each chunk.
      * Optionally:
          - Adds audio-stream BOS/EOS tokens around audio codebooks.
          - Applies a “delay pattern” mask to audio code sequences.
          - Masks labels for `<|AUDIO_OUT|>` tokens.

    Args:
        whisper_processor: Whisper processor used to compute audio features.
        audio_in_token_id: Token ID representing `<|AUDIO_IN|>` (or equivalent) in the text stream.
        audio_out_token_id: Token ID representing `<|AUDIO_OUT|>` (or equivalent) in the text stream.
        pad_token_id: Token ID used for padding text `input_ids`.
        audio_stream_bos_id: BOS ID used inside the audio code streams.
        audio_stream_eos_id: EOS ID used inside the audio code streams.
        round_to: Pad sequence length up to a multiple of this value.
        pad_left: If True, pad on the left; otherwise pad on the right.
        encode_whisper_embed: If True, compute Whisper features for audio inputs.
        return_audio_in_tokens: If True, return audio-in codebooks (`audio_in_ids`).
        audio_num_codebooks: If not None, truncate audio codebooks to this many codebooks.
        use_delay_pattern: If True, apply delay pattern mask to audio code sequences.
        disable_audio_codes_transform: If True, do not add BOS/EOS to audio code sequences.
        chunk_size_seconds: Maximum duration (in seconds) of each audio chunk when splitting long audio.
        add_new_bos_eos_for_long_chunk: Unused here but kept for compatibility.
        mask_audio_out_token_label: If True, always mask the label at `<|AUDIO_OUT|>` (loss ignored).
    """

    def __init__(
        self,
        whisper_processor: WhisperProcessor,
        audio_in_token_id: int,
        audio_out_token_id: int,
        pad_token_id: int,
        audio_stream_bos_id: int,
        audio_stream_eos_id: int,
        round_to: int = 8,
        pad_left: bool = False,
        encode_whisper_embed: bool = True,
        return_audio_in_tokens: bool = True,
        audio_num_codebooks: Optional[int] = None,
        use_delay_pattern: bool = False,
        disable_audio_codes_transform: bool = False,
        chunk_size_seconds: int = 30,  # Maximum duration for each audio chunk
        add_new_bos_eos_for_long_chunk: bool = True,
        mask_audio_out_token_label: bool = True,
    ) -> None:
        self.whisper_processor = whisper_processor
        self.round_to = round_to
        self.pad_left = pad_left

        # Special token IDs in the text stream
        self.audio_in_token_id = audio_in_token_id
        self.audio_out_token_id = audio_out_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.pad_token_id = pad_token_id

        # Audio-related options
        self.encode_whisper_embed = encode_whisper_embed
        self.return_audio_in_tokens = return_audio_in_tokens
        self.audio_num_codebooks = audio_num_codebooks
        self.use_delay_pattern = use_delay_pattern
        self.disable_audio_codes_transform = disable_audio_codes_transform
        self.add_new_bos_eos_for_long_chunk = add_new_bos_eos_for_long_chunk
        self.mask_audio_out_token_label = mask_audio_out_token_label

        # Pre-compute chunk size in samples if we’re going to encode Whisper features
        if encode_whisper_embed:
            self.chunk_size_seconds = chunk_size_seconds
            self.chunk_size_samples = int(
                chunk_size_seconds * whisper_processor.feature_extractor.sampling_rate
            )
        else:
            self.chunk_size_seconds = None
            self.chunk_size_samples = None

    def _process_and_duplicate_audio_tokens(
        self,
        input_ids: torch.Tensor,
        audio_idx: int,
        wv: torch.Tensor,
        sr: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Handle long audio by duplicating its `<|audio_bos|><|AUDIO|><|audio_eos|>` sequence.

        When a waveform is longer than `self.chunk_size_samples`, we conceptually split it
        into N chunks and duplicate the corresponding 3 token span (audio_bos, AUDIO, audio_eos)
        N times in the text sequence.

        Args:
            input_ids: 1D tensor of text input IDs for a single example.
            audio_idx: Index in `input_ids` for the central `<|AUDIO|>` token.
            wv: 1D waveform tensor for the corresponding audio.
            sr: Sampling rate of `wv`.
            labels: Optional 1D label tensor aligned with `input_ids`.

        Returns:
            new_input_ids: 1D tensor with the audio triplet duplicated for each chunk.
            new_labels:   1D label tensor duplicated accordingly (if provided), else None.
            num_chunks:   Number of chunks (1 if no splitting occurs).
        """
        total_samples = len(wv)
        num_chunks = math.ceil(total_samples / self.chunk_size_samples)

        # If audio fits in a single chunk, no duplication is needed.
        if num_chunks <= 1:
            return input_ids, labels, 1

        # The audio sequence in text is `<|audio_bos|><|AUDIO|><|audio_eos|>`.
        # Given `audio_idx` points at `<|AUDIO|>`, we take one token before and one after.
        audio_token_seq = input_ids[audio_idx - 1 : audio_idx + 2]  # shape: [3]

        # Duplicate that 3-token sequence `num_chunks` times, flattening along the 1D axis.
        duplicated_sequence = audio_token_seq.repeat(num_chunks)

        # Stitch new text sequence:
        #   [tokens before audio_bos] + [duplicated audio tokens] + [tokens after audio_eos]
        new_input_ids = torch.cat(
            [input_ids[: audio_idx - 1], duplicated_sequence, input_ids[audio_idx + 2 :]]
        )

        new_labels = None
        if labels is not None:
            # Duplicate the corresponding label span as well.
            label_seq = labels[audio_idx - 1 : audio_idx + 2]
            duplicated_labels = label_seq.repeat(num_chunks)
            new_labels = torch.cat(
                [labels[: audio_idx - 1], duplicated_labels, labels[audio_idx + 2 :]]
            )

        return new_input_ids, new_labels, num_chunks

    def __call__(self, batch: List[HiggsAudioModelInput]) -> HiggsAudioBatchModelInput:
        """Collate a list of `HiggsAudioModelInput` samples into a single `HiggsAudioBatchModelInput`.

        This supports:
          * Long audio splitting & token duplication (if `encode_whisper_embed=True`).
          * Whisper feature extraction.
          * Audio codebook aggregation for `<|audio_in|>` and `<|audio_out|>`.
          * Optional label construction for audio codebooks.
          * Padding text sequences to a common length.
        """
        label_ids = None
        label_audio_ids = None

        # Determine whether we should return label IDs at all.
        if all(ele.label_ids is None for ele in batch):
            return_labels = False
        else:
            return_labels = True

        # ---------------------------------------------------------------------
        # Step 1: Handle long audio (if we are using Whisper embeddings).
        #         Long audio is split into chunks, and the corresponding
        #         audio token triplets in the text sequence are duplicated.
        # ---------------------------------------------------------------------
        if self.encode_whisper_embed:
            processed_batch: List[HiggsAudioModelInput] = []

            for sample_idx, sample in enumerate(batch):
                # Find all `<|AUDIO_IN|>` and `<|AUDIO_OUT|>` positions in the text sequence.
                audio_in_mask = sample.input_ids == self.audio_in_token_id
                audio_in_indices = torch.where(audio_in_mask)[0]
                audio_out_mask = sample.input_ids == self.audio_out_token_id

                # We will build modified text/labels and waveforms as we split/duplicate.
                modified_input_ids = sample.input_ids
                modified_labels = sample.label_ids if return_labels else None

                # These lists accumulate possibly-chunked waveforms and their offsets.
                modified_waveforms_concat = []
                modified_waveforms_start = []
                modified_sample_rate = []
                offset = 0  # Tracks how much the text positions shift due to duplication.
                curr_wv_offset = 0  # Offset inside the concatenated waveform tensor.

                # Process all `<|AUDIO_IN|>` tokens in this sample.
                for idx, audio_idx in enumerate(audio_in_indices):
                    # `idx` here is the “audio index” used in `get_wv(idx)` and
                    # refers to the original audio order, not the text position.
                    wv, sr = sample.get_wv(idx)

                    # Resample audio to match Whisper's feature extractor sampling rate.
                    target_sr = self.whisper_processor.feature_extractor.sampling_rate
                    # Convert tensor to Python number if needed
                    sr = sr.item() if isinstance(sr, torch.Tensor) else sr
                    if sr != target_sr:
                        resampled_wv = librosa.resample(
                            wv.cpu().numpy(),
                            orig_sr=sr,
                            target_sr=target_sr,
                        )
                    else:
                        resampled_wv = wv.cpu().numpy()

                    wv = torch.tensor(resampled_wv, device=wv.device)
                    sr = target_sr

                    # Adjust for previous token insertions/duplications.
                    token_pos = audio_idx + offset

                    # Duplicate the `<|audio_bos|><|AUDIO|><|audio_eos|>` triplet
                    # if the waveform needs to be split.
                    modified_input_ids, modified_labels, num_chunks = self._process_and_duplicate_audio_tokens(
                        modified_input_ids, token_pos, wv, sr, modified_labels
                    )

                    # Slice the waveform into `num_chunks` time segments and accumulate.
                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * self.chunk_size_samples
                        chunk_end = min((chunk_idx + 1) * self.chunk_size_samples, len(wv))
                        chunk_wv = wv[chunk_start:chunk_end]
                        modified_waveforms_concat.append(chunk_wv)
                        modified_waveforms_start.append(curr_wv_offset)
                        curr_wv_offset += len(chunk_wv)
                        modified_sample_rate.append(sr)

                    # Each new chunk introduces 3 additional tokens (the audio triplet).
                    offset += (num_chunks - 1) * 3

                # Build a new `ModelInput` with updated text and audio fields.
                processed_sample = HiggsAudioModelInput(
                    input_ids=modified_input_ids,
                    label_ids=modified_labels if return_labels else sample.label_ids,
                    audio_ids_concat=sample.audio_ids_concat,
                    audio_ids_start=sample.audio_ids_start,
                    audio_waveforms_concat=torch.cat(modified_waveforms_concat)
                    if modified_waveforms_concat
                    else sample.audio_waveforms_concat,
                    audio_waveforms_start=torch.tensor(modified_waveforms_start, dtype=torch.long)
                    if modified_waveforms_start
                    else sample.audio_waveforms_start,
                    audio_sample_rate=torch.tensor(modified_sample_rate)
                    if modified_sample_rate
                    else sample.audio_sample_rate,
                    audio_speaker_indices=torch.tensor([]),
                    # NOTE: The comment from original code:
                    # FIXME(sxjscience): The logic here is not correct for audio_label_ids_concat.
                    audio_label_ids_concat=sample.audio_label_ids_concat,
                )

                processed_batch.append(processed_sample)
        else:
            # If we don't need Whisper embeddings, we keep the batch as-is.
            processed_batch = batch

        # ---------------------------------------------------------------------
        # Step 2: Compute max sequence length (for text) and pad length target.
        # ---------------------------------------------------------------------
        max_seq_length = _ceil_to_nearest(
            max(len(sample.input_ids) for sample in processed_batch),
            self.round_to,
        )

        # Containers for audio-related outputs.
        audio_in_wv_l = []          # List of audio waveforms for Whisper features.
        audio_in_ids_l = []         # List of audio-in code tensors (per audio instance).
        audio_out_ids_l = []        # List of audio-out code tensors (per audio instance).
        audio_out_ids_group_loc_l = []  # For each audio-out group, which batch index it came from.

        audio_in_label_ids_l = None     # Optional labels for audio-in codebooks.
        audio_out_label_ids_l = None    # Optional labels for audio-out codebooks.
        reward_l = []                   # Per-sample reward values (if present).

        if return_labels:
            # Tracks which audio-out instances should NOT be trained on.
            audio_out_no_train_flag = []

        # ---------------------------------------------------------------------
        # Step 3: For each sample, map `<|audio_in|>` / `<|audio_out|>` tokens to
        #         audio codebooks and build mapping between text and audio indices.
        # ---------------------------------------------------------------------
        for i, sample in enumerate(processed_batch):
            # Logical masks to find `<|audio_in|>` and `<|audio_out|>` tokens.
            audio_in_mask = sample.input_ids == self.audio_in_token_id
            audio_out_mask = sample.input_ids == self.audio_out_token_id

            # `audio_ids` maps each audio token occurrence to a monotonically increasing ID.
            # We use XOR so that positions with exactly one of (audio_in, audio_out) are 1.
            audio_ids = torch.ones_like(sample.input_ids)
            audio_ids[audio_in_mask ^ audio_out_mask] = (
                torch.cumsum(audio_ids[audio_in_mask ^ audio_out_mask], 0) - 1
            )
            audio_in_ids = audio_ids[audio_in_mask]
            audio_out_ids = audio_ids[audio_out_mask]

            if return_labels:
                # `label_ids < 0` means masked labels (e.g., -100). We mark those audio-out
                # instances as "no train" for their audio code sequences.
                audio_out_no_train_flag.append(sample.label_ids[audio_out_mask] < 0)
                if self.mask_audio_out_token_label:
                    # Optionally mask the `<|AUDIO_OUT|>` position itself in the labels.
                    sample.label_ids[audio_out_mask] = -100

            # ----------------------
            # Audio-in codebooks
            # ----------------------
            if self.return_audio_in_tokens:
                # Collect the discrete audio codes for each `<|audio_in|>`.
                audio_in_ids_l.extend(
                    [sample.get_audio_codes(idx)[: self.audio_num_codebooks, :] for idx in audio_in_ids]
                )
                # Optional label codebooks for audio-in.
                if sample.audio_label_ids_concat is not None:
                    if audio_in_label_ids_l is None:
                        audio_in_label_ids_l = []
                    audio_in_label_ids_l.extend(
                        [
                            sample.get_audio_codes_labels(idx)[: self.audio_num_codebooks, :]
                            for idx in audio_in_ids
                        ]
                    )

            # ----------------------
            # Audio-out codebooks
            # ----------------------
            audio_out_ids_l.extend(
                [sample.get_audio_codes(idx)[: self.audio_num_codebooks, :] for idx in audio_out_ids]
            )
            audio_out_ids_group_loc_l.append(i)

            if sample.reward is not None:
                reward_l.append(sample.reward)

            if sample.audio_label_ids_concat is not None:
                if audio_out_label_ids_l is None:
                    audio_out_label_ids_l = []
                audio_out_label_ids_l.extend(
                    [
                        sample.get_audio_codes_labels(idx)[: self.audio_num_codebooks, :]
                        for idx in audio_out_ids
                    ]
                )

            # ----------------------
            # Audio waveforms for Whisper feature extraction
            # ----------------------
            if self.encode_whisper_embed:
                for idx in audio_in_ids:
                    wv, sr = sample.get_wv(idx)
                    resampled_wv = wv.cpu().numpy()
                    total_samples = len(resampled_wv)

                    # Split into fixed-size chunks for Whisper feature extraction.
                    for chunk_start in range(0, total_samples, self.chunk_size_samples):
                        chunk_end = min(chunk_start + self.chunk_size_samples, total_samples)
                        chunk = resampled_wv[chunk_start:chunk_end]
                        audio_in_wv_l.append(chunk)

        if return_labels:
            # Flatten the boolean "no-train" flags into a 1D tensor.
            audio_out_no_train_flag = torch.cat(audio_out_no_train_flag, dim=0)

        # ---------------------------------------------------------------------
        # Step 4: Run Whisper feature extractor on all audio chunks (if any).
        # ---------------------------------------------------------------------
        if len(audio_in_wv_l) > 0:
            feature_ret = self.whisper_processor.feature_extractor(
                audio_in_wv_l,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_attention_mask=True,
                padding="max_length",
            )
            audio_features = torch.from_numpy(feature_ret["input_features"])
            audio_feature_attention_mask = torch.from_numpy(feature_ret["attention_mask"])
        else:
            if self.encode_whisper_embed:
                # If we expect features but have no audio, return empty tensors
                # with the correct shape layout.
                audio_features = torch.zeros(
                    (
                        0,
                        self.whisper_processor.feature_extractor.feature_size,
                        self.whisper_processor.feature_extractor.nb_max_frames,
                    ),
                    dtype=torch.float32,
                )
                audio_feature_attention_mask = torch.zeros(
                    (0, self.whisper_processor.feature_extractor.nb_max_frames),
                    dtype=torch.int32,
                )
            else:
                audio_features = None
                audio_feature_attention_mask = None

        # ---------------------------------------------------------------------
        # Step 5: Build audio-in token sequences (discrete codebooks).
        # ---------------------------------------------------------------------
        if len(audio_in_ids_l) > 0:
            new_audio_in_ids_l = []
            for ele in audio_in_ids_l:
                if self.disable_audio_codes_transform:
                    # Do not add audio-stream BOS/EOS.
                    # Typically used when codes are already shaped by some dataset.
                    audio_codes = ele
                else:
                    # Add audio-stream BOS/EOS tokens around each code sequence.
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long, device=ele.device),
                            ele,
                            torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long, device=ele.device),
                        ],
                        dim=1,
                    )
                    if self.use_delay_pattern:
                        # Optionally transform codes into a delay pattern representation.
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                new_audio_in_ids_l.append(audio_codes)

            # Concatenate along the time dimension; codebook dimension is along axis 0.
            audio_in_ids = torch.cat(new_audio_in_ids_l, dim=1).long()

            # Compute start indices for each audio-in sequence.
            device = new_audio_in_ids_l[0].device if new_audio_in_ids_l else torch.device("cpu")
            audio_in_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_in_ids_l[:-1]], device=device),
                dim=0,
            )
        else:
            # If there are no audio-in codes, return empty placeholders.
            audio_in_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_in_ids_start = torch.zeros(0, dtype=torch.long)

        # ---------------------------------------------------------------------
        # Step 6: Build audio-out token sequences (discrete codebooks + labels).
        # ---------------------------------------------------------------------
        audio_out_ids_start_group_loc = None
        if len(audio_out_ids_l) > 0:
            new_audio_out_ids_l = []
            label_audio_ids_l = []

            for idx, ele in enumerate(audio_out_ids_l):
                if self.disable_audio_codes_transform:
                    # Again, do not add audio-stream BOS/EOS.
                    audio_codes = ele
                    if return_labels:
                        label_audio_ids = audio_out_label_ids_l[idx]
                else:
                    # Add BOS/EOS to audio-out codes, and construct label sequence.
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long, device=ele.device),
                            ele,
                            torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long, device=ele.device),
                        ],
                        dim=1,
                    )
                    if return_labels:
                        # Labels for audio codes: BOS is masked, EOS is a real token.
                        # Use label codes if available, otherwise use audio codes
                        if audio_out_label_ids_l and idx < len(audio_out_label_ids_l):
                            label_ele = audio_out_label_ids_l[idx]
                        else:
                            label_ele = ele
                        label_audio_ids = torch.cat(
                            [
                                torch.full((label_ele.shape[0], 1), -100, dtype=torch.long, device=label_ele.device),
                                label_ele,
                                torch.full((label_ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long, device=label_ele.device),
                            ],
                            dim=1,
                        )
                    if self.use_delay_pattern:
                        # Apply delay pattern to both codes and labels (with -100 as "masked").
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                        if return_labels:
                            label_audio_ids = build_delay_pattern_mask(
                                label_audio_ids.unsqueeze(0),
                                bos_token_id=-100,
                                pad_token_id=-100,
                            )[0].squeeze(0)

                new_audio_out_ids_l.append(audio_codes)

                if return_labels:
                    # If this audio-out should not be trained at all, mask its labels.
                    if audio_out_no_train_flag[idx]:
                        label_audio_ids[:] = -100
                    label_audio_ids_l.append(label_audio_ids)

            # Concatenate all audio-out codes/labels.
            audio_out_ids = torch.cat(new_audio_out_ids_l, dim=1).long()
            if return_labels:
                label_audio_ids = torch.cat(label_audio_ids_l, dim=1).long()

            # Start indices for each audio-out sequence.
            device = new_audio_out_ids_l[0].device if new_audio_out_ids_l else torch.device("cpu")
            audio_out_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_out_ids_l[:-1]], device=device),
                dim=0,
            )
            # For each audio-out group, store which sample index it came from.
            audio_out_ids_start_group_loc = torch.tensor(audio_out_ids_group_loc_l, dtype=torch.long, device=device)
        else:
            audio_out_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_out_ids_start = torch.zeros(0, dtype=torch.long)
            if return_labels:
                label_audio_ids = torch.zeros((0, 0), dtype=torch.long)

        # Rewards: simple tensor from list (may be empty).
        reward = torch.tensor(reward_l, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # Step 7: Pad text `input_ids` and (optionally) `label_ids` to max_seq_length.
        # ---------------------------------------------------------------------
        if self.pad_left:
            # Left padding: pad at the beginning of the sequence.
            input_ids = torch.stack(
                [
                    F.pad(
                        sample.input_ids,
                        (max_seq_length - len(sample.input_ids), 0),
                        value=self.pad_token_id,
                    )
                    for sample in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(
                            sample.label_ids,
                            (max_seq_length - len(sample.label_ids), 0),
                            value=-100,
                        )
                        for sample in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(
                        torch.ones_like(sample.input_ids),
                        (max_seq_length - len(sample.input_ids), 0),
                        value=0,
                    )
                    for sample in processed_batch
                ]
            )
        else:
            # Right padding: pad at the end of the sequence.
            input_ids = torch.stack(
                [
                    F.pad(
                        sample.input_ids,
                        (0, max_seq_length - len(sample.input_ids)),
                        value=self.pad_token_id,
                    )
                    for sample in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(
                            sample.label_ids,
                            (0, max_seq_length - len(sample.label_ids)),
                            value=-100,
                        )
                        for sample in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(
                        torch.ones_like(sample.input_ids),
                        (0, max_seq_length - len(sample.input_ids)),
                        value=0,
                    )
                    for sample in processed_batch
                ]
            )

        # If we don't want to return audio-in tokens at all, zero them out.
        if not self.return_audio_in_tokens:
            audio_in_ids = None
            audio_in_ids_start = None

        # ---------------------------------------------------------------------
        # Step 8: Optionally truncate to a fixed number of codebooks.
        # ---------------------------------------------------------------------
        if self.audio_num_codebooks is not None:
            if audio_in_ids is not None:
                audio_in_ids = audio_in_ids[: self.audio_num_codebooks]
            if audio_out_ids is not None:
                audio_out_ids = audio_out_ids[: self.audio_num_codebooks]
            if label_audio_ids is not None:
                label_audio_ids = label_audio_ids[: self.audio_num_codebooks]

        # ---------------------------------------------------------------------
        # Step 9: Return the batched model input.
        # ---------------------------------------------------------------------
        return HiggsAudioBatchModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
            reward=reward,
        )
