from __future__ import annotations

import base64
import uuid
from io import BytesIO
from typing import List, Tuple, Optional, Union

import librosa
import numpy as np
import torch
from transformers import AutoTokenizer

from src.data_models.generation_input import GenerationInput
from src.data_models.message import Message
from src.data_models.message_content import TextContent, AudioContent
from src.data_models.speaker import Speaker
from src.data_models.constants import (
    AUDIO_IN_TOKEN,
    AUDIO_OUT_TOKEN,
    AUDIO_BOS,
    AUDIO_EOS,
    AUDIO_OUT_BOS,
    AUDIO_PLACEHOLDER_TOKEN,
    BEGIN_OF_TEXT,
    START_HEADER_ID,
    END_HEADER_ID,
    RECIPIENT,
    EOT_ID,
    EOM_ID,
    SCENE_DESC_START,
    SCENE_DESC_END,
    DEFAULT_SYSTEM_MESSAGE,
)
from src.data_models.model_input import HiggsAudioModelInput
from src.audio_tokenizer.higgs_audio_tokenizer import HiggsAudioTokenizer

_SOUND_EFFECT_MAP = [
    ("[laugh]", "<SE>[Laughter]</SE>"),
    ("[humming start]", "<SE_s>[Humming]</SE_s>"),
    ("[humming end]", "<SE_e>[Humming]</SE_e>"),
    ("[music start]", "<SE_s>[Music]</SE_s>"),
    ("[music end]", "<SE_e>[Music]</SE_e>"),
    ("[music]", "<SE>[Music]</SE>"),
    ("[sing start]", "<SE_s>[Singing]</SE_s>"),
    ("[sing end]", "<SE_e>[Singing]</SE_e>"),
    ("[applause]", "<SE>[Applause]</SE>"),
    ("[cheering]", "<SE>[Cheering]</SE>"),
    ("[cough]", "<SE>[Cough]</SE>"),
]

_TERMINAL_PUNCTUATION = frozenset({
    ".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>",
})

class InputProcessor:
    """Converts generation inputs into model-ready tensors for a multimodal
    (text + audio) model.

    Two-stage pipeline:

    1. **Preparation** – :meth:`prepare` takes a :class:`GenerationInput`
       and returns a ``List[Message]`` (system message, speaker audio
       pairs, normalized user prompt).
    2. **Tokenization** – :meth:`process_input` converts a
       ``List[Message]`` into a :class:`HiggsAudioModelInput` tensor
       ready for the model.
    """

    def __init__(
        self,
        text_tokenizer: AutoTokenizer,
        audio_tokenizer: HiggsAudioTokenizer,
        device: Optional[torch.device] = None,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device if device is not None else torch.device('cpu')

    # =====================================================================
    # Stage 1: Preparation  (GenerationInput → List[Message])
    # =====================================================================

    def prepare(self, gen_input: GenerationInput) -> List[Message]:
        """Build the internal message list from a :class:`GenerationInput`."""
        speaker_index_map, msg_speaker_labels = self._assign_speaker_labels(
            gen_input.messages
        )

        indexed_speakers = self._collect_indexed_speakers(
            gen_input.messages, speaker_index_map
        )
        has_undescribed = any(m.speaker is None for m in gen_input.messages)

        messages: List[Message] = [
            self.build_system_message(
                indexed_speakers,
                system_prompt=gen_input.system_prompt,
                scene_description=gen_input.scene_description,
                has_undescribed_speakers=has_undescribed,
            )
        ]
        messages.extend(self._build_speaker_audio_pairs(indexed_speakers))

        for msg, label in zip(gen_input.messages, msg_speaker_labels):
            normalized_msg = self._normalize_message(msg, label)
            messages.append(normalized_msg)

        return messages

    def _normalize_message(self, msg: Message, speaker_label: str) -> Message:
        """Normalize a message and prepend its speaker label to text content."""
        if isinstance(msg.content, TextContent):
            text = self.normalize_prompt(msg.content.text)
            return Message(
                role=msg.role,
                speaker=msg.speaker,
                content=TextContent(text=f"[{speaker_label}] {text}"),
            )

        if isinstance(msg.content, list):
            normalized_parts = []
            for i, item in enumerate(msg.content):
                if isinstance(item, TextContent):
                    text = self.normalize_prompt(item.text)
                    if i == 0:
                        text = f"[{speaker_label}] {text}"
                    normalized_parts.append(TextContent(text=text))
                else:
                    normalized_parts.append(item)
            return Message(role=msg.role, speaker=msg.speaker, content=normalized_parts)

        return msg

    @staticmethod
    def _assign_speaker_labels(
        messages: List[Message],
    ) -> Tuple[dict, List[str]]:
        """Assign ``SPEAKER<i>`` labels to every message.

        Speakers with a :class:`Speaker` object are deduplicated by UUID.
        Messages without a speaker each receive a unique label.

        Returns ``(speaker_index_map, per_message_labels)`` where
        *speaker_index_map* maps ``Speaker.uuid → int`` and
        *per_message_labels* is a list of label strings parallel to
        *messages*.
        """
        speaker_index_map: dict[uuid.UUID, int] = {}
        next_idx = 0

        for m in messages:
            if m.speaker and m.speaker.uuid not in speaker_index_map:
                speaker_index_map[m.speaker.uuid] = next_idx
                next_idx += 1

        labels: List[str] = []
        for m in messages:
            if m.speaker:
                labels.append(f"SPEAKER{speaker_index_map[m.speaker.uuid]}")
            else:
                labels.append(f"SPEAKER{next_idx}")
                next_idx += 1

        return speaker_index_map, labels

    @staticmethod
    def _collect_indexed_speakers(
        messages: List[Message],
        speaker_index_map: dict,
    ) -> List[Tuple[int, Speaker]]:
        """Return ``(index, Speaker)`` pairs ordered by index."""
        seen: set = set()
        result: List[Tuple[int, Speaker]] = []
        for m in messages:
            if m.speaker and m.speaker.uuid not in seen:
                seen.add(m.speaker.uuid)
                result.append((speaker_index_map[m.speaker.uuid], m.speaker))
        result.sort(key=lambda x: x[0])
        return result

    @staticmethod
    def normalize_prompt(text: str) -> str:
        """Normalize prompt text for audio generation.

        Handles parentheses removal, temperature symbol expansion,
        sound-effect tag conversion, whitespace normalization, and
        ensures text ends with terminal punctuation.
        """
        text = text.replace("(", " ").replace(")", " ")
        text = text.replace("°F", " degrees Fahrenheit")
        text = text.replace("°C", " degrees Celsius")

        for tag, replacement in _SOUND_EFFECT_MAP:
            text = text.replace(tag, replacement)

        lines = text.split("\n")
        text = "\n".join(" ".join(line.split()) for line in lines if line.strip())
        text = text.strip()

        if not any(text.endswith(p) for p in _TERMINAL_PUNCTUATION):
            text += "."

        return text

    @staticmethod
    def build_system_message(
        speakers: List[Tuple[int, Speaker]],
        system_prompt: Optional[str] = None,
        scene_description: Optional[str] = None,
        has_undescribed_speakers: bool = False,
    ) -> Message:
        """Build a system :class:`Message`.

        The message is constructed from three optional parts:

        * ``system_prompt`` – free-form instruction text prepended before
          the scene block.  Falls back to :data:`DEFAULT_SYSTEM_MESSAGE`
          when not provided.
        * ``scene_description`` – environment context placed inside the
          scene descriptor tags.
        * ``speakers`` – ``(index, Speaker)`` pairs whose descriptions
          (or audio placeholders for voice-cloning speakers) are appended
          inside the scene block.
        * ``has_undescribed_speakers`` – when *True*, an extra instruction
          is added asking the model to select appropriate voices for
          speakers that lack a description.
        """
        base_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_MESSAGE

        speaker_lines = []
        for idx, speaker in speakers:
            label = f"SPEAKER{idx}"
            if speaker.audio_url:
                speaker_lines.append(f"{label}: {AUDIO_PLACEHOLDER_TOKEN}")
            else:
                speaker_lines.append(f"{label}: {speaker.description}")

        if has_undescribed_speakers:
            speaker_lines.append(
                "Some speakers do not have a voice description. "
                "Select an appropriate and distinct voice for each of them."
            )

        system_text = "\n".join([
            base_prompt,
            "",
            SCENE_DESC_START,
            scene_description or "",
            "",
            "\n".join(speaker_lines),
            SCENE_DESC_END,
        ])

        content_parts: List[Union[TextContent, AudioContent]] = []
        remaining = system_text
        while AUDIO_PLACEHOLDER_TOKEN in remaining:
            idx = remaining.find(AUDIO_PLACEHOLDER_TOKEN)
            if idx > 0:
                content_parts.append(TextContent(text=remaining[:idx]))
            content_parts.append(AudioContent(audio_url=""))
            remaining = remaining[idx + len(AUDIO_PLACEHOLDER_TOKEN):]
        if remaining:
            content_parts.append(TextContent(text=remaining))

        if len(content_parts) == 1:
            return Message(role="system", content=content_parts[0])
        return Message(role="system", content=content_parts)

    @staticmethod
    def _build_speaker_audio_pairs(
        speakers: List[Tuple[int, Speaker]],
    ) -> List[Message]:
        """Create user/assistant message pairs for speakers with reference audio."""
        pairs: List[Message] = []
        for idx, speaker in speakers:
            if speaker.audio_url:
                label = f"SPEAKER{idx}"
                pairs.append(Message(
                    role="user",
                    content=TextContent(text=f"{label}: {speaker.description}"),
                ))
                pairs.append(Message(
                    role="assistant",
                    content=AudioContent(audio_url=speaker.audio_url),
                ))
        return pairs

    # =====================================================================
    # Stage 2: Tokenization  (GenerationInput → HiggsAudioModelInput)
    # =====================================================================

    def process_inputs(self, inputs: List[GenerationInput]) -> List[HiggsAudioModelInput]:
        """Prepare and tokenize multiple GenerationInputs."""
        return [self.process_input(inp) for inp in inputs]

    def process_input(self, gen_input: GenerationInput) -> HiggsAudioModelInput:
        """Prepare and tokenize a GenerationInput into a model-ready input.

        Calls :meth:`prepare` to build the internal message list, then
        tokenizes the result.
        """
        messages = self.prepare(gen_input)
        input_tokens: List[int] = []
        label_tokens: List[int] = []
        audio_contents: List[AudioContent] = []
        start_index: Optional[int] = None  # None = predict labels for all assistant turns

        try:
            for turn_id, message in enumerate(messages):
                role_prefix_tokens = self._tokenize_role_prefix(message.role, turn_id, self.text_tokenizer)
                input_tokens.extend(role_prefix_tokens)
                label_tokens.extend([-100] * len(role_prefix_tokens))

                recipient_tokens = self._tokenize_recipient(message, self.text_tokenizer)
                if recipient_tokens:
                    input_tokens.extend(recipient_tokens)
                    label_tokens.extend(recipient_tokens)

                content_in, content_label, turn_audio = self._process_message_content(
                    message, messages, turn_id, start_index
                )
                input_tokens.extend(content_in)
                label_tokens.extend(content_label)
                audio_contents.extend(turn_audio)

                termination_tokens = self._tokenize_termination(message, messages, turn_id, self.text_tokenizer)
                input_tokens.extend(termination_tokens)
                if message.role == "assistant" and (start_index is None or turn_id >= start_index):
                    label_tokens.extend(termination_tokens)
                else:
                    label_tokens.extend([-100] * len(termination_tokens))

        except Exception as e:
            print(f"Error processing messages: {str(e)}")
            raise ValueError("Failed to process message tokens") from e

        if not input_tokens:
            raise ValueError("Messages produced no tokens")

        # 3. Audio Processing
        # Bulk process all collected audio content
        (
            audio_ids_concat,
            audio_ids_start,
            audio_waveforms_concat,
            audio_waveforms_start,
            audio_sample_rate
        ) = self._process_audio_content(audio_contents)

        # 4. Tensor Creation
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        label_ids = (
            torch.tensor(label_tokens, dtype=torch.long, device=self.device)
            if label_tokens is not None
            else torch.full_like(input_ids, -100)
        )

        # 5. Model Input Assembly
        return HiggsAudioModelInput(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=torch.tensor([], dtype=torch.long, device=self.device),
        )

    def _process_audio_content(
        self,
        audio_contents: List[AudioContent],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads, encodes, and concatenates all audio content into model tensors.
        """
        # Create empty tensors inline to avoid shared state issues
        empty_tensor = lambda dtype: torch.tensor([], dtype=dtype, device=self.device)
        
        if not audio_contents or self.audio_tokenizer is None:
            return (
                torch.tensor([[]], dtype=torch.long, device=self.device),
                empty_tensor(torch.long),
                empty_tensor(torch.float32),
                empty_tensor(torch.long),
                empty_tensor(torch.float32)
            )

        audio_ids_list: List[torch.Tensor] = []
        audio_waveforms_list: List[torch.Tensor] = []
        sample_rates: List[float] = []
        target_sr = self.audio_tokenizer.sampling_rate

        for audio_content in audio_contents:
            raw_audio = None
            sr = target_sr

            try:
                if audio_content.audio_url and audio_content.audio_url not in ["placeholder", ""]:
                    raw_audio, sr = librosa.load(audio_content.audio_url, sr=target_sr)
                elif audio_content.raw_audio:
                    decoded = base64.b64decode(audio_content.raw_audio)
                    raw_audio, sr = librosa.load(BytesIO(decoded), sr=target_sr)
            except Exception as e:
                print(f"Failed to load audio content: {e}")
                continue

            if raw_audio is not None:
                audio_waveforms_list.append(torch.tensor(raw_audio, dtype=torch.float32))
                sample_rates.append(float(sr))
                
                # Encode to tokens [Codebooks, Time]
                audio_ids = self.audio_tokenizer.encode(raw_audio, sr)
                audio_ids_list.append(audio_ids.squeeze(0).cpu())

        if not audio_ids_list:
            return (
                torch.tensor([[]], dtype=torch.long, device=self.device),
                empty_tensor(torch.long),
                empty_tensor(torch.float32),
                empty_tensor(torch.long),
                empty_tensor(torch.float32)
            )

        # Calculate Start Indices (Offsets)
        audio_ids_lengths = [x.shape[1] for x in audio_ids_list]
        audio_ids_start = torch.tensor(
            np.cumsum([0] + audio_ids_lengths)[:-1], dtype=torch.long, device=self.device
        )

        waveform_lengths = [len(x) for x in audio_waveforms_list]
        audio_waveforms_start = torch.tensor(
            np.cumsum([0] + waveform_lengths)[:-1], dtype=torch.long, device=self.device
        )

        # Concatenate
        audio_ids_concat = torch.cat(audio_ids_list, dim=1).to(self.device)
        audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0).to(self.device)
        audio_sample_rate = torch.tensor(sample_rates, dtype=torch.float32, device=self.device)

        return audio_ids_concat, audio_ids_start, audio_waveforms_concat, audio_waveforms_start, audio_sample_rate

    # ---------------------------------------------------------------------
    # Message Processing Helper Methods
    # ---------------------------------------------------------------------
    
    def _tokenize_role_prefix(self, role: str, turn_id: int, tokenizer: AutoTokenizer) -> List[int]:
        """Tokenize role prefix with special tokens (step 3 & 4: Text tokenization & Message formatting)."""
        if turn_id == 0:
            prefix = f"{BEGIN_OF_TEXT}{START_HEADER_ID}{role}{END_HEADER_ID}\n\n"
        else:
            prefix = f"{START_HEADER_ID}{role}{END_HEADER_ID}\n\n"
        return tokenizer.encode(prefix, add_special_tokens=False)

    def _tokenize_recipient(self, message: Message, tokenizer: AutoTokenizer) -> List[int]:
        """Tokenize recipient tokens if present (step 7: Recipient handling)."""
        if message.recipient and message.role == "assistant":
            recipient_text = f"{message.recipient}{RECIPIENT}"
            return tokenizer.encode(recipient_text, add_special_tokens=False)
        return []

    def _process_message_content(
        self,
        message: Message,
        messages: List[Message],
        turn_id: int,
        start_index: Optional[int] = None,
    ) -> Tuple[List[int], List[int], List[AudioContent]]:
        """Process message content: text and audio with teacher forcing."""
        input_tokens: List[int] = []
        label_tokens: List[int] = []
        audio_contents: List[AudioContent] = []

        role = message.role
        content = message.content
        
        # Normalize content to list
        content_list = []
        if isinstance(content, str):
            content_list.append(TextContent(text=content))
        elif isinstance(content, TextContent):
            content_list.append(content)
        elif isinstance(content, AudioContent):
            content_list.append(content)
        elif isinstance(content, list):
            for ele in content:
                if isinstance(ele, str):
                    content_list.append(TextContent(text=ele))
                else:
                    content_list.append(ele)
        
        # Process each content item
        for content_item in content_list:
            if content_item.type == "text":
                # Tokenize text content
                text_tokens = self.text_tokenizer.encode(content_item.text, add_special_tokens=False)
                input_tokens.extend(text_tokens)
                
                # Apply teacher forcing with start_index handling
                if role == "assistant" and (start_index is None or turn_id >= start_index):
                    label_tokens.extend(text_tokens)
                else:
                    label_tokens.extend([-100] * len(text_tokens))
            
            elif content_item.type == "audio":
                # Collect audio content for later processing
                audio_contents.append(content_item)
                
                # Tokenize audio placeholder tokens
                if role == "user" or role == "system":
                    audio_tokens = self.text_tokenizer.encode(
                        f"{AUDIO_BOS}{AUDIO_IN_TOKEN}{AUDIO_EOS}", add_special_tokens=False
                    )
                else:  # assistant
                    audio_tokens = self.text_tokenizer.encode(
                        f"{AUDIO_OUT_BOS}{AUDIO_OUT_TOKEN}{AUDIO_EOS}", add_special_tokens=False
                    )
                
                input_tokens.extend(audio_tokens)
                
                # Apply teacher forcing for audio tokens
                if role == "assistant" and (start_index is None or turn_id >= start_index):
                    label_tokens.extend(audio_tokens)
                else:
                    label_tokens.extend([-100] * len(audio_tokens))
        
        return input_tokens, label_tokens, audio_contents

    def _tokenize_termination(
        self,
        message: Message,
        messages: List[Message],
        turn_id: int,
        tokenizer: AutoTokenizer,
    ) -> List[int]:
        """Tokenize termination tokens."""
        next_id = turn_id + 1
        if message.role == "assistant" and next_id < len(messages) and messages[next_id].role == "assistant":
            termination_text = EOM_ID
        else:
            termination_text = EOT_ID
        return tokenizer.encode(termination_text, add_special_tokens=False)
