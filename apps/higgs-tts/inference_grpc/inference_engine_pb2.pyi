from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "max_tokens", "seed", "ras_win_len", "ras_win_max_num_repeat", "force_audio_gen")
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    RAS_WIN_LEN_FIELD_NUMBER: _ClassVar[int]
    RAS_WIN_MAX_NUM_REPEAT_FIELD_NUMBER: _ClassVar[int]
    FORCE_AUDIO_GEN_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int
    ras_win_len: int
    ras_win_max_num_repeat: int
    force_audio_gen: bool
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., max_tokens: _Optional[int] = ..., seed: _Optional[int] = ..., ras_win_len: _Optional[int] = ..., ras_win_max_num_repeat: _Optional[int] = ..., force_audio_gen: bool = ...) -> None: ...

class Speaker(_message.Message):
    __slots__ = ("uuid", "description", "audio_url")
    UUID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    uuid: str
    description: str
    audio_url: str
    def __init__(self, uuid: _Optional[str] = ..., description: _Optional[str] = ..., audio_url: _Optional[str] = ...) -> None: ...

class InputMessage(_message.Message):
    __slots__ = ("text", "speaker")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    text: str
    speaker: Speaker
    def __init__(self, text: _Optional[str] = ..., speaker: _Optional[_Union[Speaker, _Mapping]] = ...) -> None: ...

class GenerationInput(_message.Message):
    __slots__ = ("messages", "system_prompt", "scene_description")
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    SCENE_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[InputMessage]
    system_prompt: str
    scene_description: str
    def __init__(self, messages: _Optional[_Iterable[_Union[InputMessage, _Mapping]]] = ..., system_prompt: _Optional[str] = ..., scene_description: _Optional[str] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "inputs", "sampling_params", "stream")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    inputs: _containers.RepeatedCompositeFieldContainer[GenerationInput]
    sampling_params: SamplingParams
    stream: bool
    def __init__(self, request_id: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[GenerationInput, _Mapping]]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., stream: bool = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("chunk", "complete")
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    chunk: GenerateStreamChunk
    complete: GenerateComplete
    def __init__(self, chunk: _Optional[_Union[GenerateStreamChunk, _Mapping]] = ..., complete: _Optional[_Union[GenerateComplete, _Mapping]] = ...) -> None: ...

class GenerateStreamChunk(_message.Message):
    __slots__ = ("token_ids", "prompt_tokens", "completion_tokens", "audio_data", "sampling_rate")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    prompt_tokens: int
    completion_tokens: int
    audio_data: bytes
    sampling_rate: int
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., audio_data: _Optional[bytes] = ..., sampling_rate: _Optional[int] = ...) -> None: ...

class GenerateComplete(_message.Message):
    __slots__ = ("output_ids", "finish_reason", "prompt_tokens", "completion_tokens", "audio_data", "sampling_rate")
    OUTPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    output_ids: _containers.RepeatedScalarFieldContainer[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    audio_data: bytes
    sampling_rate: int
    def __init__(self, output_ids: _Optional[_Iterable[int]] = ..., finish_reason: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., audio_data: _Optional[bytes] = ..., sampling_rate: _Optional[int] = ...) -> None: ...
