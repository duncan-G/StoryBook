from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioContent:
    audio_url: str
    raw_audio: Optional[str] = None  # Base64-encoded audio bytes
    type: str = "audio"

@dataclass
class TextContent:
    text: str
    type: str = "text"