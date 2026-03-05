from dataclasses import dataclass
from typing import Optional, Union

from .message_content import AudioContent, TextContent
from .speaker import Speaker

@dataclass
class Message:
    role: str
    content: Union[AudioContent, TextContent]
    recipient: Optional[str] = None
    speaker: Optional[Speaker] = None