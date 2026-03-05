from dataclasses import dataclass, field
from typing import List, Optional
import uuid

from .message import Message


@dataclass
class GenerationInput:
    """Client-facing input for audio generation.

    Contains an array of messages (each with text content and a speaker)
    and optional overrides.  When ``system_prompt`` is provided the
    server uses it verbatim; otherwise it builds one from
    ``scene_description`` and the speakers found in ``messages``.
    """
    messages: List[Message]
    system_prompt: Optional[str] = None
    scene_description: Optional[str] = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
