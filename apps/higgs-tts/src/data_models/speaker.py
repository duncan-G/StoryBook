from dataclasses import dataclass, field
from typing import Optional
import uuid as _uuid


@dataclass
class Speaker:
    description: str
    audio_url: Optional[str] = None
    uuid: _uuid.UUID = field(default_factory=_uuid.uuid4)
