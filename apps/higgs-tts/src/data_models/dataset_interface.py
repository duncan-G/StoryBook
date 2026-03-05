from abc import ABC, abstractmethod
from typing import Union, Optional
from dataclasses import dataclass

from .model_input import HiggsAudioModelInput, RankedHiggsAudioModelInputTuple


class DatasetInterface(ABC):
    @abstractmethod
    def __getitem__(self, idx) -> Union[HiggsAudioModelInput, RankedHiggsAudioModelInputTuple]:
        """Retrieve a dataset sample by index."""
        raise NotImplementedError


class IterableDatasetInterface(ABC):
    @abstractmethod
    def __iter__(self) -> Union[HiggsAudioModelInput, RankedHiggsAudioModelInputTuple]:
        """Retrieve a sample by iterating through the dataset."""
        raise NotImplementedError


@dataclass
class DatasetInfo:
    dataset_type: str
    group_type: Optional[str] = None
    mask_text: Optional[bool] = None  # Whether to mask the text tokens for pretraining samples.
