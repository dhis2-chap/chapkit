from typing import TypeVar

from chapkit.types import ChapConfig


class ChapModelConfig(ChapConfig):
    pass


TChapModelConfig = TypeVar("TChapModelConfig", bound=ChapModelConfig)
