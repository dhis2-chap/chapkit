from typing import Any, Awaitable, Callable, TypeVar

import pandas as pd
from geojson_pydantic import FeatureCollection

from chapkit.types import ChapConfig


class ChapModelConfig(ChapConfig):
    pass


TChapModelConfig = TypeVar("TChapModelConfig", bound=ChapModelConfig)

OnTrainCallable = Callable[[TChapModelConfig, pd.DataFrame, FeatureCollection | None], Awaitable[Any]]
OnPredictCallable = Callable[[TChapModelConfig, Any, pd.DataFrame, pd.DataFrame], Awaitable[pd.DataFrame]]
