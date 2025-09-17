import pandas as pd
import pytest
from chapkit.database import SqlAlchemyChapDatabase
from chapkit.model.runner import FunctionalChapModelRunner
from chapkit.model.types import ChapModelConfig
from chapkit.types import ChapServiceInfo, PredictParams, TrainParams, PredictData, TrainData, DataFrameSplit


def sync_on_train(config: ChapModelConfig, data: pd.DataFrame, geo=None):
    return {"sync_train": True}


async def async_on_train(config: ChapModelConfig, data: pd.DataFrame, geo=None):
    return {"async_train": True}


def sync_on_predict(config: ChapModelConfig, model: any, historic: pd.DataFrame, future: pd.DataFrame, geo=None):
    return pd.DataFrame([{"sync_predict": True}])


async def async_on_predict(config: ChapModelConfig, model: any, historic: pd.DataFrame, future: pd.DataFrame, geo=None):
    return pd.DataFrame([{"async_predict": True}])


@pytest.fixture
def runner_components():
    info = ChapServiceInfo(display_name="Test Runner")
    database = SqlAlchemyChapDatabase(config_type=ChapModelConfig)
    return info, database


@pytest.mark.asyncio
async def test_functional_runner_with_sync_functions(runner_components):
    info, database = runner_components
    runner = FunctionalChapModelRunner(
        info,
        database,
        config_type=ChapModelConfig,
        on_train=sync_on_train,
        on_predict=sync_on_predict,
    )

    config = ChapModelConfig(name="test")
    database.add_config(config)
    train_params = TrainParams(config=config, body=TrainData(data=pd.DataFrame()))
    predict_params = PredictParams(
        config=config,
        artifact_id=None,
        artifact=None,
        body=PredictData(historic=pd.DataFrame(), future=pd.DataFrame()),
    )

    train_artifact_id = await runner.on_train(train_params)
    predict_artifact_id = await runner.on_predict(predict_params)

    trained_model = database.get_artifact(train_artifact_id)
    assert trained_model == {"sync_train": True}

    prediction_result = database.get_artifact(predict_artifact_id)
    expected_df = pd.DataFrame([{"sync_predict": True}])
    pd.testing.assert_frame_equal(DataFrameSplit.to_pandas(prediction_result), expected_df)


@pytest.mark.asyncio
async def test_functional_runner_with_async_functions(runner_components):
    info, database = runner_components
    runner = FunctionalChapModelRunner(
        info,
        database,
        config_type=ChapModelConfig,
        on_train=async_on_train,
        on_predict=async_on_predict,
    )

    config = ChapModelConfig(name="test")
    database.add_config(config)
    train_params = TrainParams(config=config, body=TrainData(data=pd.DataFrame()))
    predict_params = PredictParams(
        config=config,
        artifact_id=None,
        artifact=None,
        body=PredictData(historic=pd.DataFrame(), future=pd.DataFrame()),
    )

    train_artifact_id = await runner.on_train(train_params)
    predict_artifact_id = await runner.on_predict(predict_params)

    trained_model = database.get_artifact(train_artifact_id)
    assert trained_model == {"async_train": True}

    prediction_result = database.get_artifact(predict_artifact_id)
    expected_df = pd.DataFrame([{"async_predict": True}])
    pd.testing.assert_frame_equal(DataFrameSplit.to_pandas(prediction_result), expected_df)
