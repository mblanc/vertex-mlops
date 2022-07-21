import pathlib

from src.pipelines.forecasting.automl.pipeline import TabularForecastingAutoMLPipeline
from src.pipelines.forecasting.bqml.pipeline import TabularForecastingBQMLPipeline
from src.pipelines.tabular_classification.automl.pipeline import (
    TabularClassificationAutoMLPipeline,
)
from src.pipelines.tabular_classification.automl_evaluation.pipeline import (
    TabularClassificationAutoMLEvaluationPipeline,
)
from src.pipelines.tabular_classification.bqml.pipeline import (
    TabularClassificationBQMLPipeline,
)
from src.pipelines.tabular_classification.custom.pipeline import (
    TabularClassificationCustomPipeline,
)
from src.pipelines.tabular_regression.automl.pipeline import (
    TabularRegressionAutoMLPipeline,
)
from src.pipelines.tabular_regression.bqml.pipeline import TabularRegressionBQMLPipeline


def test_forecasting_automl_compile(tmp_path: pathlib.Path) -> None:
    TabularForecastingAutoMLPipeline().compile_pipeline(str(tmp_path / "pipeline.json"))
    assert True


def test_forecasting_bqml_compile(tmp_path: pathlib.Path) -> None:
    TabularForecastingBQMLPipeline().compile_pipeline(str(tmp_path / "pipeline.json"))
    assert True


def test_classification_automl_compile(tmp_path: pathlib.Path) -> None:
    TabularClassificationAutoMLPipeline().compile_pipeline(
        str(tmp_path / "pipeline.json")
    )
    assert True


def test_classification_automl_evaluation_compile(tmp_path: pathlib.Path) -> None:
    TabularClassificationAutoMLEvaluationPipeline().compile_pipeline(
        str(tmp_path / "pipeline.json")
    )
    assert True


def test_classification_bqml_compile(tmp_path: pathlib.Path) -> None:
    TabularClassificationBQMLPipeline().compile_pipeline(
        str(tmp_path / "pipeline.json")
    )
    assert True


def test_classification_custom_compile(tmp_path: pathlib.Path) -> None:
    TabularClassificationCustomPipeline().compile_pipeline(
        str(tmp_path / "pipeline.json")
    )
    assert True


def test_regression_automl_compile(tmp_path: pathlib.Path) -> None:
    TabularRegressionAutoMLPipeline().compile_pipeline(str(tmp_path / "pipeline.json"))
    assert True


def test_regression_bqml_compile(tmp_path: pathlib.Path) -> None:
    TabularRegressionBQMLPipeline().compile_pipeline(str(tmp_path / "pipeline.json"))
    assert True
