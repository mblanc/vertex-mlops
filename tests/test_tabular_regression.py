import pytest

from src.pipelines.tabular_classification.bqml import pipeline

def test_main_succeeds(tmp_path):
    pipeline.compile(str(tmp_path / "pipeline.json"))
    assert True
