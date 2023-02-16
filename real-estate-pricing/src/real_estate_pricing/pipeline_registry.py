"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from real_estate_pricing.pipelines import data_engineering as de
from real_estate_pricing.pipelines import data_science as ds
from real_estate_pricing.pipelines import model_metrics as mm
from real_estate_pricing.pipelines import fastapi as fa


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_engineering = de.create_pipeline()
    data_science = ds.create_pipeline()
    model_metrics = mm.create_pipeline()
    fastapi = fa.create_pipeline()
    pipelines = find_pipelines()
    pipelines["de"] = data_engineering
    pipelines["ds"] = data_science
    pipelines["model_metrics"] = model_metrics
    pipelines["fastapi"] = fastapi
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
