"""
This is a boilerplate pipeline 'fastapi'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import save_predictor


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=save_predictor,
                inputs=[
                    "fitted_rental_pipe",
                    "fitted_sales_pipe",
                    "params:categorical_features",
                ],
                outputs="MLPredictor",
            )
        ]
    )
