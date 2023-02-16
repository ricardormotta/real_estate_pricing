"""
This is a boilerplate pipeline 'model_metrics'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import metrics_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=metrics_node,
                inputs=[
                    "fitted_rental_pipeline",
                    "X_train_rental",
                    "X_test_rental",
                    "y_train_rental",
                    "y_test_rental",
                ],
                outputs="metrics_rent",
                name="metrics_rent",
            ),
            node(
                func=metrics_node,
                inputs=[
                    "fitted_sales_pipeline",
                    "X_train_sales",
                    "X_test_sales",
                    "y_train_sales",
                    "y_test_sales",
                ],
                outputs="metrics_sales",
                name="metrics_sales",
            ),
        ]
    )
