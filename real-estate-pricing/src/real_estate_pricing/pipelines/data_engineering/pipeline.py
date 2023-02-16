"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_cleaning


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_cleaning,
                inputs=[
                    "raw_data",
                    "params:accepted_cities",
                    "params:categorical_features",
                    "params:numerical_features",
                    "params:bigger_than_zero_cols",
                    "params:target",
                    "params:rental_transaction_type",
                    "params:sales_transaction_type",
                ],
                outputs=["df_rental", "df_sales"],
            )
        ]
    )
