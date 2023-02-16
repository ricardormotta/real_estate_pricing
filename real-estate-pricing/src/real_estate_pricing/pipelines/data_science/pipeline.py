"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_test, create_data_pipeline, fit_grid_search_cv_on_pipe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_train_test,
                inputs=[
                    "df_rental",
                    "params:perc_test",
                    "params:numerical_features",
                    "params:categorical_features",
                    "params:target",
                    "params:SEED",
                ],
                outputs=[
                    "X_train_rental",
                    "X_test_rental",
                    "y_train_rental",
                    "y_test_rental",
                ],
                name="split_train_test_rental",
                tags=["rental_pipeline", "fit_models"],
            ),
            node(
                func=create_data_pipeline,
                inputs=[
                    "params:numerical_features",
                    "params:categorical_features",
                    "params:SEED",
                ],
                outputs="rental_pipeline",
                name="create_data_pipeline_rental",
                tags="rental_pipeline",
            ),
            node(
                func=fit_grid_search_cv_on_pipe,
                inputs=[
                    "X_train_rental",
                    "y_train_rental",
                    "X_test_rental",
                    "y_test_rental",
                    "rental_pipeline",
                    "params:params_grid",
                ],
                outputs="fitted_rental_pipeline",
                name="fit_grid_search_cv_on_pipe_rental",
                tags=["rental_pipeline", "fit_models"],
            ),
            node(
                func=split_train_test,
                inputs=[
                    "df_sales",
                    "params:perc_test",
                    "params:numerical_features",
                    "params:categorical_features",
                    "params:target",
                    "params:SEED",
                ],
                outputs=[
                    "X_train_sales",
                    "X_test_sales",
                    "y_train_sales",
                    "y_test_sales",
                ],
                name="split_train_test_sales",
                tags=["sales_pipeline", "fit_models"],
            ),
            node(
                func=create_data_pipeline,
                inputs=[
                    "params:numerical_features",
                    "params:categorical_features",
                    "params:SEED",
                ],
                outputs="sales_pipeline",
                name="create_data_pipeline_sales",
                tags="sales_pipeline",
            ),
            node(
                func=fit_grid_search_cv_on_pipe,
                inputs=[
                    "X_train_sales",
                    "y_train_sales",
                    "X_test_sales",
                    "y_test_sales",
                    "sales_pipeline",
                    "params:params_grid",
                ],
                outputs="fitted_sales_pipeline",
                name="fit_grid_search_cv_on_pipe_sales",
                tags=["sales_pipeline", "fit_models"],
            ),
        ]
    )
