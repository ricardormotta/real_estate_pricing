"""
This is a boilerplate pipeline 'model_metrics'
generated using Kedro 0.18.4
"""
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


def metrics_node(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    metrics = {
        "MAPE_test": {"value": MAPE(y_test, y_pred), "step": 1},
        "MAPE_train": {"value": MAPE(y_train, y_pred_train), "step": 1},
        "MAE_test": {"value": MAE(y_test, y_pred), "step": 1},
        "MAE_train": {"value": MAE(y_train, y_pred_train), "step": 1},
        "MSE_test": {"value": MSE(y_test, y_pred), "step": 1},
        "MSE_train": {"value": MSE(y_train, y_pred_train), "step": 1},
        "RMSE_test": {"value": MSE(y_test, y_pred, squared=False), "step": 1},
        "RMSE_train": {"value": MSE(y_train, y_pred_train, squared=False), "step": 1},
        "r2_score_test": {"value": r2_score(y_test, y_pred), "step": 1},
        "r2_score_train": {"value": r2_score(y_train, y_pred_train), "step": 1},
    }
    return metrics
