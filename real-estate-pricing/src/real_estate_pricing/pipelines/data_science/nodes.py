"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from xgboost import XGBRegressor
from scipy import sparse


def split_train_test(
    df, perc_test, numerical_features, categorical_features, target, SEED
):
    np.random.seed(SEED)
    X = df[[*numerical_features, *categorical_features]]
    y = df[target]
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=perc_test)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    for col in numerical_features:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        X_train = X_train.loc[(X_train[col] > fence_low) & (X_train[col] < fence_high)]
    y_train = y_train.iloc[X_train.index]

    return (X_train, X_test, y_train, y_test)


def create_data_pipeline(numerical_features, categorical_features, SEED):
    np.random.seed(SEED)
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    CT = ColumnTransformer(
        [
            ("numerical", numeric_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    pipe = Pipeline(
        [
            ("ColumnTransformer", CT),
        ]
    )

    return pipe


def fit_grid_search_cv_on_pipe(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    pipe: Pipeline,
    params_grid: dict,
):
    model = XGBRegressor(early_stopping_rounds=200)

    X_train = sparse.csr_matrix(pipe.fit_transform(X_train))
    X_test = sparse.csr_matrix(pipe.transform(X_test))
    eval_set = [(X_train, y_train), (X_test, y_test)]

    search = RandomizedSearchCV(model, params_grid, n_jobs=2, cv=5)
    search.fit(X_train, y_train.to_numpy(), eval_set=eval_set, verbose=False)
    pipe = Pipeline([("CT", pipe), ("trained_CV", search)])
    return pipe
