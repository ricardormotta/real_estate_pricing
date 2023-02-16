"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.4
"""
import pandas as pd


def data_cleaning(
    df: pd.DataFrame,
    accepted_cities: list,
    categorical_features,
    numerical_features,
    bigger_than_zero_cols,
    target,
    rental_transaction_type,
    sales_transaction_type,
):
    df = df.loc[df["Mes"]>=9]
    df = df.loc[df["shp_municipio"].isin(accepted_cities)]
    cols_to_keep = [*categorical_features, *numerical_features, target]
    df = df.dropna(subset=cols_to_keep)
    for col in bigger_than_zero_cols:
        df = df.loc[df[col] > 0]

    df_rental = df.loc[(df["tipo_transacao"] == rental_transaction_type), cols_to_keep]
    df_sales = df.loc[(df["tipo_transacao"] == sales_transaction_type), cols_to_keep]
    del df

    return df_rental, df_sales
