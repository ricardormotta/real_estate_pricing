"""
This is a boilerplate pipeline 'fastapi'
generated using Kedro 0.18.4
"""
import pandas as pd
import numpy as np
import sys
class MLPredictor:
    def __init__(self, model_rent, model_sales, cat_cols):
        self.model_rent = model_rent
        self.model_sales = model_sales
        self.cat_cols = cat_cols

    def predict(self, args_API: pd.DataFrame, context):
        # sys.breakpointhook()
        for col in self.cat_cols:
            args_API.loc[:,col] = args_API.loc[:,col][0].value
            args_API.loc[:,col] = args_API.loc[:,col].replace({"_", " "})
        df_args = args_API
        rent_prediction = self.model_rent.predict(df_args).tolist()
        sales_prediction = self.model_sales.predict(df_args).tolist()
        print("\n", "="*100)
        print(rent_prediction, sales_prediction)
        print("\n", "="*100)
        return {
            "rent_prediction": rent_prediction,
            "sales_prediction": sales_prediction,
        }


def save_predictor(model_rent, model_sales, categorical_features):
    predictor = MLPredictor(model_rent, model_sales, categorical_features)
    return predictor
