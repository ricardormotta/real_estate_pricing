import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def CustomSampler_IQR (X, features):
    
    bool_filter = np.ones(len(X)).astype(bool)
    for col in features:
        #Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(X[col], 25.)
        Q3 = np.nanpercentile(X[col], 75.)
        
        cut_off = (Q3 - Q1) * 1.5
        upper = Q3 + cut_off
        lower = Q1 - cut_off

        outliers = (
            (X[col] >= lower)
            & (X[col] <= upper)
        ).to_numpy()
        bool_filter = bool_filter * outliers

    return bool_filter.astype(bool)

def prepare_data_to_models(df):
    perc_test = 0.2
    X = df[[*num_cols, *cat_cols]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=perc_test)
    outlier_removal_index = CustomSampler_IQR (X_train, num_cols)
    X_train = X_train.loc[outlier_removal_index]
    y_train = y_train.loc[outlier_removal_index]

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    data_pipeline = ColumnTransformer([
        ('numerical',  num_pipeline, num_cols),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist'), cat_cols),
    ])


    X_train = data_pipeline.fit_transform(X_train)
    X_test = data_pipeline.transform(X_test)
    return (X_train, X_test, y_train, y_test)

def train_models(df, SEED:int=42):
    len_df = len(df_aluguel)
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data_to_models(df)

    np.random.seed(SEED)

    models = [
        DecisionTreeRegressor(),
        LinearRegression(),
        XGBRegressor(t= round(len_df/10), max_depth=10,),
        # SVR(),
        RandomForestRegressor(n_estimators= 2000),
    ]

    results = {
        "Model": [],
        "MAPE": [],
        "MAE": [],
        "MSE": [],
        "RMSE": [],
    }
    from tqdm import tqdm
    for model in tqdm(models):
        results['Model'].append(type(model).__name__)
        if type(model).__name__ == 'XGBRegressor':
            eval_set = [
                (X_train, y_train),
                (X_test, y_test)
            ]
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
                eval_metric='mape',
                early_stopping_rounds=500
            )
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results["MAPE"].append(MAPE(y_test, y_pred))
        results["MAE"].append(MAE(y_test, y_pred))
        results["MSE"].append(MSE(y_test, y_pred))
        results["RMSE"].append(MSE(y_test, y_pred, squared=False))
    results = pd.DataFrame.from_dict(results)
    return models,results

df = pd.read_csv("./data/amostra_sp.txt", sep="|")
df = df.loc[df['shp_municipio'] == "São Paulo"]

cat_cols = [
    "shp_municipio",
    "shp_bairro_distrito",
    "shp_microarea",
    "tipo_imovel",
]
num_cols = [
    'area_util',
    'dormitorios',
    'suites',
    'banheiros',
    'vagas',
    'salas',
]
target = "preco_imovel_mediana"

cols_to_keep = [*cat_cols, *num_cols, target]
df = df.dropna(subset=cols_to_keep)

# df_venda = df.loc[df['tipo_transacao']=='VENDA'][cols_to_keep]
df_aluguel = df.loc[df['tipo_transacao']=='LOCACAO'][cols_to_keep]
del df


models_aluguel, results_aluguel = train_models(df_aluguel)

results_aluguel.to_csv('results_aluguel.csv', sep=',')
print(50*"=")
print(20*" ", "Acabou!")
print(50*"=")