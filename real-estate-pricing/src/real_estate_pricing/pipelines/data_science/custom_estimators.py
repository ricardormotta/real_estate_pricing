import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class IQRFilter(BaseEstimator, TransformerMixin):
    def __init__(self, factor=2):
        self.factor = factor

    def outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.NaN
            X.iloc[:, i] = x
        return X


from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor


class OutlierExtractor(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.threshold = kwargs.pop("neg_conf_val", -10.0)

        self.kwargs = kwargs

    def transform(self, X, y=None):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        z = np.abs(stats.zscore(X))
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (
            X[lcf.negative_outlier_factor_ > self.threshold, :],
            y[lcf.negative_outlier_factor_ > self.threshold],
        )

    def fit(self, *args, **kwargs):
        return self


class RemoveIQROutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in X_.columns:
            X_.fillna(X[col].median(), inplace=True)
        return X_


from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split

# class XGBRegressor_Custom_Fit(XGBRegressor):

#     def fit(self, X_train, y_train, *, eval_test_size=200, **kwargs):

#         # if eval_test_size is not None:

#             # params = super(XGBRegressor, self).get_xgb_params()

#             # eval_set = [(X_train, y_train), (X_test, y_test)]

#             # kwargs['eval_set'] = eval_set
#             # kwargs['early_stopping_rounds'] = 200

#         return super(XGBRegressor_Custom_Fit, self).fit(X_train, y_train, **kwargs)
