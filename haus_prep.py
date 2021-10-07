from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AlgebricImputer(BaseEstimator,TransformerMixin):
    def __init__(self, features, refvariable):
        if not isinstance(features, list):
            raise TypeError('features must be a list')
        self.features = features
        self.refvariable = refvariable

    def fit(self,X,y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for var in self.features:
            X[var] = X[self.refvariable] - X[var]
        return X

class Mapping(BaseEstimator,TransformerMixin):
    def __init__(self, features, mapdict):
        if not isinstance(features, list):
            raise TypeError('features must be a list')
        self.features = features
        self.mapdict = mapdict
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for var in self.features:
            X[var] = X[var].map(self.mapdict)
        return X
    
class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if not isinstance(features, list):
            raise TypeError('features must be a list')
        self.features = features
    def fit(self,X, y=None):
        self.param_dict_ = X[self.features].mode().to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        for var in self.features:
            X[var] = X[var].fillna(self.param_dict_[var])
        return X

