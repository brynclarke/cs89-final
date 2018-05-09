import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, QuantileTransformer, StandardScaler
from scipy.special import erfinv

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self._reshape = (X.ndim == 1)
        pass

    def transform(self, X):
        if self._reshape:
            X_ = X.reshape(-1, 1)
        else:
            X_ = X
            
        return X_

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class RankGaussTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self._n = len(X)
        self.qt = QuantileTransformer(subsample=self._n)
        self._reshape = (X.ndim == 1)
        
        if self._reshape:
            X_ = X.reshape(-1, 1)
        else:
            X_ = X
            
        self.qt.fit(X_)
        return self

    def transform(self, X):
        if self._reshape:
            X_ = X.reshape(-1, 1)
        else:
            X_ = X
        
        X_unif = (self.qt.transform(X_)*(self._n-1)+1)/(self._n+1)
        
        X_norm = erfinv(2*X_unif-1)
        
        return X_norm

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rgt = RankGaussTransformer()
        self.sct = StandardScaler()
    
    def fit(self, X):
        self.rgt.fit(X)
        X_rg = self.rgt.transform(X)
        self.sct.fit(X_rg)
    
    def transform(self, X):
        X_rg = self.rgt.transform(X)
        return self.sct.transform(X_rg)
    
    def fit_transform(self, X):
        X_rg = self.rgt.fit_transform(X)
        return self.sct.fit_transform(X_rg)

class OneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        self.lb.fit(X)
        self.classes_ = self.lb.classes_

    def transform(self, X):
        Xlb = self.lb.transform(X)
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nmt = NumericalTransformer()
        self.oht = OneHotTransformer()
    
    def fit(self, X):
        self.nmt.fit(X)
        self.oht.fit(X)
    
    def transform(self, X):
        return np.hstack([
            self.nmt.transform(X),
            self.oht.transform(X)
        ])
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class PortoSeguroTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_types):
        self.transformers = []
        for ftype in feature_types:
            if ftype["type"] == "cat":
                tform = CategoricalTransformer()
            elif ftype["type"] == "bin":
                tform = IdentityTransformer()
            elif ftype["type"] == "num":
                tform = NumericalTransformer()
            
            self.transformers += [{
                "feature": ftype["name"],
                "transformer": tform
            }]
    
    def fit(self, df):
        for tform in self.transformers:
            tform["transformer"].fit(
                df[tform["feature"]].values
            )
    
    def transform(self, df):
        transformed = []
        for tform in self.transformers:
            transformed += [
                tform["transformer"].transform(
                    df[tform["feature"]].values
                )                
            ]
        return np.hstack(transformed)
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)