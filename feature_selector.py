import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class FeatureSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = {}

    def correlation_filter(self, threshold=0.8):
        corr_matrix = self.X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        self.results['correlation_filter'] = [col for col in self.X.columns if col not in to_drop]
        return self.results['correlation_filter']

    def mutual_info(self):
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        mi_series = pd.Series(mi_scores, index=self.X.columns).sort_values(ascending=False)
        self.results['mutual_info'] = list(mi_series.index)
        return mi_series

    def rfe(self, estimator=None, n_features_to_select=5):
        if estimator is None:
            estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector.fit(self.X, self.y)
        self.results['rfe'] = list(self.X.columns[selector.support_])
        return self.results['rfe']

    def tree_importance(self, model=None):
        if model is None:
            model = RandomForestClassifier(random_state=42)
        model.fit(self.X, self.y)
        importances = pd.Series(model.feature_importances_, index=self.X.columns).sort_values(ascending=False)
        self.results['tree_importance'] = list(importances.index)
        return importances