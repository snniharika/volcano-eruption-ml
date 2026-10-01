from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans


class KMeansPrototypeClassifier(BaseEstimator, ClassifierMixin):
    """Paper-inspired classifier using 8 K-means prototypes per class."""

    def __init__(self, n_clusters_per_class=8, random_state=42, n_init=20):
        self.n_clusters_per_class = n_clusters_per_class
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.models_ = {}

        for cls in self.classes_:
            km = KMeans(
                n_clusters=self.n_clusters_per_class,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            km.fit(X[y == cls])
            self.models_[int(cls)] = km
        return self

    def _distances(self, X):
        d0 = self.models_[0].transform(X).min(axis=1)
        d1 = self.models_[1].transform(X).min(axis=1)
        return d0, d1

    def predict_proba(self, X):
        d0, d1 = self._distances(np.asarray(X, dtype=float))
        score = np.clip(d0 - d1, -50, 50)
        p1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class KMeansTimeRegressor(BaseEstimator, RegressorMixin):
    """Paper-inspired K-means regression using 10 clusters in X + y space."""

    def __init__(self, n_clusters=10, random_state=42, n_init=20):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.kmeans_.fit(np.column_stack([X, y]))
        self.feature_centers_ = self.kmeans_.cluster_centers_[:, :-1]
        self.target_centers_ = self.kmeans_.cluster_centers_[:, -1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = (
            (X[:, None, :] - self.feature_centers_[None, :, :]) ** 2
        ).sum(axis=2)
        nearest = distances.argmin(axis=1)
        return self.target_centers_[nearest]
