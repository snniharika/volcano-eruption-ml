from __future__ import annotations

import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansPrototypeClassifier(
    BaseEstimator,
    ClassifierMixin,
):
    """
    Paper-inspired K-Means classifier.

    The training data is divided into:
        class 0 = repose
        class 1 = erupting

    Each class receives 8 clusters.
    """

    def __init__(
        self,
        n_clusters_per_class=8,
        random_state=42,
        n_init=20,
    ):
        self.n_clusters_per_class = (
            n_clusters_per_class
        )

        self.random_state = random_state

        self.n_init = n_init

    def fit(self, X, y):
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=int,
        )

        self.classes_ = np.array(
            [0, 1]
        )

        self.models_ = {}

        for class_value in self.classes_:

            class_data = X[
                y == class_value
            ]

            model = KMeans(
                n_clusters=(
                    self.n_clusters_per_class
                ),
                random_state=(
                    self.random_state
                ),
                n_init=self.n_init,
            )

            model.fit(
                class_data
            )

            self.models_[
                int(class_value)
            ] = model

        return self

    def _class_distances(self, X):

        X = np.asarray(
            X,
            dtype=float,
        )

        distances_to_repose = (
            self.models_[0]
            .transform(X)
            .min(axis=1)
        )

        distances_to_erupting = (
            self.models_[1]
            .transform(X)
            .min(axis=1)
        )

        return (
            distances_to_repose,
            distances_to_erupting,
        )

    def decision_function(self, X):

        d0, d1 = (
            self._class_distances(X)
        )

        # Larger values mean stronger evidence
        # for the erupting class.
        return d0 - d1

    def predict_proba(self, X):

        decision = (
            self.decision_function(X)
        )

        decision = np.clip(
            decision,
            -50,
            50,
        )

        probability_erupting = (
            1.0
            / (
                1.0
                + np.exp(-decision)
            )
        )

        return np.column_stack(
            [
                1.0
                - probability_erupting,
                probability_erupting,
            ]
        )

    def predict(self, X):

        probabilities = (
            self.predict_proba(X)
        )

        return (
            probabilities[:, 1]
            >= 0.5
        ).astype(int)


class KMeansTimeRegressor(
    BaseEstimator,
    RegressorMixin,
):
    """
    Paper-inspired K-Means regression.

    The paper uses 10 clusters.

    We scale both:
        - input features
        - time-to-eruption target

    because the paper explicitly states that both
    features and time-to-eruption are scaled.
    """

    def __init__(
        self,
        n_clusters=10,
        random_state=42,
        n_init=20,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        self.target_scaler_ = (
            StandardScaler()
        )

        y_scaled = (
            self.target_scaler_
            .fit_transform(
                y.reshape(-1, 1)
            )
            .ravel()
        )

        combined_data = np.column_stack(
            [
                X,
                y_scaled,
            ]
        )

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.kmeans_.fit(
            combined_data
        )

        self.feature_centers_ = (
            self.kmeans_
            .cluster_centers_[:, :-1]
        )

        self.target_centers_scaled_ = (
            self.kmeans_
            .cluster_centers_[:, -1]
        )

        return self

    def predict(self, X):

        X = np.asarray(
            X,
            dtype=float,
        )

        distances = (
            (
                X[:, None, :]
                - self.feature_centers_[
                    None, :, :
                ]
            )
            ** 2
        ).sum(axis=2)

        nearest_clusters = (
            distances.argmin(axis=1)
        )

        predictions_scaled = (
            self.target_centers_scaled_[
                nearest_clusters
            ]
        )

        predictions = (
            self.target_scaler_
            .inverse_transform(
                predictions_scaled.reshape(
                    -1,
                    1,
                )
            )
            .ravel()
        )

        return predictions
