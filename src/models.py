from __future__ import annotations

import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)

from sklearn.cluster import KMeans


class KMeansPrototypeClassifier(
    BaseEstimator,
    ClassifierMixin,
):
    """
    Paper-inspired K-Means classifier.

    The classifier is applied to the new daily-window
    forecasting task. Training data is divided into:

        class 0 = no eruption start in next 7 days
        class 1 = eruption start in next 7 days

    Eight clusters are fitted separately for each class.
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

        distances_to_no_eruption = (
            self.models_[0]
            .transform(X)
            .min(axis=1)
        )

        distances_to_eruption = (
            self.models_[1]
            .transform(X)
            .min(axis=1)
        )

        return (
            distances_to_no_eruption,
            distances_to_eruption,
        )

    def decision_function(self, X):

        distance_0, distance_1 = (
            self._class_distances(X)
        )

        # Positive values indicate that the observation
        # is closer to an eruption prototype than to a
        # no-eruption prototype.
        return (
            distance_0
            - distance_1
        )

    def predict_proba(self, X):

        decision = (
            self.decision_function(X)
        )

        decision = np.clip(
            decision,
            -50,
            50,
        )

        probability_eruption = (
            1.0
            / (
                1.0
                + np.exp(-decision)
            )
        )

        return np.column_stack(
            [
                1.0
                - probability_eruption,
                probability_eruption,
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
