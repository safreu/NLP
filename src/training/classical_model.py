"""Scikit-learn model wrappers for KEEP/SIMPLIFY prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from storage.classical_io import atomic_write

DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    display_name: str
    artifact_name: str
    sklearn_class: str
    parameters: dict[str, Any]


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        slug="logistic_regression",
        display_name="sklearn LogisticRegression",
        artifact_name="logistic_regression.pkl",
        sklearn_class="sklearn.linear_model.LogisticRegression",
        parameters={
            "max_iter": 1000,
            "class_weight": "balanced",
        },
    ),
    "svm": ModelSpec(
        slug="svm",
        display_name="sklearn LinearSVC",
        artifact_name="svm.pkl",
        sklearn_class="sklearn.svm.LinearSVC",
        parameters={
            "class_weight": "balanced",
            "dual": "auto",
            "max_iter": 5000,
            "tol": 0.0001,
        },
    ),
    "random_forest": ModelSpec(
        slug="random_forest",
        display_name="sklearn RandomForestClassifier",
        artifact_name="random_forest.pkl",
        sklearn_class="sklearn.ensemble.RandomForestClassifier",
        parameters={
            "n_estimators": 25,
            "max_depth": 15,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    ),
}


def ensure_sparse_int32(matrix: Any) -> Any:
    """Normalize sparse matrix index arrays for estimators that require int32."""
    if hasattr(matrix, "tocsr"):
        matrix = matrix.tocsr(copy=False)
    if hasattr(matrix, "indices") and matrix.indices.dtype != "int32":
        matrix.indices = matrix.indices.astype("int32", copy=False)
    if hasattr(matrix, "indptr") and matrix.indptr.dtype != "int32":
        matrix.indptr = matrix.indptr.astype("int32", copy=False)
    return matrix


def get_model_spec(model_type: str) -> ModelSpec:
    normalized = model_type.strip().lower().replace("-", "_")
    if normalized not in MODEL_SPECS:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model type '{model_type}'. Known model types: {known}.")
    return MODEL_SPECS[normalized]


class SimplificationModel:
    """Thin wrapper around DictVectorizer and an interchangeable classifier."""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        random_state: int = DEFAULT_RANDOM_STATE,
        classifier_parameters: dict[str, Any] | None = None,
    ) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer
            from sklearn.svm import LinearSVC
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for training/evaluation. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        spec = get_model_spec(model_type)
        self.model_type = spec.slug
        self.model_spec = spec
        self.random_state = random_state
        parameters = {**spec.parameters, **(classifier_parameters or {})}

        if spec.slug == "logistic_regression":
            classifier = LogisticRegression(
                **parameters,
                random_state=random_state,
            )
        elif spec.slug == "svm":
            classifier = LinearSVC(
                **parameters,
                random_state=random_state,
            )
        elif spec.slug == "random_forest":
            classifier = RandomForestClassifier(
                **parameters,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unhandled model type '{spec.slug}'.")

        steps: list[tuple[str, Any]] = [("vectorizer", DictVectorizer(sparse=True))]
        if spec.slug == "svm":
            steps.append(
                (
                    "sparse_index_dtype",
                    FunctionTransformer(ensure_sparse_int32, accept_sparse=True),
                )
            )
        steps.append(("classifier", classifier))
        self.pipeline = Pipeline(steps)

    def fit(self, features: list[dict[str, object]], labels: list[str]) -> None:
        self.pipeline.fit(features, labels)

    def fit_vectorizer(self, features: list[dict[str, object]]) -> None:
        self.pipeline.named_steps["vectorizer"].fit(features)

    def transform_features(self, features: list[dict[str, object]]) -> Any:
        return self.pipeline.named_steps["vectorizer"].transform(features)

    def set_vectorizer(self, vectorizer: Any) -> None:
        self.pipeline.named_steps["vectorizer"] = vectorizer
        self.pipeline.steps[0] = ("vectorizer", vectorizer)

    def fit_classifier_from_matrix(self, matrix: Any, labels: list[str]) -> None:
        if "sparse_index_dtype" in self.pipeline.named_steps:
            matrix = self.pipeline.named_steps["sparse_index_dtype"].transform(matrix)
        self.pipeline.named_steps["classifier"].fit(matrix, labels)

    def predict(self, features: list[dict[str, object]]) -> list[str]:
        return list(self.pipeline.predict(features))

    def report(self, features: list[dict[str, object]], labels: list[str]) -> str:
        from sklearn.metrics import classification_report

        predictions = self.predict(features)
        return str(classification_report(labels, predictions, zero_division=0))

    def metrics(self, features: list[dict[str, object]], labels: list[str]) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score

        predictions = self.predict(features)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        }

    def feature_names(self) -> list[str]:
        vectorizer = self.pipeline.named_steps["vectorizer"]
        return list(vectorizer.get_feature_names_out())

    def save(self, path: Path) -> None:
        import joblib

        def writer(tmp_path: Path) -> None:
            joblib.dump(
                {
                    "pipeline": self.pipeline,
                    "model_type": self.model_type,
                    "model_spec": self.model_spec,
                    "random_state": self.random_state,
                    "parameters": self.training_parameters(),
                },
                tmp_path,
            )

        atomic_write(path, writer, path.suffix or ".pkl")

    def training_parameters(self) -> dict[str, Any]:
        classifier = self.pipeline.named_steps["classifier"]
        return {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "sklearn_class": self.model_spec.sklearn_class,
            "classifier_parameters": classifier.get_params(deep=False),
            "vectorizer": "sklearn.feature_extraction.DictVectorizer",
        }

    @classmethod
    def load(cls, path: Path) -> SimplificationModel:
        import joblib

        payload = joblib.load(path)
        model = cls.__new__(cls)
        if isinstance(payload, dict) and "pipeline" in payload:
            model.pipeline = payload["pipeline"]
            model.model_type = payload.get("model_type", "logistic_regression")
            model.model_spec = get_model_spec(model.model_type)
            model.random_state = payload.get("random_state", DEFAULT_RANDOM_STATE)
        else:
            model.pipeline = payload
            model.model_type = "logistic_regression"
            model.model_spec = get_model_spec(model.model_type)
            model.random_state = DEFAULT_RANDOM_STATE
        return model
