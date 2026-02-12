from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass(frozen=True)
class ModelBuildSpec:
    kind: str
    numeric_features: list[str]
    categorical_features: list[str]
    max_iter: int
    n_jobs: int
    random_seed: int


def build_model(
    *,
    kind: str,
    numeric_features: list[str],
    categorical_features: list[str],
    max_iter: int,
    n_jobs: int,
    random_seed: int,
) -> Pipeline:
    """
    Build a sklearn Pipeline for binary classification.

    Supported kinds:
    - "logreg": LogisticRegression + OneHotEncoder for categoricals, class_weight="balanced"
    - "hgb": HistGradientBoostingClassifier + OrdinalEncoder for categoricals (dense)

    Notes:
    - HGB cannot consume sparse matrices from OneHotEncoder. For HGB we use OrdinalEncoder.
    """
    spec = ModelBuildSpec(
        kind=kind,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_seed=random_seed,
    )

    if spec.kind == "logreg":
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ],
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ],
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, spec.numeric_features),
                ("cat", categorical_pipe, spec.categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        clf = LogisticRegression(
            max_iter=spec.max_iter,
            n_jobs=spec.n_jobs,
            class_weight="balanced",
            random_state=spec.random_seed,
            solver="lbfgs",
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    if spec.kind == "hgb":
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ],
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ],
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, spec.numeric_features),
                ("cat", categorical_pipe, spec.categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
            sparse_threshold=0.0,
        )

        clf = HistGradientBoostingClassifier(
            random_state=spec.random_seed,
            max_depth=None,
            learning_rate=0.1,
            max_iter=300,
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    raise ValueError(f"Unsupported model kind: {spec.kind}")
