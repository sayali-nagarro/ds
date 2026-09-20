from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT.parent / "TelcoCustomerChurn.csv"
DATA_DICTIONARY_PATH = PROJECT_ROOT.parent / "TelcoCustomerChurn - Data Dictionary.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "churn_model.joblib"
SUMMARY_PATH = PROJECT_ROOT / "model" / "training_summary.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20
POSITIVE_CLASS = "Yes"

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count",
    "avg_monthly_charge_per_tenure",
    "is_month_to_month",
]

EXPECTED_INPUT_COLUMNS = [
    ID_COLUMN,
    *CATEGORICAL_COLUMNS,
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create model features from the raw customer record."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        for column in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        for column in SERVICE_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].astype(str).str.strip()

        if "Contract" in frame.columns:
            frame["is_month_to_month"] = (frame["Contract"].astype(str).str.strip() == "Month-to-month").astype(int)
        else:
            frame["is_month_to_month"] = 0

        service_flags = []
        for column in SERVICE_COLUMNS:
            if column in frame.columns:
                service_flags.append(frame[column].astype(str).str.strip().eq("Yes").astype(int))
        if service_flags:
            frame["service_count"] = np.sum(service_flags, axis=0)
        else:
            frame["service_count"] = 0

        tenure = frame.get("tenure", pd.Series(index=frame.index, dtype=float)).replace(0, np.nan)
        monthly = frame.get("MonthlyCharges", pd.Series(index=frame.index, dtype=float))
        frame["avg_monthly_charge_per_tenure"] = (monthly / tenure).replace([np.inf, -np.inf], np.nan)

        if ID_COLUMN in frame.columns:
            frame = frame.drop(columns=[ID_COLUMN])

        return frame


@dataclass
class ModelBundle:
    pipeline: Pipeline
    positive_class: str = POSITIVE_CLASS
    selected_config_name: str | None = None


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return clean_dataframe(frame)


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].astype(str).str.strip()
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce")
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(str).str.strip()
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_pipeline(
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | dict[str, float] | None = None,
    ccp_alpha: float = 0.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def split_data(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=relative_validation_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
    }


def evaluate_predictions(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["No", "Yes"]).tolist(),
    }


def compare_configs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for config_name, config in configs.items():
        pipeline = build_pipeline(**config)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_validation)
        metrics = evaluate_predictions(y_validation, predictions)
        comparisons.append(
            {
                "config_name": config_name,
                "params": config,
                "metrics": metrics,
                "pipeline": pipeline,
            }
        )
    comparisons.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["accuracy"]), reverse=True)
    return comparisons


def fit_final_model(X_train_full: pd.DataFrame, y_train_full: pd.Series, config: dict[str, Any]) -> Pipeline:
    pipeline = build_pipeline(**config)
    pipeline.fit(X_train_full, y_train_full)
    return pipeline


def get_positive_class_probability(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(frame)
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(POSITIVE_CLASS)
    return probabilities[:, positive_index]


def save_artifact(path: Path | str, pipeline: Pipeline, selected_config_name: str, selected_config: dict[str, Any]) -> None:
    artifact = {
        "pipeline": pipeline,
        "positive_class": POSITIVE_CLASS,
        "selected_config_name": selected_config_name,
        "selected_config": selected_config,
    }
    joblib.dump(artifact, path)


def load_artifact(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def write_summary(path: Path | str, summary: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
