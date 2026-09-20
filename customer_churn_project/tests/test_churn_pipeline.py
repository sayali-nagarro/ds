from __future__ import annotations

import pandas as pd

from churn_pipeline import FeatureEngineer, build_pipeline, clean_dataframe, evaluate_predictions


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            },
            {
                "customerID": "0002",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 34,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 56.95,
                "TotalCharges": "1889.5",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_dataframe_converts_total_charges_to_numeric() -> None:
    cleaned = clean_dataframe(sample_frame())
    assert cleaned["TotalCharges"].dtype.kind in {"f", "i"}
    assert cleaned.loc[0, "Churn"] == "No"


def test_feature_engineer_adds_expected_columns() -> None:
    cleaned = clean_dataframe(sample_frame()).drop(columns=["Churn"])
    engineered = FeatureEngineer().fit_transform(cleaned)
    assert "service_count" in engineered.columns
    assert "avg_monthly_charge_per_tenure" in engineered.columns
    assert "is_month_to_month" in engineered.columns
    assert "customerID" not in engineered.columns


def test_pipeline_fits_and_predicts_on_small_sample() -> None:
    frame = clean_dataframe(sample_frame())
    X = frame.drop(columns=["Churn"])
    y = frame["Churn"]
    pipeline = build_pipeline(max_depth=3, min_samples_leaf=1)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    metrics = evaluate_predictions(y, predictions)
    assert len(predictions) == len(y)
    assert 0.0 <= metrics["accuracy"] <= 1.0
