from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import app as app_module
from churn_pipeline import build_pipeline


def build_trained_pipeline() -> object:
    frame = pd.DataFrame(
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
                "TotalCharges": 29.85,
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
                "TotalCharges": 1889.5,
            },
            {
                "customerID": "0003",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "Yes",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.7,
                "TotalCharges": 151.65,
            },
            {
                "customerID": "0004",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 45,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 42.3,
                "TotalCharges": 1840.75,
            },
        ]
    )
    labels = pd.Series(["No", "No", "Yes", "No"])
    pipeline = build_pipeline(max_depth=3, min_samples_leaf=1)
    pipeline.fit(frame, labels)
    return pipeline


def test_predict_endpoint_returns_prediction(monkeypatch) -> None:
    trained_pipeline = build_trained_pipeline()

    monkeypatch.setattr(
        app_module,
        "get_artifact",
        lambda: {"pipeline": trained_pipeline},
    )

    client = TestClient(app_module.app)
    response = client.post(
        "/predict",
        json={
            "customerID": "9999",
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
            "TotalCharges": 29.85,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {"Yes", "No"}
    assert 0.0 <= payload["churn_probability"] <= 1.0


def test_predict_endpoint_rejects_invalid_payload(monkeypatch) -> None:
    trained_pipeline = build_trained_pipeline()
    monkeypatch.setattr(app_module, "get_artifact", lambda: {"pipeline": trained_pipeline})
    client = TestClient(app_module.app)
    response = client.post(
        "/predict",
        json={
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
            "MonthlyCharges": "invalid",
            "TotalCharges": 29.85,
        },
    )

    assert response.status_code == 422
