https://github.com/sayali-nagarro/ds

# Telco Customer Churn Project

This project implements the IBM Telco Customer Churn assignment as a reproducible Python workflow.

## What is included

- `churn_pipeline.py` - shared feature engineering, preprocessing, training, evaluation, and persistence utilities
- `train.py` - trains the final decision-tree pipeline and saves the artifact
- `app.py` - FastAPI service with `POST /predict`
- `notebook/churn_assignment.ipynb` - notebook that documents the assignment requirements, analysis, and results
- `tests/` - pytest specs for the pipeline and API
- `sample_request.json` - example payload for the API

## Expected input files

Keep these files one directory above `customer_churn_project/`:

- `Data Science Assignment.pdf`
- `TelcoCustomerChurn.csv`
- `TelcoCustomerChurn - Data Dictionary.csv`

## Setup

```bash
cd ds/customer_churn_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have an environment, only install the requirements.

## Run the notebook

Open `notebook/churn_assignment.ipynb` in VS Code and run it top to bottom. The notebook:

- reads the assignment PDF and both CSV files
- validates the CSV schema
- performs EDA
- engineers features
- compares two decision-tree configurations
- saves the final model artifact
- prints clear run steps and submission artifacts

## Train and save the model

From the `customer_churn_project/` directory:

```bash
python train.py
```

This saves:

- `model/churn_model.joblib`
- `model/training_summary.json`

## Start the API

```bash
uvicorn app:app --reload
```

Then call `POST /predict` with JSON matching the sample request.

## Run tests

```bash
pytest
```

## Example request

Use `sample_request.json` as the request body for `POST /predict`.

## Notes

- The model uses a 70/30 train/test split with `random_state=42`.
- The preprocessing pipeline is saved with the model so new data is handled consistently.
- The API rejects invalid payloads through FastAPI/Pydantic validation.
