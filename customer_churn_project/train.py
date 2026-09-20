from __future__ import annotations

import json
from pathlib import Path

from churn_pipeline import (
    DATA_PATH,
    MODEL_PATH,
    SUMMARY_PATH,
    compare_configs,
    evaluate_predictions,
    fit_final_model,
    load_data,
    save_artifact,
    split_data,
    write_summary,
)


def main() -> None:
    data = load_data(DATA_PATH)
    split = split_data(data)

    configs = {
        "baseline": {
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": None,
        },
        "constrained": {
            "max_depth": 6,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
        },
    }

    comparisons = compare_configs(
        split["X_train"],
        split["y_train"],
        split["X_validation"],
        split["y_validation"],
        configs,
    )
    best = comparisons[0]

    final_pipeline = fit_final_model(split["X_train_full"], split["y_train_full"], best["params"])
    final_predictions = final_pipeline.predict(split["X_test"])
    final_metrics = evaluate_predictions(split["y_test"], final_predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(MODEL_PATH, final_pipeline, best["config_name"], best["params"])

    summary = {
        "data_path": str(DATA_PATH),
        "selected_config_name": best["config_name"],
        "selected_config": best["params"],
        "validation_comparison": [
            {
                "config_name": item["config_name"],
                "params": item["params"],
                "metrics": item["metrics"],
            }
            for item in comparisons
        ],
        "test_metrics": final_metrics,
    }
    write_summary(SUMMARY_PATH, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
