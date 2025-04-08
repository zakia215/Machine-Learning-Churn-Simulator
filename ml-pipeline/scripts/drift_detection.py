import pandas as pd
import numpy as np
import json
import sys
import mlflow

def calculate_psi(expected, actual, bins=10):
    expected_percents, bins = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bins)

    expected_percents = expected_percents / sum(expected_percents)
    actual_percents = actual_percents / sum(actual_percents)

    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def drift_detection(input_path, train_path, test_path, output_path):
    baseline_data = pd.read_parquet(train_path)
    new_data = pd.read_parquet(test_path)

    psi = calculate_psi(baseline_data["features"], new_data["features"])

    mlflow.set_experiment("Customer Churn Prediction")
    with mlflow.start_run():
        mlflow.log_metric("psi", psi)

    # Write the PSI value to a JSON file
    with open(output_path, "w") as f:
        json.dump({"psi": psi}, f)

    print(f"PSI calculated and saved: {psi}")
    return psi

if __name__ == "__main__":
    input_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]
    drift_detection(input_path, train_path, test_path, output_path)
