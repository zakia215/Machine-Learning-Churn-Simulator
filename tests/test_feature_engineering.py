import pytest
from pyspark.sql import SparkSession
import os

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.appName("TestFeatureEngineering").getOrCreate()

def test_feature_engineering_script(spark):
    from scripts.feature_engineering import feature_engineering
    
    input_path = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    train_path = "dataset/train_data_test.parquet"
    test_path = "dataset/test_data_test.parquet"

    # Ensure the dataset exists
    assert os.path.exists(input_path), f"Dataset not found at {input_path}"

    # Run feature engineering
    feature_engineering(input_path, train_path, test_path)

    # Check output files
    assert os.path.exists(train_path), "Train dataset not generated"
    assert os.path.exists(test_path), "Test dataset not generated"

    # Load the train dataset and check structure
    train_data = spark.read.parquet(train_path)
    assert "features" in train_data.columns, "'features' column missing in train data"
    assert "Churn" in train_data.columns, "'Churn' column missing in train data"

    # Cleanup test artifacts
    os.remove(train_path)
    os.remove(test_path)
