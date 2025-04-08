import pytest
from pyspark.sql import SparkSession
import os

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.appName("TestModelTraining").getOrCreate()

def test_model_training_script(spark):
    from scripts.model_training import model_training
    
    train_path = "dataset/train_data.parquet"
    test_path = "dataset/test_data.parquet"
    model_path = "models/test_model_output"

    # Ensure train and test datasets exist
    assert os.path.exists(train_path), f"Train dataset not found at {train_path}"
    assert os.path.exists(test_path), f"Test dataset not found at {test_path}"

    # Run model training
    model_training(train_path, test_path, model_path)

    # Check model output
    assert os.path.exists(model_path), "Model not saved to specified path"

    # Cleanup test artifacts
    import shutil
    shutil.rmtree(model_path)
