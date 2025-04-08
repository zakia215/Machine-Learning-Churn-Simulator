import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import os
import logging
from typing import Optional

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_spark_session(app_name: str = "ModelTraining") -> SparkSession:
    """Create and return a Spark session."""
    return (
        SparkSession.builder.appName(app_name)
        # .config(
        #     "spark.jars.packages",
        #     "org.mlflow.mlflow-spark",
        # )
        .getOrCreate()
    )


def init_mlflow(experiment_name: str):
    """Initialize MLflow tracking."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        mlflow.set_experiment(experiment_name, artifact_location="s3://mlflow")
    except Exception:
        pass

    mlflow.set_experiment(experiment_name)


def load_data(spark: SparkSession, train_path: str, test_path: str):
    """Load train and test data from parquet files."""
    try:
        train_data = spark.read.parquet(train_path)
        test_data = spark.read.parquet(test_path)
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise


def model_training(
    train_path: str, test_path: str, experiment_name: str = "churn-customer"
) -> Optional[float]:
    """
    Train and log a Spark ML model using MLflow.

    Args:
        train_path: Path to training data parquet file
        test_path: Path to test data parquet file
        experiment_name: Name of the MLflow experiment

    Returns:
        float: AUC score if successful, None if failed
    """
    setup_logging()
    logging.info("Starting model training process")

    try:
        spark = create_spark_session()
        init_mlflow(experiment_name)

        train_data, test_data = load_data(spark, train_path, test_path)

        # mlflow.spark.autolog( # Uncompatible with multithread jvm, disable for now
        #     silent=True
        # )

        with mlflow.start_run(run_name="LogisticRegression") as run:
            logging.info(f"MLflow run ID: {run.info.run_id}")

            lr = LogisticRegression(
                labelCol="Churn", featuresCol="features", maxIter=10
            )
            model = lr.fit(train_data)

            predictions = model.transform(test_data)

            feature_df = train_data.select("features").limit(5).toPandas()
            prediction_df = predictions.select("prediction").limit(5).toPandas()
            signature = infer_signature(feature_df, prediction_df)

            evaluator_auc = BinaryClassificationEvaluator(
                labelCol="Churn", metricName="areaUnderROC"
            )
            evaluator_pr = BinaryClassificationEvaluator(
                labelCol="Churn", metricName="areaUnderPR"
            )
            auc = evaluator_auc.evaluate(predictions)
            pr = evaluator_pr.evaluate(predictions)
            logging.info(f"Model AUC: {auc:.4f}")
            logging.info(f"Model PR: {pr:.4f}")

            mlflow.log_metric("auc", auc)
            mlflow.log_metric("pr", pr)

            # Input example
            train_data = train_data.withColumn(
                "features", vector_to_array(col("features"))
            )

            input_example = train_data.limit(5).toPandas()

            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="churn_log_reg",
            )
            mlflow.log_params(lr.extractParamMap())

    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

    finally:
        if "spark" in locals():
            spark.stop()
            logging.info("Spark session stopped")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <train_path> <test_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    try:
        model_training(train_path, test_path)
        print("Training completed successfully")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
