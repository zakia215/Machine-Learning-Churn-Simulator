from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import os
import mlflow
import logging

def init_mlflow(experiment_name: str):
    """Initialize MLflow tracking."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        mlflow.set_experiment(experiment_name, artifact_location="s3://mlflow")
    except Exception:
        pass

    mlflow.set_experiment(experiment_name)

def feature_engineering(input_path, train_path, test_path, test_ratio=0.3):
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    
    init_mlflow("feature_engineering")
    mlflow.start_run(run_name="FeatureEngineering")

    # Load the data
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Convert TotalCharges to numeric
    df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))

    # Drop rows with missing values in TotalCharges
    df = df.na.drop(subset=["TotalCharges"])

    # Convert Churn to numeric using StringIndexer
    churn_indexer = StringIndexer(inputCol="Churn", outputCol="Churn_indexed")
    df = churn_indexer.fit(df).transform(df).drop("Churn").withColumnRenamed("Churn_indexed", "Churn")

    # Define columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols_ohe = ['PaymentMethod', 'Contract', 'InternetService']  # For one-hot encoding
    cat_cols_le = list(set(df.columns) - set(num_cols) - set(cat_cols_ohe) - {'customerID', 'Churn'})  # For label encoding

    # Split data into train and test sets
    train_data, test_data = df.randomSplit([1 - test_ratio, test_ratio], seed=42)

    # Step 1: Label encoding for cat_cols_le
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in cat_cols_le + cat_cols_ohe]

    # Step 2: One-hot encoding for cat_cols_ohe
    encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_ohe") for col in cat_cols_ohe]

    # Step 3: Assemble numeric features
    assembler = VectorAssembler(inputCols=num_cols, outputCol="num_features")

    # Step 4: Scale numeric features
    scaler = StandardScaler(inputCol="num_features", outputCol="scaled_num_features", withMean=True, withStd=True)

    # Build the pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

    # Fit the pipeline on training data only
    pipeline_model = pipeline.fit(train_data)

    # Transform train and test data
    train_data = pipeline_model.transform(train_data)
    test_data = pipeline_model.transform(test_data)

    # Step 5: Assemble all features
    feature_columns = [f"{col}_ohe" for col in cat_cols_ohe] + [f"{col}_indexed" for col in cat_cols_le] + ["scaled_num_features"]
    final_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    train_data = final_assembler.transform(train_data).select("features", "Churn")
    test_data = final_assembler.transform(test_data).select("features", "Churn")

    # Save train and test sets
    train_data.write.mode("overwrite").parquet(train_path)
    test_data.write.mode("overwrite").parquet(test_path)
    
    # Save pipeline model
    mlflow.spark.log_model(spark_model=pipeline_model,artifact_path="model",registered_model_name="feature_engineering")
    
    spark.stop()


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    feature_engineering(input_path, train_path, test_path)
