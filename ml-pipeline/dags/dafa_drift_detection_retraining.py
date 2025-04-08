from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "check_data_drift_and_retrain",
    default_args=default_args,
    description="Check data drift and retrain model if necessary",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

spark_jobs_dir = "scripts"
data_dir = "dataset"

# Task to calculate PSI using Spark
calculate_psi = SparkSubmitOperator(
    task_id="calculate_psi_task",
    application=f"{spark_jobs_dir}/drift_detection.py",
    name="calculate_psi",
    conn_id="spark_default",
    application_args=[
        f"{data_dir}/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        f"{data_dir}/train_data.parquet",
        f"{data_dir}/test_data.parquet",
        "psi_result.json",  # Output JSON path for PSI
    ],
    dag=dag,
)

# Task to push PSI value from JSON to XCom
def push_psi_to_xcom(**kwargs):
    import json
    ti = kwargs["ti"]
    output_path = "psi_result.json"

    # Read PSI from the output file
    with open(output_path, "r") as f:
        psi_data = json.load(f)
    psi_value = psi_data.get("psi", 0.0)

    # Push PSI value to XCom
    ti.xcom_push(key="psi", value=psi_value)

push_psi_task = PythonOperator(
    task_id="push_psi_to_xcom",
    python_callable=push_psi_to_xcom,
    provide_context=True,
    dag=dag,
)

# Branching task to decide based on PSI
def decide_based_on_psi(**kwargs):
    ti = kwargs["ti"]
    psi_value = float(ti.xcom_pull(task_ids="push_psi_to_xcom", key="psi"))
    threshold = 0.2
    return "feature_engineering" if psi_value > threshold else "stop_dag_execution"

branch_task = PythonOperator(
    task_id="decide_next_step",
    python_callable=decide_based_on_psi,
    provide_context=True,
    dag=dag,
)

# Stop DAG if no drift
stop_dag = DummyOperator(task_id="stop_dag_execution", dag=dag)

# Proceed with retraining
feature_engineering = SparkSubmitOperator(
    task_id="feature_engineering",
    application=f"{spark_jobs_dir}/feature_engineering.py",
    name="feature_engineering",
    conn_id="spark_default",
    application_args=[
        f"{data_dir}/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        f"{data_dir}/train_data.parquet",
        f"{data_dir}/test_data.parquet",
    ],
    dag=dag,
)

model_training = SparkSubmitOperator(
    task_id="model_training",
    application=f"{spark_jobs_dir}/model_training.py",
    name="model_training",
    conn_id="spark_default",
    application_args=[
        f"{data_dir}/train_data.parquet",
        f"{data_dir}/test_data.parquet",
    ],
    dag=dag,
)

end_dag = DummyOperator(task_id="end_dag", dag=dag)

# Define task dependencies
calculate_psi >> push_psi_task >> branch_task
branch_task >> [stop_dag, feature_engineering]
feature_engineering >> model_training >> end_dag
