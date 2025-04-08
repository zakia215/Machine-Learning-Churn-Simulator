from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    description='End-to-end pipeline for customer churn prediction',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

spark_jobs_dir = 'scripts'
data_dir = 'dataset'

feature_engineering = SparkSubmitOperator(
    task_id='feature_engineering',
    application=f'{spark_jobs_dir}/feature_engineering.py',
    name='feature_engineering',
    conn_id='spark_default',
    application_args=[
        f'{data_dir}/WA_Fn-UseC_-Telco-Customer-Churn.csv',  # Input raw data
        f'{data_dir}/train_data.parquet',  # Output train data
        f'{data_dir}/test_data.parquet',  # Output test data
    ],
    dag=dag,
)

model_training = SparkSubmitOperator(
    task_id='model_training',
    application=f'{spark_jobs_dir}/model_training.py',
    name='model_training',
    conn_id='spark_default',
    application_args=[
        f'{data_dir}/train_data.parquet',  # Input train data
        f'{data_dir}/test_data.parquet',  # Input test data
    ],
    dag=dag,
)

feature_engineering >> model_training
