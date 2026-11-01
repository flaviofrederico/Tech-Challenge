from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-scientist',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'medical_screening_training_pipeline',
    default_args=default_args,
    description='DAG para automação do treino do modelo de triagem médica',
    schedule_interval='@monthly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    # Tarefa para executar o script de treino do modelo
    run_training = BashOperator(
        task_id='run_model_training',
        bash_command='python /app/app/train.py',
    )

    run_training