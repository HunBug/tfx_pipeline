import datetime
import tfx.v1 as tfx
from absl import logging
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig

import pipeline_settings as settings
from .pipeline_def import create_pipeline

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}
logging.set_verbosity(logging.INFO)

# 'DAG' below need to be kept for Airflow to detect dag.
DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    create_pipeline(
        pipeline_name=settings.PIPELINE_NAME,
        pipeline_root=settings.PIPELINE_ROOT,
        serving_model_dir=settings.SERVING_MODEL_DIR,
        enable_cache=settings.ENABLE_CACHE,
        use_latest_example_artifacts=settings.USE_LATEST_EXAMPLE_ARTIFACTS,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            settings.METADATA_PATH)
    ))
