import tfx.v1 as tfx
from absl import logging

import pipeline_settings as settings
from pipeline_def import create_pipeline


def run_pipeline():
    my_pipeline = create_pipeline(
        pipeline_name=settings.PIPELINE_NAME,
        pipeline_root=settings.PIPELINE_ROOT,
        serving_model_dir=settings.SERVING_MODEL_DIR,
        enable_cache=settings.ENABLE_CACHE,
        use_latest_example_artifacts=settings.USE_LATEST_EXAMPLE_ARTIFACTS,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            settings.METADATA_PATH)
    )
    tfx.orchestration.LocalDagRunner().run(my_pipeline)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()
