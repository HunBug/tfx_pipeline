import os

DATA_ROOT = os.path.join(os.environ['HOME'], 'tfx_data')
SERVING_ROOT = "/workspaces/tfx_pipeline"
PIPELINE_NAME = 'minst_pipeline'
PIPELINE_ROOT = os.path.join(DATA_ROOT, PIPELINE_NAME + '_root')
METADATA_PATH = os.path.join(
    DATA_ROOT, 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
ENABLE_CACHE = True
USE_LATEST_EXAMPLE_ARTIFACTS = False
SERVING_MODEL_DIR = os.path.join(SERVING_ROOT, 'serving_model', PIPELINE_NAME)
