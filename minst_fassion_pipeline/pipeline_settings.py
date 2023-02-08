import os

DATA_ROOT = os.path.join(os.environ['HOME'], 'tfx_data')
PIPELINE_NAME = 'minst_pipeline'
PIPELINE_ROOT = os.path.join(DATA_ROOT, PIPELINE_NAME + '_root')
METADATA_PATH = os.path.join(
    DATA_ROOT, 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
ENABLE_CACHE = True
