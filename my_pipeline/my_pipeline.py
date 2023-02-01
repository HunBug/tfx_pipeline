import os
from typing import Optional, List, Text
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx

PIPELINE_NAME = 'my_pipeline'
PIPELINE_ROOT = os.path.join('.', PIPELINE_NAME + '_root')
METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
ENABLE_CACHE = True

from tfx.components import CsvExampleGen
DATA_PATH = os.path.join('.', 'data')

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    enable_cache: Optional[bool] = False,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> tfx.dsl.Pipeline:
    components = []

    example_gen = tfx.components.CsvExampleGen(input_base=DATA_PATH)
    components.append(example_gen)
    
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,  
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )

def run_pipeline():
    my_pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
    )
    tfx.orchestration.LocalDagRunner().run(my_pipeline)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()