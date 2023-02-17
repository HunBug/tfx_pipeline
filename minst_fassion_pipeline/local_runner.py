from typing import Optional, List, Text
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx
from components.example_gen.example_generator import get_example_gen
from components.statistics_generator.statistics_generator import get_statistics_gen
from components.schema_generator.schema_generator import get_schema_gen
from components.trainer.trainer import get_trainer, get_transform
import pipeline_settings as settings


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    enable_cache: Optional[bool] = False,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> tfx.dsl.Pipeline:
    LOAD_CACHED_DATA = False
    if (LOAD_CACHED_DATA):
        example_producer_component = get_example_gen(from_cache=True)
        example_gen = tfx.dsl.Resolver(
            strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
            examples=tfx.dsl.Channel(
                type=tfx.types.standard_artifacts.Examples, producer_component_id=example_producer_component.id)
        ).with_id('ExampleResolver_LatestArtifactStrategy')
    else:
        example_gen = get_example_gen()

    statistics_gen = get_statistics_gen(example_gen)
    schema_gen = get_schema_gen(statistics_gen)
    transofrm = get_transform(example_gen, schema_gen)
    trainer = get_trainer(example_gen, transofrm)

    components = []
    components.append(example_gen)
    components.append(statistics_gen)
    components.append(schema_gen)
    components.append(transofrm)
    components.append(trainer)

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
        pipeline_name=settings.PIPELINE_NAME,
        pipeline_root=settings.PIPELINE_ROOT,
        enable_cache=settings.ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            settings.METADATA_PATH)
    )
    tfx.orchestration.LocalDagRunner().run(my_pipeline)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()
