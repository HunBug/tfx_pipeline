from tfx import v1 as tfx
from absl import logging


def get_schema_gen(statistics_gen) -> tfx.components.SchemaGen:
    logging.info('Creating SchemaGen component')
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    return schema_gen
