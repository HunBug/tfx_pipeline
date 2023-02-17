from tfx import v1 as tfx
from absl import logging


def get_statistics_gen(example_gen) -> tfx.components.StatisticsGen:
    logging.info('Creating StatisticsGen component')
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
    return statistics_gen