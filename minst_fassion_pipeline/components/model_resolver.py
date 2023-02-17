from tfx import v1 as tfx


def get_latest_blessed_model_resolver() -> tfx.dsl.Resolver:
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    return model_resolver
