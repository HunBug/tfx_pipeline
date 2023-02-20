from tfx import v1 as tfx
import tensorflow_model_analysis as tfma
import components.model_settings as settings


def get_evaluator(example_gen, trainer, model_resolver) -> tfx.components.Evaluator:
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(
            signature_name='serving_default',
            preprocessing_function_names=['transform_features'],
            label_key=settings.LABEL_KEY)],
        slicing_specs=[
            # An empty slice spec means the overall slice, i.e. the whole dataset.
            tfma.SlicingSpec(),
            # Calculate metrics for each penguin species.
            tfma.SlicingSpec(feature_keys=[settings.LABEL_KEY]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(per_slice_thresholds={
                'sparse_categorical_accuracy':
                tfma.PerSliceMetricThresholds(thresholds=[
                    tfma.PerSliceMetricThreshold(
                        slicing_specs=[tfma.SlicingSpec()],
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.6}),
                            # Change threshold will be ignored if there is no
                            # baseline model resolved from MLMD (first run).
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10}))
                    )]),
            })],
    )
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)
    return evaluator
