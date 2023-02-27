from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
import os
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
import tensorflow_transform as tft
import components.model_settings as settings


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature.
    """
    outputs = {}
    # raw_image = tf.io.parse_tensor(inputs[_IMAGE_KEY][0], out_type=tf.float32)
    # raw_image = tf.reshape(raw_image, _IMAGE_SHAPE)
    # outputs[_IMAGE_KEY] = raw_image
    raw_image_dataset = inputs[settings.IMAGE_KEY]
    raw_image_dataset = tf.map_fn(
        fn=lambda x: tf.io.parse_tensor(x[0], tf.uint8, name=None),
        elems=raw_image_dataset,
        fn_output_signature=tf.TensorSpec((28, 28), dtype=tf.uint8, name=None),
        infer_shape=True)
    raw_image_dataset = tf.cast(raw_image_dataset, tf.float32)
    outputs[settings.IMAGE_KEY] = raw_image_dataset / 255.0
    outputs[settings.LABEL_KEY] = inputs[settings.LABEL_KEY]
    return outputs


def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if settings.LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(settings.LABEL_KEY)
        return transformed_features, transformed_label
    else:
        return transformed_features, None


def _get_serve_tf_examples_fn(model, tf_transform_output):
    # We must save the tft_layer to the model to ensure its assets are kept and
    # tracked.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        # Expected input is a string which is serialized tf.Example format.
        feature_spec = tf_transform_output.raw_feature_spec()
        # Because input schema includes unnecessary fields like 'species' and
        # 'island', we filter feature_spec to include required keys only.
        required_feature_spec = {
            k: v for k, v in feature_spec.items() if k in [settings.IMAGE_KEY]
        }
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              required_feature_spec)

        # Preprocess parsed input with transform operation defined in
        # preprocessing_fn().
        transformed_features, _ = _apply_preprocessing(parsed_features,
                                                       model.tft_layer)
        # Run inference with ML model.
        return model(transformed_features)

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    # We need to track the layers in the model in order to save it.
    # TODO(b/162357359): Revise once the bug is resolved.
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(
            serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema=tf_transform_output.raw_metadata.schema)

    transform_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        return _apply_preprocessing(raw_features, transform_layer)

    return dataset.map(apply_transform).repeat()


def _build_keras_model() -> tf.keras.Model:
    """Creates a DNN Keras model for classifying penguin data.

    Returns:
      A Keras Model.
    """
    # The model below is built with Functional API, please refer to
    # https://www.tensorflow.org/guide/keras/overview for all API options.
    inputs = keras.layers.Input(
        shape=settings.IMAGE_SHAPE, name=settings.IMAGE_KEY)

    d = inputs
    d = keras.layers.Flatten()(d)
    for _ in range(3):
        d = keras.layers.Dense(16, activation='relu')(d)
    outputs = keras.layers.Dense(10)(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=settings.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=settings.EVAL_BATCH_SIZE)

    model = _build_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)


def get_trainer(examples, transform) -> tfx.components.Trainer:
    # TODO num_steps should be a parameter
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath(__file__),
        examples=examples.outputs['examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=20000),
        eval_args=tfx.proto.EvalArgs(num_steps=500))
    return trainer


def get_transform(examples, schema) -> tfx.components.Transform:
    transform = tfx.components.Transform(
        examples=examples.outputs['examples'],
        schema=schema.outputs['schema'],
        module_file=os.path.abspath(__file__))
    return transform
