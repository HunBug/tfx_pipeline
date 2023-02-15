from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx_bsl.public import tfxio


_LABEL_KEY = 'label'
_IMAGE_KEY = 'image'
_IMAGE_SHAPE = (28, 28)

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


# NEW: TFX Transform will call this function.
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
    raw_image_dataset = inputs[_IMAGE_KEY]
    raw_image_dataset = tf.map_fn(fn=lambda x: tf.io.parse_tensor(
        x[0], tf.uint8, name=None), elems=raw_image_dataset, fn_output_signature=tf.TensorSpec((28, 28), dtype=tf.uint8,    name=None), infer_shape=True)
    raw_image_dataset = tf.cast(raw_image_dataset, tf.float32)
    outputs[_IMAGE_KEY] = raw_image_dataset / 255.0
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]
    return outputs


# NEW: This function will apply the same transform operation to training data
#      and serving requests.
def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if _LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(_LABEL_KEY)
        return transformed_features, transformed_label
    else:
        return transformed_features, None


# NEW: This function will create a handler function which gets a serialized
#      tf.example, preprocess and run an inference with it.
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
            k: v for k, v in feature_spec.items() if k in [_IMAGE_KEY]
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
    inputs = keras.layers.Input(shape=_IMAGE_SHAPE, name=_IMAGE_KEY)

    d = inputs
    d = keras.layers.Flatten()(d)
    for _ in range(2):
        d = keras.layers.Dense(8, activation='relu')(d)
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
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    # NEW: Save a computation graph including transform layer.
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
