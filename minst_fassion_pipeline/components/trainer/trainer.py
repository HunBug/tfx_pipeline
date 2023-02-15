from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
import os
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

# We don't need to specify _FEATURE_KEYS and _FEATURE_SPEC any more.
# Those information can be read from the given schema file.

_LABEL_KEY = 'label'
_IMAGE_KEY = 'image'
_IMAGE_SHAPE = (28, 28)

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


@tf.function
def _image_decoder(input, output):
    tf.print("output:", output)
    tf.print("output[0]:", output[0])
    numpy_image = tf.io.parse_tensor(
        input[_IMAGE_KEY][0][0], out_type=tf.uint8)
    transformed_input = {}
    transformed_input[_IMAGE_KEY] = numpy_image
    transformed_output = {}
    transformed_output[_LABEL_KEY] = output[0][0]
    return transformed_input, output[0][0]


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      schema: schema of the input data.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema).repeat()


def _build_keras_model(schema: schema_pb2.Schema) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying penguin data.

    Returns:
      A Keras Model.
    """
    # The model below is built with Functional API, please refer to
    # https://www.tensorflow.org/guide/keras/overview for all API options.

    inputs = keras.layers.Input(shape=_IMAGE_SHAPE, name=_IMAGE_KEY)

    d = inputs
    for _ in range(2):
        d = keras.layers.Dense(8, activation='relu')(d)
    outputs = keras.layers.Dense(10, name=_LABEL_KEY)(d)

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

    # ++ Changed code: Reads in schema file passed to the Trainer component.
    schema = tfx.utils.parse_pbtxt_file(
        fn_args.schema_path, schema_pb2.Schema())
    # ++ End of the changed code.

    tf.print("fn_args.train_files:", fn_args.train_files)
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=_TRAIN_BATCH_SIZE)
    tf.print("fn_args.eval_files:", fn_args.eval_files)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model(schema)
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf')


def get_trainer(examples, schema) -> tfx.components.Trainer:
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath(__file__),
        examples=examples.outputs['examples'],
        schema=schema.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5))
    return trainer
