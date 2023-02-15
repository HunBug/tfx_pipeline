from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
import tensorflow as tf
import os
import numpy as np
from absl import logging
from tqdm import tqdm
from pathlib import Path
import tempfile

RAW_DATA_VERSION = "v1.0"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_example(image, label):
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _write_tfrecord(images, labels, file: Path):
    with tf.io.TFRecordWriter(str(file)) as writer:
        for image, label in tqdm(zip(images, labels), total=len(images)):
            serialized_example = _serialize_example(image, label)
            writer.write(serialized_example)


def _generate_tfrecord(output_dir: Path = Path(os.getcwd()) / "temp_data"):
    logging.info('Generating raw data')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    data = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)
    logging.info(
        f'Writing raw data to tfrecord, {len(data)} examples, path: {output_dir}')
    _write_tfrecord(data, labels, output_dir / 'fashion_mnist.tfrecord')


def get_example_gen(from_cache: bool = False) -> ImportExampleGen:
    if (from_cache):
        logging.info('Creating ExampleGen component from cache, return dummy')
        example_gen = ImportExampleGen(input_base="dummy")
    else:
        # Create a temporary directory.
        temp_root = Path(tempfile.mkdtemp(prefix='tfx-data'))
        # add version to the path
        temp_root = temp_root / RAW_DATA_VERSION
        Path.mkdir(temp_root, parents=True, exist_ok=True)
        _generate_tfrecord(temp_root)

        # Create an instance of the ExampleGen component
        logging.info('Creating ExampleGen component from tfrecord')
        example_gen = ImportExampleGen(input_base=str(temp_root))
    return example_gen
