from functools import partial
from multiprocessing.pool import ThreadPool
from os.path import join
import glob
from tqdm import tqdm
from model.data import load_manifest
from model.image import get_channel_filename, stack_channels
import tensorflow as tf
from argparse import ArgumentParser

CHANNELS = ['blue', 'green', 'red', 'yellow']


def _int64array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-path', type=str, help='Path of the manifest file')
    parser.add_argument('--image-path', type=str, help='Path to the image directory')
    parser.add_argument('--image-size', type=int, default=224, help='Output size of the image')
    parser.add_argument('--output-path', type=str, help='Output file path')
    args, _ = parser.parse_known_args()

    filenames = glob.glob1(args.image_path, '*.png')
    file_ids = [x.rsplit('_')[0] for x in filenames]
    image_ids, labels, _ = load_manifest(args.train_path)
    image_dict = {k: v for k, v in zip(image_ids, labels) if k in file_ids}

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(args.output_path, options=options) as writer:
        stack_func = partial(stack_channels, image_size=args.image_size)
        for image_id, label in tqdm(image_dict.items(), total=len(image_dict)):
            image_paths = [join(args.image_path, get_channel_filename(image_id, channel)) for channel in CHANNELS]
            image = stack_func(image_paths)

            feature = {'label': _int64array_feature(label),
                       'image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

