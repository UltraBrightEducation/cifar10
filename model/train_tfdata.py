import json
import os
import shutil
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils import compute_class_weight

from model.data import generate_tf_dataset
from model.model import MODELS



def get_class_weight(y_true):
    classes = np.arange(y_true.shape[-1])
    class_counts = y_true.sum(0)
    pos = 0
    y_weight = np.zeros(class_counts.sum())
    for i, count in enumerate(class_counts):
        y_weight[pos:pos + count] = i
        pos += count
    weights = np.sqrt(compute_class_weight('balanced', classes, y_weight))
    return {i: weight
            for i, weight in enumerate(weights)}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-path')
    parser.add_argument('--image-path')
    parser.add_argument('--model-name', choices=list(MODELS.keys()))
    parser.add_argument('--params-path')
    parser.add_argument('--artifact-directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    artifact_directory = os.path.join(args.artifact_directory, timestamp)
    checkpoint_format = os.path.join(artifact_directory, 'weights.{epoch:02d}-{val_loss:.6f}.hdf5')
    log_dir = os.path.join(artifact_directory, 'logs')
    os.makedirs(log_dir)
    shutil.copy(args.params_path, artifact_directory)

    with open(args.params_path) as params_file:
        hyperparameters = json.load(params_file)

    image_size = hyperparameters['image_size']
    batch_size = hyperparameters['batch_size']
    ds, labels = generate_tf_dataset(args.train_path, args.image_path, batch_size, image_size)
    model = MODELS.get(args.model_name)(
                                        input_shape=(image_size, image_size, 4),
                                        n_classes=labels.shape[-1],
                                        base_filters=hyperparameters['base_filters'],
                                        activation=hyperparameters['activation'],
                                        fc_size=hyperparameters['fc_size'],
                                        dropout=hyperparameters['dropout'],
                                        classifier_activation=hyperparameters['classifier_activation']
                                        )
    model.summary()
    model.compile('adam', loss=hyperparameters['loss'], metrics=['acc'])

    model.fit(x=ds.make_one_shot_iterator(),
              epochs=100,
              batch_size=batch_size,
              validation_split=0.15,
              callbacks=[ModelCheckpoint(checkpoint_format), TensorBoard(log_dir=log_dir)]
              )
