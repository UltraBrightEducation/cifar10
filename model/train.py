import json
import os
from argparse import ArgumentParser
from datetime import datetime

import keras
import numpy as np
from sklearn.utils import compute_class_weight
from keras.datasets import cifar10

from data.processing import standardize_data
from model.trainer import Cifar10Trainer, MODELS


def get_class_weight(y_true):
    classes = np.arange(y_true.shape[-1])
    class_counts = y_true.sum(0).astype(np.int64)
    pos = 0
    y_weight = np.zeros(int(class_counts.sum()))
    for i, count in enumerate(class_counts):
        y_weight[pos : pos + count] = i
        pos += count
    weights = np.sqrt(compute_class_weight("balanced", classes, y_weight))
    return {i: weight for i, weight in enumerate(weights)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", choices=list(MODELS.keys()))
    parser.add_argument("--params-path", help="path to hyperparameters.json")
    parser.add_argument("--job_name", help="name of the training job")
    parser.add_argument("--artifact-directory", help="path to save training artifacts")
    parser.add_argument(
        "--model-artifact", help="existing model checkpoint to continue training"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    artifact_directory = os.path.join(
        args.artifact_directory, args.job_name + timestamp
    )

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    X_train = standardize_data(X_train)
    X_test = standardize_data(X_test)

    with open(args.params_path) as params_file:
        hyperparameters = json.load(params_file)

    trainer = Cifar10Trainer(
        model_name=args.model_name,
        hyperparameters=hyperparameters,
        artifact_directory=artifact_directory,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        model_artifact=args.model_artifact,
    )

    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    trainer.train()
    trainer.evaluate()
