import json
import os
import pathlib
import shutil
from argparse import ArgumentParser
from datetime import datetime

import keras
import numpy as np
from keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from keras.datasets import cifar10

from data.processing import standardize_data
from model.model import convnet6, convnet4, inception_v3

MODELS = {"convnet6": convnet6, "convnet4": convnet4, "inception_v3": inception_v3}


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
    parser.add_argument("--params-path")
    parser.add_argument("--artifact-directory")
    parser.add_argument(
        "--model-artifact", help="existing model checkpoint to continue training"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    artifact_directory = os.path.join(args.artifact_directory, timestamp)
    checkpoint_format = os.path.join(
        artifact_directory, "weights.{epoch:02d}-{val_loss:.6f}.hdf5"
    )
    log_dir = os.path.join(artifact_directory, "logs")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.params_path, artifact_directory)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    X_train = standardize_data(X_train)
    X_test = standardize_data(X_test)

    with open(args.params_path) as params_file:
        hyperparameters = json.load(params_file)

    if args.model_artifact:
        model = load_model(args.model_artifact)
    else:
        model = MODELS.get(args.model_name)(
            input_shape=X_train[0].shape, n_classes=y_train.shape[-1], **hyperparameters
        )
        optim = keras.optimizers.rmsprop(
            lr=hyperparameters["learning_rate"], decay=1e-6
        )
        model.compile(optimizer=optim, loss=hyperparameters["loss"], metrics=["acc"])
    model.summary()

    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    image_data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.1,  # set range for random shear
        zoom_range=0.0,  # set range for random zoom
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        cval=0.0,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
    )

    batch_size = hyperparameters["batch_size"]
    model.fit_generator(
        generator=image_data_generator.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=200,
        validation_data=(X_test, y_test),
        class_weight=get_class_weight(y_train),
        callbacks=[
            ModelCheckpoint(checkpoint_format),
            TensorBoard(log_dir=log_dir),
            EarlyStopping(patience=15),
            ReduceLROnPlateau(patience=6, factor=0.3),
        ],
    )

    # Score trained model.
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])
