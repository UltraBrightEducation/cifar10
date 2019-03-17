import json
import os
import pathlib

import keras
from keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import compute_class_weight

from model.train import MODELS


class Cifar10Trainer:
    def __init__(
        self,
        model_name,
        hyperparameters,
        artifact_directory,
        X_train,
        y_train,
        X_test,
        y_test,
        model_artifact=None,
    ):
        self.hyperparameters = hyperparameters
        self.batch_size = self.hyperparameters["batch_size"]
        self.artifact_directory = artifact_directory
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.input_shape = X_train[0].shape
        self.n_classes = y_train.shape[-1]
        if model_artifact:
            self.model = load_model(model_artifact)
        else:
            self.model = self._build_model(model_name)
        self.image_data_generator = self._get_image_generator()
        pathlib.Path(self.artifact_directory).mkdir(parents=True, exist_ok=True)

    def _build_model(self, model_name):
        model = MODELS.get(model_name)(
            input_shape=self.input_shape,
            n_classes=self.n_classes,
            **self.hyperparameters
        )
        optim = keras.optimizers.rmsprop(
            lr=self.hyperparameters.get("learning_rate", 1e-3),
            decay=self.hyperparameters.get("lr_decay", 1e-6),
        )
        model.compile(
            optimizer=optim, loss=self.hyperparameters["loss"], metrics=["acc"]
        )
        model.summary()
        return model

    def _get_image_generator(self):
        image_generator = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=self.hyperparameters.get(
                "rotation_range", 15
            ),  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=self.hyperparameters.get("width_shift_range", 0.1),
            # randomly shift images vertically (fraction of total height)
            height_shift_range=self.hyperparameters.get("height_shift_range", 0.1),
            shear_range=self.hyperparameters.get(
                "shear_range", 0.1
            ),  # set range for random shear
            zoom_range=self.hyperparameters.get(
                "zoom_range", 0.0
            ),  # set range for random zoom
            channel_shift_range=0.0,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode="nearest",
            cval=0.0,  # value used for fill_mode = "constant"
            horizontal_flip=self.hyperparameters.get(
                "horizontal_flip", True
            ),  # randomly flip images
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
        return image_generator.flow(
            self.X_train, self.y_train, batch_size=self.batch_size
        )

    def train(self):
        checkpoint_format = os.path.join(
            self.artifact_directory, "weights.{epoch:02d}-{val_loss:.6f}.hdf5"
        )
        with open(
            os.path.join(self.artifact_directory, "hyperparameters.json"), "w"
        ) as f:
            json.dump(self.hyperparameters, f, indent=4)

        log_dir = os.path.join(self.artifact_directory, "logs")
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.model.fit_generator(
            generator=self.image_data_generator,
            steps_per_epoch=len(self.X_train) // self.batch_size,
            epochs=200,
            validation_data=(self.X_test, self.y_test),
            class_weight=self.get_class_weight(self.y_train),
            callbacks=[
                ModelCheckpoint(checkpoint_format),
                TensorBoard(log_dir=log_dir),
                EarlyStopping(
                    patience=self.hyperparameters.get("early_stopping_patience", 15)
                ),
                ReduceLROnPlateau(
                    patience=self.hyperparameters.get("reduce_lr_patience", 6),
                    factor=self.hyperparameters.get("reduce_lr_factor", 0.3),
                ),
            ],
        )

    @staticmethod
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

    def evaluate(self):
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Test loss:", scores[0])
        print("Test accuracy:", scores[1])
