import datetime
import json

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

class Trainer:
    def __init__(self, hyperparameters_path):
        with open(hyperparameters_path) as f:
            self.hyperparameters = json.load(f)

        self.batch_size = self.hyperparameters.pop('batch_size', None)
        self.model = None

    def build_model(self, input_shape=150, num_conv_max_pool=4, filters=32, kernel_size=3,
                    pool_size=2, activation='relu', num_dense_layers=2, dense_n=64):

        classifier = Sequential()
        for i in range(num_conv_max_pool-1):
            if i == 0:
                classifier.add(Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                                      input_shape=(input_shape, input_shape, 3),
                                      activation=activation))
            else:
                classifier.add(Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                                      activation=activation))
            classifier.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        classifier.add(Flatten())

        for i in range(num_dense_layers):
            classifier.add(Dense(units=dense_n, activation=activation))
            classifier.add(Dropout(0.5))
        classifier.add(Dense(units=1, activation='sigmoid'))

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = classifier

    def train(self):
        if not self.model:
            self.build_model(**self.hyperparameters)

        self.model.fit_generator(generator=self._create_training_generator(),
                                steps_per_epoch=8000 / self.batch_size,
                                epochs=5,
                                validation_data=self._create_eval_generator(),
                                validation_steps=2000 / self.batch_size,
                                workers=4)


    def _create_training_generator(self, data_path='dataset/training_set'):
        image_generator = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        return image_generator.flow_from_directory(
            data_path,
            target_size=(self.hyperparameters['input_shape'], self.hyperparameters['input_shape']),
            batch_size=self.batch_size,
            class_mode='binary')

    def _create_eval_generator(self, data_path='dataset/test_set'):
        image_generator = ImageDataGenerator(rescale=1. / 255)

        return image_generator.flow_from_directory(
            data_path,
            target_size=(self.hyperparameters['input_shape'], self.hyperparameters['input_shape']),
            batch_size=self.batch_size,
            class_mode='binary')

    def save(self, model_save_path):
        self.model.save(model_save_path)


    @classmethod
    def load(cls, model_load_path, hyperparameters_path):

        trainer = Trainer(hyperparameters_path)
        trainer.model = load_model(model_load_path)

        return trainer


if __name__ == '__main__':

    trainer = Trainer('hyperparameters.json')
    trainer.train()

    trainer.save('models/cat_or_dog_model{}.h5'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

