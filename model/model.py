from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    GlobalMaxPooling2D,
    Dense,
    Dropout,
    Flatten,
)
from keras.models import Model


def _conv_block(x, filters, kernel_size, activation, pool=(2, 2)):
    hidden = Conv2D(filters, kernel_size, activation=activation, padding="same")(x)
    hidden = Conv2D(filters, kernel_size, activation=activation, padding="same")(hidden)
    hidden = MaxPool2D(pool_size=pool)(hidden)
    return hidden


def convnet7(
    input_shape,
    n_classes,
    base_filters,
    activation,
    fc_size,
    dropout,
    classifier_activation,
):
    image = Input(shape=input_shape)

    conv_1 = _conv_block(
        image, filters=base_filters, kernel_size=3, activation=activation, pool=2
    )
    conv_2 = Dropout(dropout)(conv_1)
    conv_3 = _conv_block(conv_2, base_filters * 2, kernel_size=3, activation=activation)
    conv_4 = _conv_block(
        conv_3, base_filters * 2, kernel_size=3, activation=activation, pool=2
    )
    conv_5 = Dropout(dropout)(conv_4)
    conv_6 = Conv2D(
        base_filters * 2, kernel_size=1, activation=activation, padding="same"
    )(conv_5)

    hidden = GlobalMaxPooling2D()(conv_6)
    hidden = Dropout(dropout)(hidden)

    fc_1 = Dense(fc_size, activation=activation)(hidden)
    fc_1 = Dropout(dropout)(fc_1)
    fc_2 = Dense(fc_size, activation=activation)(fc_1)
    fc_2 = Dropout(dropout)(fc_2)

    predictions = Dense(n_classes, activation=classifier_activation)(fc_2)
    return Model(image, predictions)
