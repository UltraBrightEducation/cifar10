from keras import Sequential
from keras.applications import InceptionV3
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    GlobalMaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    Activation,
    GlobalAveragePooling2D,
    regularizers,
)
from keras.models import Model


def _conv_block(
    x,
    filters,
    activation,
    dropout=0.2,
    weight_decay=1e-4,
    kernel_size=3,
    pool=(2, 2),
    batch_normalization=True,
):
    hidden = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    if batch_normalization:
        hidden = BatchNormalization()(hidden)
    hidden = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(hidden)
    if batch_normalization:
        hidden = BatchNormalization()(hidden)
    hidden = MaxPool2D(pool_size=pool)(hidden)
    hidden = Dropout(dropout)(hidden)
    return hidden


def convnet6(
    input_shape,
    n_classes,
    base_filters,
    activation,
    fc_size,
    dropout,
    classifier_activation,
    weight_decay,
    batch_normalization,
    **kwargs
):
    image = Input(shape=input_shape)
    conv_1 = _conv_block(
        image,
        filters=base_filters,
        activation=activation,
        dropout=dropout,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
    )
    conv_2 = _conv_block(
        conv_1,
        filters=base_filters * 2,
        activation=activation,
        dropout=dropout + 0.1,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
    )
    conv_3 = _conv_block(
        conv_2,
        filters=base_filters * 4,
        activation=activation,
        dropout=dropout + 0.2,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
    )
    # hidden = GlobalMaxPooling2D()(conv_6)
    # conv_4 = Conv2D(filters=base_filters * 4, kernel_size=1, activation=activation)
    hidden = Flatten()(conv_3)

    fc_1 = Dense(fc_size, activation=activation)(hidden)
    predictions = Dense(n_classes, activation=classifier_activation)(fc_1)

    return Model(image, predictions)


def convnet4(
    input_shape,
    n_classes,
    base_filters,
    activation,
    fc_size,
    dropout,
    classifier_activation,
    **kwargs
):
    model = Sequential()
    model.add(Conv2D(base_filters, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation(activation))
    model.add(Conv2D(base_filters, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2))

    model.add(Conv2D(base_filters * 2, (3, 3), padding="same"))
    model.add(Activation(activation))
    model.add(Conv2D(base_filters * 2, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2))

    model.add(Flatten())
    model.add(Dense(fc_size))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes))
    model.add(Activation(classifier_activation))

    return model


def inception_v3(
    input_shape, n_classes, activation, classifier_activation, fc_size, **kwargs
):
    image = Input(shape=input_shape)
    base_model = InceptionV3(input_tensor=image, weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation=activation)(x)
    predictions = Dense(n_classes, activation=classifier_activation)(x)

    for layer in base_model.layers:
        layer.trainable = False

    return Model(image, predictions)
