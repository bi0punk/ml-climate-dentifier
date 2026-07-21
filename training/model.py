import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 150
TIME_LABELS = ["day", "evening", "night"]
WEATHER_LABELS = ["clear", "cloudy", "partly_cloudy"]
NUM_TIME = len(TIME_LABELS)
NUM_WEATHER = len(WEATHER_LABELS)


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    time_out = Dense(NUM_TIME, activation="softmax", name="time")(x)
    weather_out = Dense(NUM_WEATHER, activation="softmax", name="weather")(x)

    model = Model(inputs=inputs, outputs=[time_out, weather_out])
    return model


def build_mobilenetv2(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    time_out = Dense(NUM_TIME, activation="softmax", name="time")(x)
    weather_out = Dense(NUM_WEATHER, activation="softmax", name="weather")(x)

    model = Model(inputs=inputs, outputs=[time_out, weather_out])
    return model


def get_model(backbone="cnn", input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    if backbone == "mobilenetv2":
        return build_mobilenetv2(input_shape)
    return build_cnn(input_shape)


def masked_categorical_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss * mask


def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "time": masked_categorical_crossentropy,
            "weather": masked_categorical_crossentropy,
        },
        loss_weights={"time": 1.0, "weather": 1.0},
        metrics={
            "time": "accuracy",
            "weather": "accuracy",
        },
    )
    return model
