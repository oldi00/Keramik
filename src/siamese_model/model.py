import tensorflow as tf
import tensorflow.keras.backend as K  # type: ignore
from keras.layers import Lambda, Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential


# Bild-Einstellungen
IMG_SIZE = (128, 128)
MODEL_INPUT_SHAPE = (128, 128, 1)

# Training-Einstellungen
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001


def initialize_weights(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def initialize_bias(shape, dtype=None):
    return tf.random.normal(shape, mean=0.5, stddev=0.01, dtype=dtype)


def build_siamese_network(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()

    # Conv 1
    model.add(Conv2D(64, (10, 10), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    # Conv 2
    model.add(Conv2D(128, (7, 7), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    # Conv 3
    model.add(Conv2D(128, (4, 4), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())

    # Dense Layer
    model.add(Dense(4096, activation="sigmoid"))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Distance Layer
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Prediction
    prediction = Dense(1, activation="sigmoid")(L1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net
