"""This module defines a siamese CNN architecture for CNN learning."""

import tensorflow as tf

# Define the network params
IMG_SHAPE = (128, 128, 1)
BATCH_SIZE = 16
EPOCHS = 20


def get_data(data_format="svg"):
    """Get the input data."""
    if data_format == "svg":
        pass

    elif data_format == "png":
        pass


def preprocess_data():
    """Scale and cut the image data."""
    pass


def build_siamese_network(inputShape, embeddingDim=200):
    inputs = tf.keras.layers.Input(inputShape)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # prepare the final outputs
    pooledOutput = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(embeddingDim)(pooledOutput)
	# build the model
    model = tf.keras.models.Model(inputs, outputs)
	# return the model to the calling function
    return model



if __name__ == "__main__":
    build_siamese_network(IMG_SHAPE)