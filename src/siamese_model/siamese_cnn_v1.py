"""This module defines a siamese CNN architecture for CNN learning."""

import tensorflow as tf
import utils
import numpy as np
from pathlib import Path

# Define the network params
MODEL_INPUT_SHAPE = (128, 128, 1)
BATCH_SIZE = 16
EPOCHS = 20
margin = 1


def get_raw_img_from_path():
    """Get the input data."""

    images = []
    labels = []

    data_path = Path("data/PoC")
    
    # Store all files.
    for i, (_, _, files) in enumerate(data_path.walk()):
        images += files

        # Somehow the index: 0 is used for another folder.
        labels += [i-1]*len(files)
    
    #!TESTING
    images = [np.random.rand(128, 128, 1) for _ in range(50)]
    labels = [f"SVG_IMG {i}" for i in range(50)]

    return images, labels


def preprocess_data(images):
    """Scale and cut the image data."""

    #TODO: Scale and cutting using the cairosvg library.

    #!TESTING
    for img in images:
        pass

    print("Loading fake data")
    return images


def make_pairs(images, labels):
    images = np.array(images)

    #!TESTING
    selection_1 = np.random.choice(images.shape[0], 50)
    images_1 = images[selection_1]

    selection_2 = np.random.choice(images.shape[0], 50)
    images_2 = images[selection_2]

    pair_images = zip(images_1, images_2)
    pair_labels = np.random.choice([0, 1], 50)
    return pair_images, pair_labels


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
    images, labels = get_raw_img_from_path()
    png_images = preprocess_data(images)
    pair_images, pair_gtr_labels = make_pairs(png_images, labels)
    print(pair_images, pair_gtr_labels)
    build_siamese_network()
