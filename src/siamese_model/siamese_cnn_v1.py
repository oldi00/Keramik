"""This module defines a siamese CNN architecture for CNN learning."""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from pathlib import Path
from keras.layers import Lambda, Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
import cairosvg
import io
from PIL import Image
import matplotlib.pyplot as plt


MODEL_INPUT_SHAPE = (128, 128, 1)
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30


def process_svg(path):
    try:
        png_bytes = cairosvg.svg2png(
            url=str(path), write_to=None, output_width=512, output_height=512
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        img.thumbnail(IMG_SIZE, Image.Resampling.LANCZOS)

        background = Image.new("RGB", IMG_SIZE, (255, 255, 255))

        offset_x = (IMG_SIZE[0] - img.width) // 2
        offset_y = (IMG_SIZE[1] - img.height) // 2
        background.paste(img, (offset_x, offset_y), mask=img.split()[3])

        arr = np.array(background.convert("L")).astype("float32") / 255.0
        return np.expand_dims(arr, axis=-1)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def load_data(root_dir="data/PoC"):
    X = []
    y = []
    data_path = Path(root_dir)
    class_names = sorted([item.name for item in data_path.iterdir() if item.is_dir()])
    class_map = {name: idx for idx, name in enumerate(class_names)}

    files = list(data_path.rglob("*.svg"))
    print(f"Found {len(files)} SVGs.")

    for file_path in files:
        img_data = process_svg(file_path)
        if img_data is not None:
            X.append(img_data)
            parent_folder = file_path.parent.name
            if parent_folder in class_map:
                y.append(class_map[parent_folder])
            else:
                y.append(0)
    return np.array(X), np.array(y)


def make_pairs(images, labels):
    images_left = []
    images_right = []
    pair_labels = []

    unique_classes = np.unique(labels)
    idx = {i: np.where(labels == i)[0] for i in unique_classes}

    for i in range(len(images)):
        current_img = images[i]
        label = labels[i]

        # 1. Positive Pair
        class_indices = idx[label]
        if len(class_indices) > 1:
            possible = class_indices[class_indices != i]
            pos_idx = np.random.choice(possible)
        else:
            pos_idx = class_indices[0]

        images_left.append(current_img)
        images_right.append(images[pos_idx])
        pair_labels.append(1.0)

        # 2. Negative Pair
        neg_indices = np.where(labels != label)[0]
        if len(neg_indices) > 0:
            neg_idx = np.random.choice(neg_indices)
            images_left.append(current_img)
            images_right.append(images[neg_idx])
            pair_labels.append(0.0)

    left_arr = np.array(images_left)
    right_arr = np.array(images_right)
    lbl_arr = np.array(pair_labels).astype("float32")

    shuffle_indices = np.random.permutation(len(lbl_arr))
    return (
        left_arr[shuffle_indices],
        right_arr[shuffle_indices],
        lbl_arr[shuffle_indices],
    )


def initialize_weights(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def initialize_bias(shape, dtype=None):
    return tf.random.normal(shape, mean=0.5, stddev=0.01, dtype=dtype)


def build_siamese_network(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # REMOVED ALL L2 REGULARIZATION
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

    # Dense Layer (ReLU is safer than Sigmoid for hidden layers)
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


if __name__ == "__main__":
    images, labels = load_data()
    print(f"Images Loaded: {images.shape}")

    images_L, images_R, pair_labels = make_pairs(images, labels)
    print(f"Pairs Created: {images_L.shape}")

    print("Displaying first pair... close window to continue.")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images_L[0].squeeze(), cmap="gray")
    ax[0].set_title("Left Image")
    ax[1].imshow(images_R[0].squeeze(), cmap="gray")
    ax[1].set_title(f"Right Image (Label: {pair_labels[0]})")
    plt.show()

    model = build_siamese_network(MODEL_INPUT_SHAPE)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("\nStarting Training (No L2 Regularization)...")
    history = model.fit(
        [images_L, images_R],
        pair_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
    )


def visualize_predictions(model, images_L, images_R, labels, num_samples=5):
    """
    Picks random pairs, asks the model for a score, and visualizes the result.
    """
    print("\n--- Visualizing Predictions ---")

    indices = np.random.choice(len(images_L), num_samples, replace=False)

    for idx in indices:
        img_a = images_L[idx]
        img_b = images_R[idx]
        true_label = labels[idx]

        input_a = np.expand_dims(img_a, axis=0)
        input_b = np.expand_dims(img_b, axis=0)

        prediction = model.predict([input_a, input_b], verbose=0)[0][0]

        result = "MATCH" if prediction > 0.5 else "DIFFERENT"
        color = "green" if (round(prediction) == true_label) else "red"

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(img_a.squeeze(), cmap="gray")
        ax[0].axis("off")
        ax[0].set_title("Left")

        ax[1].imshow(img_b.squeeze(), cmap="gray")
        ax[1].axis("off")
        ax[1].set_title("Right")

        plt.suptitle(
            f"True: {true_label} | Pred: {prediction:.4f} ({result})",
            color=color,
            fontweight="bold",
        )
        plt.show()


visualize_predictions(model, images_L, images_R, pair_labels)

# Save the model so you don't lose this progress!
model.save("siamese_model_v1.h5")
print("Model saved as siamese_model_v1.h5")
