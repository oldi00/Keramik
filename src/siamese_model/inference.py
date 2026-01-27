import numpy as np
import matplotlib.pyplot as plt
import model as net
import utils
import xml.etree.ElementTree as ET
from pathlib import Path


SVG_NS = "{http://www.w3.org/2000/svg}"


def load_trained_model(weights_path):
    """
    Builds the architecture and loads the saved weights.
    Using 'load_weights' is often safer than 'load_model' for
    custom architectures with Lambda layers.
    """
    print("Loading model architecture...")
    # 1. Build the empty architecture (must match training exactly)
    model = net.build_siamese_network(net.MODEL_INPUT_SHAPE)

    # 2. Load the calculated weights
    print(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path)

    return model


def predict_single_pair(model, file_path_a, file_path_b):
    """
    Loads two SVGs, processes them, and prints the similarity score.
    """
    path_a = Path(file_path_a)
    path_b = Path(file_path_b)

    if not path_a.exists() or not path_b.exists():
        print(f"❌ Error: One or both files not found:\n  {path_a}\n  {path_b}")
        return

    print("\nProcessing images...")
    # 1. Reuse the exact same processing utils from training
    img_a = utils.process_svg(path_a)
    img_b = utils.process_svg(path_b)

    if img_a is None or img_b is None:
        print("❌ Error: Failed to process SVG (check if files are valid).")
        return

    # 2. Add Batch Dimension: (128, 128, 1) -> (1, 128, 128, 1)
    input_a = np.expand_dims(img_a, axis=0)
    input_b = np.expand_dims(img_b, axis=0)

    # 3. Predict
    # verbose=0 keeps the terminal clean
    score = model.predict([input_a, input_b], verbose=0)[0][0]
    percentage = score * 100

    # 4. Output Result
    if score > 0.5:
        result = "MATCH (Same)"
        color_code = "\033[92m"  # Green
    else:
        result = "DIFFERENT"
        color_code = "\033[91m"  # Red

    reset_code = "\033[0m"

    print("-" * 30)
    print(f"File A: {path_a.name}")
    print(f"File B: {path_b.name}")
    print(f"Score : {score:.5f}")
    print(f"Result: {color_code}{result} ({percentage:.2f}%){reset_code}")
    print("-" * 30)

    # 5. Visual Confirmation
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img_a.squeeze(), cmap="gray")
    ax[0].set_title(f"A: {path_a.name}")
    ax[0].axis("off")

    ax[1].imshow(img_b.squeeze(), cmap="gray")
    ax[1].set_title(f"B: {path_b.name}")
    ax[1].axis("off")

    plt.suptitle(
        f"Prediction: {result} ({percentage:.1f}%)",
        fontsize=14,
        color="green" if score > 0.5 else "red",
    )
    plt.show()


def clean_svg(src_path, dst_path):
    """Removes ID and scale from the given SVG image."""

    tree = ET.parse(src_path)
    root = tree.getroot()

    for child in list(root):
        if child.tag in [f"{SVG_NS}text", f"{SVG_NS}rect"]:
            root.remove(child)

    tree.write(dst_path)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Define paths
    WEIGHTS_FILE = "src/siamese_model/siamese_model_v1.h5"

    # CHANGE THESE to the files you want to test!
    TEST_IMAGE_1 = "data/svg/recons_10004.svg"
    TEST_IMAGE_2 = "test_2.svg"

    # 2. Load Model Once
    if Path(WEIGHTS_FILE).exists():
        siamese_model = load_trained_model(WEIGHTS_FILE)

        # 3. Run Prediction
        predict_single_pair(siamese_model, TEST_IMAGE_1, TEST_IMAGE_2)

        # You can add more comparisons here easily:
        # predict_single_pair(siamese_model, "path/to/other.svg", "path/to/another.svg")

    else:
        print(f"Error: Weights file '{WEIGHTS_FILE}' not found. Run train.py first.")
