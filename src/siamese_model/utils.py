import numpy as np
import cairosvg
import io
from model import IMG_SIZE
from PIL import Image
from pathlib import Path


def process_svg(path):
    """Konvertiert SVG zu normalisiertem Numpy Array (128,128,1)"""
    try:
        # SVG rendern (groß für Qualität)
        png_bytes = cairosvg.svg2png(
            url=str(path), write_to=None, output_width=512, output_height=512
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        # Resizing
        img.thumbnail(IMG_SIZE, Image.Resampling.LANCZOS)

        # Weißer Hintergrund
        background = Image.new("RGB", IMG_SIZE, (255, 255, 255))

        # Zentrieren
        offset_x = (IMG_SIZE[0] - img.width) // 2
        offset_y = (IMG_SIZE[1] - img.height) // 2
        background.paste(img, (offset_x, offset_y), mask=img.split()[3])

        # Normalisieren & Grayscale
        arr = np.array(background.convert("L")).astype("float32") / 255.0
        return np.expand_dims(arr, axis=-1)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def load_data(root_dir="data/PoC"):
    """Lädt alle SVGs und Labels"""
    X = []
    y = []
    data_path = Path(root_dir)

    # Klassen finden
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
                y.append(0)  # Fallback

    return np.array(X), np.array(y)


def make_pairs(images, labels):
    """Erstellt 50/50 Positive/Negative Paare"""
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
