import xml.etree.ElementTree as ET
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

SVG_NS = "http://www.w3.org/2000/svg"


def clean_svg(src_path, dst_path):
    """Removes ID and scale from the given SVG image."""
    tree = ET.parse(src_path)
    root = tree.getroot()
    for child in list(root):
        if child.tag in [f"{SVG_NS}text", f"{SVG_NS}rect"]:
            root.remove(child)
    tree.write(dst_path)


def load_image(path):
    """Lädt ein Bild, konvertiert es zu Grayscale und resized es."""
    # WICHTIG: Path-Objekt zu String konvertieren für TF
    path = str(path)
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (128, 128))
    return img.numpy()


def make_pairs(images, labels):
    """Erstellt Paare (Positive/Negative) im Speicher."""
    # Sicherstellen, dass wir listenartige Strukturen haben
    images = np.array(images)
    labels = np.array(labels)

    pair_images = []
    pair_labels = []

    # Indexlisten für jede Klasse ermitteln
    unique_classes = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in unique_classes]

    print(f"Generiere Paare aus {len(images)} Bildern...")

    for i in range(len(images)):
        current_img = load_image(images[i])
        label = labels[i]

        # 1. Positives Paar (Gleiche Klasse)
        # idx[label] enthält alle Indizes dieser Klasse
        # Wir müssen sicherstellen, dass der Index im Array bounds ist
        current_class_indices = idx[label]
        j = np.random.choice(current_class_indices)
        pos_img = load_image(images[j])

        pair_images.append([current_img, pos_img])
        pair_labels.append([1])

        # 2. Negatives Paar (Andere Klasse)
        neg_idx = np.where(labels != label)[0]
        if len(neg_idx) > 0:  # Nur wenn es andere Klassen gibt
            k = np.random.choice(neg_idx)
            neg_img = load_image(images[k])

            pair_images.append([current_img, neg_img])
            pair_labels.append([0])

    return np.array(pair_images), np.array(pair_labels)


def euclidean_distance(vectors):
    feats_img1, feats_img2 = vectors
    sum_squared = K.sum(K.square(feats_img1 - feats_img2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))
