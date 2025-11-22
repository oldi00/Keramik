import xml.etree.ElementTree as ET
import numpy as np


SVG_NS = "http://www.w3.org/2000/svg"


def clean_svg(src_path, dst_path):
    """Removes ID and scale from the given SVG image."""

    tree = ET.parse(src_path)
    root = tree.getroot()

    for child in list(root):
        if child.tag in [f"{SVG_NS}text", f"{SVG_NS}rect"]:
            root.remove(child)

    tree.write(dst_path)


def make_pairs(images, labels):
    images = np.array(images)
    labels = np.array(labels)

    pair_images = []
    pair_labels = []

    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]

    for i in range(len(images)):
        current_img = images[i]
        label = labels[i]

        j = np.random.choice(idx[label])
        pos_img = images[j]

        pair_images.append([current_img, pos_img])
        pair_labels.append([1])  # True label

        neg_idx = np.where(labels != label)[0]
        neg_img = images[np.random.choice(neg_idx)]

        pair_images.append([current_img, neg_img])
        pair_labels.append([0])  # False label

    return (np.array(pair_images), np.array(pair_labels))


def euclidean_distance(vectors):
    pass

def test():
    result = make_pairs(np.array(["img_A", "img_B", "img_C", "img_D", "img_E"]), np.array([0, 1, 0, 2, 1]))

    print(result)

if __name__ == "__main__":
    test()