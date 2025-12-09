"""A collection of visualization functions."""

from src.preprocess import DIR_SHARDS_CLEAN_PNG, DIR_SHARDS_PROFILES, \
    DIR_TYP_SKELETONS, DIR_TYP_CROPS
import cv2
from matplotlib import pyplot as plt


def show_img(title, img, pos):
    """Reusable code to plot an image."""

    plt.subplot(*pos)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def show_preprocess_shards(shard_id):
    """Plots the preprocessing pipeline for shards."""

    img_raw = cv2.imread(f"data/raw/png/{shard_id}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Raw", img_raw, (1, 3, 1))

    img_clean = cv2.imread(DIR_SHARDS_CLEAN_PNG / f"{shard_id}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Clean", img_clean, (1, 3, 2))

    img_profile = cv2.imread(DIR_SHARDS_PROFILES / f"{shard_id}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Profile", img_profile, (1, 3, 3))

    plt.suptitle("Preprocessing Pipeline for Shards")
    plt.tight_layout()
    plt.show()


def show_preprocess_typology(typ_name):
    """Plots the preprocessing pipeline for typology images."""

    img_raw = cv2.imread(f"data/typology/{typ_name}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Raw", img_raw, (1, 3, 1))

    img_skeleton = cv2.imread(DIR_TYP_SKELETONS / f"{typ_name}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Skeleton", img_skeleton, (1, 3, 2))

    img_crop = cv2.imread(DIR_TYP_CROPS / f"{typ_name}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Crop", img_crop, (1, 3, 3))

    plt.suptitle("Preprocessing Pipeline for Typology Images")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_preprocess_shards("10001")
    show_preprocess_typology("Drag33")
