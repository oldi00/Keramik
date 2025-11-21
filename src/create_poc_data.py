"""Create a subset of the original data to use for a PoC."""

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import pandas as pd


IMG_FORMAT = "svg"  # either 'png' or 'svg'
SVG_CLEAN = True  # removes ID and scale
SVG_NS = "{http://www.w3.org/2000/svg}"  # namespace

TYPES = ["Drag. 32", "Drag. 33", "Drag. 37", "Drag. 18/31"]


def clean_svg(src_path, dst_path):
    """Removes ID and scale from the given SVG image."""

    tree = ET.parse(src_path)
    root = tree.getroot()

    for child in list(root):
        if child.tag in [f"{SVG_NS}text", f"{SVG_NS}rect"]:
            root.remove(child)

    tree.write(dst_path)


def main():

    # Create the PoC directory.
    Path("data/PoC").mkdir(parents=True, exist_ok=True)

    # load the excel file and rename the columns just to make the code more readable.
    df = pd.read_excel("data/Gesamt_DB_export.xlsx")
    df.columns = df.columns.str.replace('Sample.', '', regex=False)

    df = df[["Id", "Typ"]]

    for type_ in TYPES:

        # Some types have a slash in their name. Replace them
        # with a dash to avoid potential path issues.
        folder_path = type_
        if "/" in folder_path:
            folder_path = folder_path.replace("/", "-")

        dst_folder = Path(f"data/PoC/{folder_path}")

        # Delete any previous data.
        if dst_folder.exists():
            shutil.rmtree(dst_folder)

        dst_folder.mkdir(parents=True, exist_ok=True)

        ids = df[df["Typ"] == type_]["Id"]
        print(f"[info] Processing {type_} with {len(ids)} items.")

        for id_ in ids:

            img_name = f"{id_}.{IMG_FORMAT}"
            if IMG_FORMAT == "svg":
                img_name = "recons_" + img_name

            src_path = Path(f"data/{IMG_FORMAT}/{img_name}")
            dst_path = dst_folder / img_name

            if not src_path.exists():
                print(f"[warn] Path '{src_path}' does not exist and will be skipped.")
                continue

            if IMG_FORMAT == "svg" and SVG_CLEAN:
                clean_svg(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_folder)


if __name__ == "__main__":
    main()
