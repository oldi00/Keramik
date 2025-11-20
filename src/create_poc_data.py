"""Create subset of the original data to use as a PoC."""

import shutil
from pathlib import Path
import pandas as pd


IMG_FORMAT = "png"  # either 'png' or 'svg'
TYPES = ["Drag. 32", "Drag. 33", "Drag. 37", "Drag. 18/31"]


def main():

    Path("data/PoC").mkdir(parents=True, exist_ok=True)

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

        # Delete any existing data.
        if dst_folder.exists():
            shutil.rmtree(dst_folder)

        dst_folder.mkdir(parents=True, exist_ok=True)

        ids = df[df["Typ"] == type_]["Id"]
        for id_ in ids:
            img_name = f"{id_}.{IMG_FORMAT}"
            if IMG_FORMAT == "svg":
                img_name = "recons_" + img_name
            img_path = Path(f"data/{IMG_FORMAT}/{img_name}")
            shutil.copy2(img_path, dst_folder)


if __name__ == "__main__":
    main()
