"""Remove specified pages from PDF files."""

import json
from pathlib import Path
from pypdf import PdfReader, PdfWriter

LITERATURE_DIR = "data/Literatur/"
LITERATURE_DIR_CLEAN = "data/Literatur_Clean/"


def main():

    Path(LITERATURE_DIR_CLEAN).mkdir(parents=True, exist_ok=True)

    with open('literature.json', 'r', encoding="utf-8") as f:
        data = json.load(f)

    for title, pages in data.items():

        reader = PdfReader(f"{LITERATURE_DIR}{title}.pdf",)
        writer = PdfWriter()

        for index in pages:

            if isinstance(index, int):
                page = reader.pages[index - 1]
                writer.add_page(page)

            elif isinstance(index, list):
                for ind in range(index[0], index[1] + 1):
                    page = reader.pages[ind - 1]
                    writer.add_page(page)

        with open(f"data/Literatur_Clean/{title}_mod.pdf", "wb") as f:
            writer.write(f)


if __name__ == "__main__":
    main()
