"""Extract only relevant 'Typtafeln' from specified PDF files."""

from pathlib import Path
from pypdf import PdfReader, PdfWriter

OUTPUT_DIR = Path("data/Literatur_Typtafeln")

# This dictionary defines which pages contain relevant 'Typtafeln' for every literature resource.
# A tuple represents a range with both numbers being inclusive.
PAGES_TYPTAFELN = {
    "Biegert - 1999 - Römische Töpfereien in der Wetterau": [
        4, 6, 8, 11, 16, 17, 21, 26, 33, 37, 38, 40, 43
    ],
    "Heising - 2007 - Figlinae Mogontiacenses die römischen Töpfereien ": [
        (418, 563)
    ],
    "Heising und Pfahl - 2000 - Der Keramiktyp Niederbieber 3233": [
        2, 10, 11, 13, 19, 21, 23
    ],
    "Oelmann - 1914 - Die Keramik des Kastells Niederbieber": [
        61, (94, 97)
    ],
    "Pirling et al. - 2006 - Die Funde aus den römischen Gräbern von Krefeld-Ge": [
        (321, 341)
    ],
    "Skript Römische Keramik": [
        6, 7, 49, 50, 51, 53, 54, 69, (77, 95)
    ]
}


def main():

    # Create the output dictionary.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for title, pages in PAGES_TYPTAFELN.items():

        reader = PdfReader(f"data/Literatur/{title}.pdf",)
        writer = PdfWriter()

        for index in pages:

            # Single page.
            if isinstance(index, int):
                page = reader.pages[index - 1]
                writer.add_page(page)

            # Range of pages.
            elif isinstance(index, tuple):
                for ind in range(index[0], index[1] + 1):
                    page = reader.pages[ind - 1]
                    writer.add_page(page)

            else:
                print(
                    f"[warning] The page reference '{index}' ({type(index)}) "
                    f"for the resource '{title}' is not valid and will be skipped.")

        with open(OUTPUT_DIR / f"{title}.pdf", "wb") as f:
            writer.write(f)


if __name__ == "__main__":
    main()
