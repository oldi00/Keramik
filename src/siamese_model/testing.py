import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import process_svg


def test_similarity(model, path_a, path_b):
    print(f"Bild A: {path_a}")
    print(f"Bild B: {path_b}")

    img_a = process_svg(Path(path_a))
    img_b = process_svg(Path(path_b))

    if img_a is None or img_b is None:
        print("Fehler: Eines der Bilder konnte nicht geladen werden.")
        return

    input_a = np.expand_dims(img_a, axis=0)
    input_b = np.expand_dims(img_b, axis=0)

    score = model.predict([input_a, input_b], verbose=0)[0][0]
    prozent = score * 100

    if score > 0.5:
        ergebnis = "MATCH (Gleiche Klasse)"
        farbe = "\033[92m"
    else:
        ergebnis = "DIFFERENT (Unterschiedlich)"
        farbe = "\033[91m"

    reset = "\033[0m"
    print(f"Ähnlichkeits-Score: {score:.5f}")
    print(f"Entscheidung: {farbe}{ergebnis} ({prozent:.2f}% sicher){reset}")

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(img_a.squeeze(), cmap="gray")
    ax[0].set_title("Bild A")
    ax[0].axis("off")
    ax[1].imshow(img_b.squeeze(), cmap="gray")
    ax[1].set_title("Bild B")
    ax[1].axis("off")
    plt.show()
