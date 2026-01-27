import matplotlib.pyplot as plt
import numpy as np


def show_first_pair(images_L, images_R, pair_labels):
    """Zeigt das erste Paar zur Kontrolle"""
    print("Displaying first pair... close window to continue.")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images_L[0].squeeze(), cmap="gray")
    ax[0].set_title("Left Image")
    ax[1].imshow(images_R[0].squeeze(), cmap="gray")
    ax[1].set_title(f"Right Image (Label: {pair_labels[0]})")
    plt.show()


def visualize_predictions(model, images_L, images_R, labels, num_samples=5):
    """Zieht zufällige Proben und visualisiert die Vorhersage"""
    print("\n--- Visualizing Predictions ---")
    indices = np.random.choice(len(images_L), num_samples, replace=False)

    for idx in indices:
        img_a = images_L[idx]
        img_b = images_R[idx]
        true_label = labels[idx]

        input_a = np.expand_dims(img_a, axis=0)
        input_b = np.expand_dims(img_b, axis=0)

        prediction = model.predict([input_a, input_b], verbose=0)[0][0]

        result = "MATCH" if prediction > 0.5 else "DIFFERENT"
        color = "green" if (round(prediction) == true_label) else "red"

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(img_a.squeeze(), cmap="gray")
        ax[0].axis("off")
        ax[0].set_title("Left")

        ax[1].imshow(img_b.squeeze(), cmap="gray")
        ax[1].axis("off")
        ax[1].set_title("Right")

        plt.suptitle(
            f"True: {true_label} | Pred: {prediction:.4f} ({result})",
            color=color,
            fontweight="bold",
        )
        plt.show()


def show_similarity_test(img_a, img_b, score, threshold=0.5):
    """
    Visualisiert das Ergebnis eines Einzel-Tests.
    """
    # Text-Logik für den Titel
    percentage = score * 100
    if score > threshold:
        result_text = "MATCH"
        color = "green"
    else:
        result_text = "DIFFERENT"
        color = "red"

    # Plot erstellen
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(img_a.squeeze(), cmap="gray")
    ax[0].set_title("Bild A")
    ax[0].axis("off")

    ax[1].imshow(img_b.squeeze(), cmap="gray")
    ax[1].set_title("Bild B")
    ax[1].axis("off")

    plt.suptitle(
        f"Ergebnis: {result_text}\nScore: {score:.4f} ({percentage:.1f}%)",
        color=color,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
