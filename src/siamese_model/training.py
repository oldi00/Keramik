import utils
import model as net
import plotting
import testing
from pathlib import Path
from keras.optimizers import Adam


if __name__ == "__main__":
    # 1. Daten laden
    images, labels = utils.load_data()
    print(f"Images Loaded: {images.shape}")

    # 2. Paare bilden
    images_L, images_R, pair_labels = utils.make_pairs(images, labels)
    print(f"Pairs Created: {images_L.shape}")

    # 3. Sanity Check (Optional)
    plotting.show_first_pair(images_L, images_R, pair_labels)

    # 4. Modell bauen
    siamese_model = net.build_siamese_network(net.MODEL_INPUT_SHAPE)

    optimizer = Adam(learning_rate=net.LEARNING_RATE)
    siamese_model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    print("\nStarting Training")
    siamese_model.summary()

    # 5. Training
    history = siamese_model.fit(
        [images_L, images_R],
        pair_labels,
        batch_size=net.BATCH_SIZE,
        epochs=net.EPOCHS,
        validation_split=0.2,
    )

    # 6. Ergebnisse prüfen
    plotting.visualize_predictions(siamese_model, images_L, images_R, pair_labels)

    # 7. Speichern
    siamese_model.save("siamese_model_v1.h5")
    print("Model saved as siamese_model_v1.h5")

    # 8. (Optional) Manueller Test
    # Passe diese Pfade an deine echten Dateien an
    path1 = "data/PoC/Drag. 32/image_001.svg"
    path2 = "data/PoC/Drag. 32/image_005.svg"

    if Path(path1).exists():
        testing.test_similarity(siamese_model, path1, path2)
