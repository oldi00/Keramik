import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt

# --- Imports aus den eigenen Modulen ---
# Wir fügen das aktuelle Verzeichnis zum Pfad hinzu, damit Module gefunden werden
import sys

sys.path.append(str(Path(__file__).parent))

import utils
import solver

# --- CONFIG ---
st.set_page_config(
    page_title="Keramik",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pfad zu den vorverarbeiteten Typologie-Daten (Crops)
# Passen Sie diesen Pfad ggf. an Ihre Ordnerstruktur an (relativ zum Root)
TYPOLOGY_FOLDER = Path("data/processed/typology")

# --- STYLES ---
st.markdown(
    """
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; border: 1px solid #e0e0e0; }
    img { border: 1px solid #ddd; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HELPER FUNCTIONS ---


@st.cache_resource
def load_typology_db():
    """Lädt die Referenz-Datenbank (Typologien) einmalig in den Cache."""
    typology = {}

    if not TYPOLOGY_FOLDER.exists():
        st.error(
            f"Typologie-Ordner nicht gefunden: {TYPOLOGY_FOLDER}. Bitte stellen Sie sicher, dass 'preprocess.py' ausgeführt wurde."
        )
        return {}

    # Alle PNGs im Typologie-Ordner durchsuchen
    files = list(TYPOLOGY_FOLDER.rglob("*.png"))

    for typ_path in files:
        # Punkte und Distanzkarte vorberechnen
        points = utils.get_points(str(typ_path))
        dist_map = utils.get_dist_map(str(typ_path))

        name = utils.normalize_name(typ_path.stem)

        typology[name] = {
            "points": points,
            "dist_map": dist_map,
            "path": str(typ_path),
            "display_name": typ_path.stem,  # Originaler Name für Anzeige
        }

    return typology


def preprocess_uploaded_image(uploaded_file):
    """
    Speichert den Upload temporär, extrahiert das Profil (wie in preprocess.py)
    und gibt den Pfad zum verarbeiteten Bild zurück.
    """
    # 1. Temporäre Datei für den Upload erstellen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_raw:
        tmp_raw.write(uploaded_file.getvalue())
        raw_path = tmp_raw.name

    # 2. Profil extrahieren (Logik aus preprocess.py -> extract_profile_shard)
    # Wir machen das hier direkt mit OpenCV, um File-IO zu minimieren,
    # speichern das Ergebnis aber, da utils.get_points einen Pfad erwartet.
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)

    # Binarisierung & Cleaning
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Kontur finden
    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    # Neues Bild erstellen (weißer Hintergrund, schwarze Kontur)
    shard_profile = np.ones_like(img) * 255
    cv2.drawContours(shard_profile, [largest_contour], -1, (0, 0, 0), thickness=1)

    # 3. Profil-Bild temporär speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_profile:
        cv2.imwrite(tmp_profile.name, shard_profile)
        profile_path = tmp_profile.name

    # Bereinigen: Raw File löschen
    os.remove(raw_path)

    return profile_path, shard_profile


def create_heatmap_figure(typ_path, dist_map, transformed_points):
    """Erstellt die Matplotlib-Figure für die Überlagerung (Visualisierung)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Typologie Hintergrund
    img_typ = cv2.imread(typ_path, cv2.IMREAD_GRAYSCALE)
    img_typ = cv2.erode(
        img_typ, np.ones((3, 3), np.uint8)
    )  # Dicker machen für Sichtbarkeit
    ax.imshow(img_typ, cmap="gray", origin="upper", alpha=0.5)

    # Punkte einfärben nach Distanz (Qualität des Matches)
    points_int = np.rint(transformed_points).astype(int)
    h, w = dist_map.shape
    dists = []

    for p in points_int:
        x, y = p
        val = (
            dist_map[y, x] if 0 <= x < w and 0 <= y < h else 50
        )  # Hohe Strafe wenn außerhalb
        dists.append(val)

    sc = ax.scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        c=dists,
        cmap="RdYlGn_r",  # Grün = gut (geringe Distanz), Rot = schlecht
        edgecolors="none",
        s=15,
        vmin=0,
        vmax=30,
    )

    ax.axis("off")
    plt.tight_layout()
    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.title("🏺 Keramik Klassifizierer")
    st.markdown("---")
    st.header("1. Upload")
    uploaded_file = st.file_uploader(
        "Scherbe hochladen (Bild)", type=["png", "jpg", "jpeg"]
    )

    st.markdown("---")
    st.info(
        """
        **Funktionsweise:**
        Das System extrahiert die Kontur der Scherbe und versucht, 
        sie geometrisch (RANSAC + Chamfer Distance) in die Typenkataloge einzupassen.
        """
    )

# --- MAINPAGE ---
st.title("Typenbestimmung: Geometrisches Matching")

# Datenbank laden
typology_db = load_typology_db()

if uploaded_file is not None:
    # Originalbild anzeigen
    input_image = Image.open(uploaded_file)

    st.write("Verarbeite Bild...")
    progress_bar = st.progress(0)

    # 1. Preprocessing
    profile_path, profile_img_cv = preprocess_uploaded_image(uploaded_file)

    if profile_path is None:
        st.error(
            "Konnte keine klare Kontur auf dem Bild finden. Bitte versuchen Sie ein Bild mit besserem Kontrast."
        )
    else:
        # Punkte für den Solver laden
        points_shard = utils.get_points(
            profile_path, step=3
        )  # step=3 für Geschwindigkeit

        progress_bar.progress(10)

        # 2. Matching Loop
        results = []
        total_types = len(typology_db)

        # Iteration über alle Typen in der DB
        for i, (name, data) in enumerate(typology_db.items()):
            # Der Solver berechnet Score und Transformation
            score, params = solver.solve_matching(
                points_shard,
                data["points"],
                data["dist_map"],
                iterations=2000,  # Reduziert für schnellere UI-Response (Default war 10000)
            )

            results.append(
                {
                    "type": data["display_name"],
                    "score": score,  # Chamfer Distance: Kleiner ist besser (0 ist perfekt)
                    "params": params,
                    "typ_data": data,
                }
            )

            # Progress Update
            if i % 5 == 0:
                progress_bar.progress(10 + int((i / total_types) * 80))

        progress_bar.progress(100)

        # Sortieren: Score aufsteigend (niedriger Score = besseres Match)
        results.sort(key=lambda x: x["score"])
        top_match = results[0]

        # --- ERGEBNISSE ANZEIGEN ---
        st.subheader("📊 Analyseergebnisse")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(input_image, caption="Originalaufnahme", width=300)
            # Optional: Das extrahierte Profil anzeigen zur Kontrolle
            st.image(
                profile_img_cv,
                caption="Extrahiertes Profil (Input für KI)",
                width=300,
                clamp=True,
            )

        with col2:
            # Score in "Konfidenz" umrechnen (nur grobe Schätzung für UI)
            # Chamfer Score < 1.0 ist sehr gut, > 10 ist schlecht.
            score_disp = top_match["score"]
            confidence_color = (
                "green" if score_disp < 5 else "orange" if score_disp < 15 else "red"
            )

            st.markdown(f"### Top Treffer: :{confidence_color}[{top_match['type']}]")
            st.caption(f"Chamfer Score: {score_disp:.4f} (niedriger ist besser)")

            with st.expander("Zeige Top 5 Kandidaten", expanded=True):
                for i, res in enumerate(results[:5]):
                    st.write(f"**#{i+1} {res['type']}** (Score: {res['score']:.2f})")

        st.markdown("---")
        st.subheader("Visuelle Validierung (Geometrischer Fit)")

        # Visualisierung des Top-Matches generieren
        best_typ_data = top_match["typ_data"]
        best_params = top_match["params"]

        # Transformation auf die Scherben-Punkte anwenden
        # Wir laden die Punkte nochmal mit step=1 für schönere Grafik
        points_shard_high_res = utils.get_points(profile_path, step=1)
        transformed_points = solver.apply_transformation(
            points_shard_high_res, *best_params
        )

        # Plot erstellen
        fig = create_heatmap_figure(
            best_typ_data["path"], best_typ_data["dist_map"], transformed_points
        )

        st.pyplot(fig)
        st.info(
            "Grüne Punkte liegen exakt auf der Referenzlinie. Rote Punkte weichen ab."
        )

        # Aufräumen der temporären Datei
        if os.path.exists(profile_path):
            os.remove(profile_path)

else:
    st.info(
        "👋 Willkommen! Bitte laden Sie ein Bild einer Scherbe hoch (Profilansicht oder Zeichnung), um das geometrische Matching zu starten."
    )
