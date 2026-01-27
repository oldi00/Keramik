import streamlit as st
from PIL import Image

# --- CONFIG ---
st.set_page_config(
    page_title="Keramik",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLES ---
st.markdown(
    """
    <style>
    .main {
        background-color: #fcfcfc;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    /* Rahmen um Bilder für besseren Kontrast */
    img {
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dummy_ground_truth(type_name):
    """Simuliert das Laden eines Katalogbildes basierend auf dem Typ."""
    return Image.new("RGB", (300, 300), color=(200, 200, 200))


def blend_images(image1, image2, alpha):
    """Überlagert zwei Bilder basierend auf dem Alpha-Wert (0.0 - 1.0)."""
    image1 = image1.convert("RGBA")
    image2 = image2.convert("RGBA").resize(image1.size)
    return Image.blend(image1, image2, alpha)


# --- SIDEBAR ---
with st.sidebar:
    st.title("🏺 Keramik Klassifizierer")
    st.markdown("---")

    st.header("1. Upload")
    uploaded_file = st.file_uploader(
        "Scherbe hochladen (.png, .svg, .jpg)", type=["png", "jpg", "svg"]
    )

    st.markdown("---")
    st.header("2. Analyse-Einstellungen")
    overlay_opacity = st.slider(
        "Overlay Transparenz",
        0.0,
        1.0,
        0.5,
        help="Verschiebe den Regler, um zwischen Scherbe und Katalogbild zu überblenden.",
    )

    st.info(
        """
    **Anleitung:**
    1. Bild der Scherbe hochladen (idealerweise freigestellt).
    2. KI berechnet die Top-Typen.
    3. Unten visuell vergleichen.
    """
    )

# --- MAINPAGE ---
st.title("Typenbestimmung: Römische Keramik (2.-3. Jh.)")

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    # --- SIMULIERTER ALGORITHMUS ---
    # Example Output:
    predictions = [
        {"type": "Dragendorff 37", "prob": 0.88, "desc": "Bilderschüssel, südgallisch"},
        {"type": "Dragendorff 18/31", "prob": 0.08, "desc": "Teller, Übergangsform"},
        {"type": "Dragendorff 27", "prob": 0.02, "desc": "Napf mit profilierter Wand"},
        {"type": "Curle 11", "prob": 0.01, "desc": "Teller mit Randlippe"},
        {"type": "Déchelette 67", "prob": 0.01, "desc": "Becher, verziert"},
    ]

    top_match = predictions[0]

    st.subheader("📊 Analyseergebnisse")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(input_image, caption="Hochgeladene Scherbe", width="stretch")

    with col2:
        st.success(
            f"Höchste Wahrscheinlichkeit: **{top_match['type']}** ({top_match['prob']*100:.1f}%)"
        )

        with st.expander("Zeige alle Top 5 Kandidaten", expanded=True):
            for i, pred in enumerate(predictions):
                cols = st.columns([1, 4, 1])
                cols[0].write(f"#{i+1}")
                cols[1].progress(int(pred["prob"] * 100))
                cols[2].write(f"**{pred['type']}**")
                st.caption(f"{pred['desc']}")

    st.markdown("---")

    st.subheader("Visuelle Validierung")
    st.markdown("Vergleichen Sie die Kontur der Scherbe mit dem Katalog-Typ.")

    tab1, tab2, tab3 = st.tabs(["Überlagerung", "Nebeneinander", "Katalog Info"])

    # Ground Truth
    gt_image = load_dummy_ground_truth(top_match["type"])

    with tab1:
        # Überlappung
        col_overlay, col_ctrl = st.columns([3, 1])
        with col_overlay:
            # Bild Mischen basierend auf Slider in der Sidebar
            blended_img = blend_images(input_image, gt_image, overlay_opacity)
            st.image(
                blended_img,
                caption=f"Überlagerung: Scherbe vs. {top_match['type']}",
                width="stretch",
            )
        with col_ctrl:
            st.markdown("**Hilfe**")
            st.write(
                "Nutzen Sie den Slider in der linken Sidebar, um die Transparenz zu ändern."
            )
            st.write(f"Aktueller Fokus: **{top_match['type']}**")

    with tab2:
        c1, c2 = st.columns(2)
        c1.image(input_image, caption="Ihre Scherbe")
        c2.image(gt_image, caption=f"Katalog: {top_match['type']}")

    with tab3:
        st.write(f"**Typ:** {top_match['type']}")
        st.write(f"**Beschreibung:** {top_match['desc']}")
        st.write("Datierung: ca. 120 - 180 n. Chr.")
        st.write("Referenz ID: #8821A")

else:
    # --- LANDING PAGE (Nichts hochgeladen) ---
    st.info(
        "👋 Willkommen! Bitte laden Sie eine Scherben-Aufnahme in der Sidebar hoch, um zu beginnen."
    )

    # Beispielbilder anzeigen?
    st.write("Beispiel für unterstützte Aufnahmen:")
    c1, c2, c3 = st.columns(3)
    c1.markdown("🟦 **Profilzeichnung**")
    c2.markdown("📷 **Foto (Draufsicht)**")
    c3.markdown("📐 **Foto (Schnitt)**")
