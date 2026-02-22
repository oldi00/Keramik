"""
Run a streamlit app to interact with the core algorithms via a web interface.

Use the below command to start the app:
>>> streamlit run src/app.py
"""

from src.solver import load_typology_data, find_top_matches
from src.preprocess import preprocess_shard
from src.visuals import get_match_overlay
from src.utils import load_config, apply_transformation
import streamlit as st

st.set_page_config(page_title="Keramik Challenge", layout="wide")


# --- CACHED FUNCTIONS ---

@st.cache_data(show_spinner=False)
def get_cached_config():
    return load_config()


@st.cache_data(show_spinner=False)
def get_cached_typology_data(config):
    return load_typology_data(config)


@st.cache_data(show_spinner=False)
def get_cached_top_matches(profile, typology_data, config):
    return find_top_matches(profile, typology_data, config)


@st.cache_data(show_spinner=False)
def get_cached_overlay(typ_path, dist_map, points):
    return get_match_overlay(typ_path, dist_map, points)


# --- FRAGMENT FUNCTIONS ---

@st.fragment
def render_match_tab(match, i, typology_data):

    col1, col2 = st.columns([3, 2], gap="large")

    with col2:

        # todo: connect to real metadata?

        st.header(f"**{match['name']}**")
        st.caption("Reference: Fasold, *Typentafel* (Page 10)")

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("ICP Error", f"{match.get('icp_error', 0.0):.2f}")
        m_col2.metric("RANSAC Score", f"{match.get('ransac_score', 0.0):.2f}")

        show_ransac = st.toggle(
            "Show RANSAC Overlay",
            value=False,
            key=f"toggle_{i}",
            help="Toggle between ICP (default) and RANSAC alignment visualizations."
        )
        st.button("Save Overlay", key=f"button_{i}", type="primary", width="stretch")

    with col1:

        points_shard = match["points_shard"]
        typ_name = match["name"]

        typ_path = typology_data[typ_name]["path"]
        dist_map = typology_data[typ_name]["dist_map"]

        if not show_ransac:
            icp_points = apply_transformation(points_shard, *match["icp_params"])
            overlay = get_cached_overlay(typ_path, dist_map, icp_points)
        else:
            # todo: add ransac legend?
            ransac_points = apply_transformation(points_shard, *match["ransac_params"])
            overlay = get_cached_overlay(typ_path, dist_map, ransac_points)

        st.image(overlay, width="stretch")


# --- SIDEBAR ---

config = get_cached_config()
config_paths = config["paths"]
config_ransac = config["parameters"]["ransac"]
config_icp = config["parameters"]["icp"]

with st.sidebar:

    st.title("⚙️ Configuration")
    st.caption(
        "Configure the core parameters of the pipeline below, but proceed with caution. "
        "Adjusting the RANSAC and ICP settings will directly impact convergence."
    )

    form = st.form("settings", border=False)

    with form.expander("File Config", expanded=True):

        config_paths["typology_clean"] = st.text_input(
            "Typology Path",
            value=config_paths["typology_clean"],
            help="Path to the clean typology dataset."
        )

    with form.expander("General Algorithm", expanded=True):

        config["parameters"]["top_k"] = st.slider(
            label="Top-K Matches",
            min_value=1,
            max_value=5,
            value=config["parameters"]["top_k"],
        )

        config_ransac["squared_dist_map"] = st.checkbox(
            label="Use Squared Distance Map",
            value=config_ransac["squared_dist_map"],
        )

        config["parameters"]["drop_bottom"] = st.checkbox(
            label="Drop Bottom of Shard",
            value=config["parameters"]["drop_bottom"],
        )

    with form.expander("RANSAC Parameters", expanded=True):

        st.caption(
            "Set boundaries for sampling, scaling, and rotation to balance alignment  "
            "precision against computation time.",
        )

        config_ransac["iterations"] = st.number_input(
            label="Iterations",
            min_value=1000,
            max_value=100000,
            value=config_ransac["iterations"],
        )

        col1, col2 = st.columns(2)
        with col1:
            config_ransac["min_scale"] = st.slider(
                label="Min Scale",
                min_value=0.1,
                max_value=1.5,
                value=config_ransac["min_scale"],
            )
        with col2:
            config_ransac["max_scale"] = st.slider(
                label="Max Scale",
                min_value=config_ransac["min_scale"],
                max_value=2.0,
                value=config_ransac["max_scale"],
            )

        config_ransac["max_rotation_deg"] = st.slider(
            label="Max Rotation (°)",
            min_value=0,
            max_value=15,
            value=config_ransac["max_rotation_deg"],
        )

    with form.expander("ICP Parameters", expanded=True):

        st.caption(
            "Control the fine-alignment refinement steps and strictness of the "
            "convergence thresholds."
        )

        config_icp["max_iterations"] = st.slider(
            label="Maximum Iterations",
            min_value=50,
            max_value=200,
            value=config_icp["max_iterations"],
        )

        config_icp["tolerance"] = st.number_input(
            label="Tolerance",
            min_value=1e-7,
            max_value=1e-4,
            value=float(config_icp["tolerance"]),
            step=1e-7,
            format="%.7f",
        )

    # todo: save config.yml to disk?

    submitted = form.form_submit_button("Apply Settings", type="primary", use_container_width=True)
    form.caption("Hint: Applying settings will re-run the algorithm using the uploaded shard.")

    if submitted:
        st.toast("Settings updated!", icon="✅")

# --- MAIN PAGE ---

st.title("Keramik Challenge 🏺")
st.caption("Upload a shard profile to find the closest matches in the typology database.")

user_file = st.file_uploader("Upload a shard:", type="svg")

if not user_file:
    # todo: add a welcome message
    st.stop()

typology_data = get_cached_typology_data(config)
profile = preprocess_shard(user_file)

with st.spinner("Finding best matches..."):
    top_matches = get_cached_top_matches(profile, typology_data, config)

st.subheader("Top Matching Typologies")

tab_titles = [f"#{i+1}: {match['name']}" for i, match in enumerate(top_matches)]
tabs = st.tabs(tab_titles)

for i, tab in enumerate(tabs):
    with tab:
        match = top_matches[i]
        render_match_tab(match, i, typology_data)
