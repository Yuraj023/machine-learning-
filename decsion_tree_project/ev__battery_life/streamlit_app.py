"""
EV Battery Failure Prediction Dashboard
=========================================
A production-quality Streamlit application for predicting EV battery
failure using a trained Decision Tree Classifier.

Run with:
    streamlit run streamlit_app.py

Author: Yuraj Chauhan
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configure dark theme for Matplotlib/Seaborn plots
plt.style.use("dark_background")

# --------------------------------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed_data")


def _find_first_existing(candidates: list) -> str:
    """Return the first path in `candidates` that exists on disk, else the first candidate."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MODEL_PATH = _find_first_existing([
    os.path.join(MODEL_DIR, "decision_tree_model.joblib"),
    os.path.join(MODEL_DIR, "best_decision_tree.pkl"),
    os.path.join(MODEL_DIR, "decision_tree_model.pkl"),
])

FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")

_dataset_candidates = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "cleaned_ev_battery_dataset*.csv")))
DATASET_PATH = _dataset_candidates[0] if _dataset_candidates else os.path.join(
    PROCESSED_DATA_DIR, "cleaned_ev_battery_dataset.csv"
)

TARGET_COLUMN = "battery_failure"

# --------------------------------------------------------------------------
# BRAND TO MODEL & CATEGORICAL CONFIGURATION
# --------------------------------------------------------------------------
BRAND_MODEL_MAP = {
    "Audi": ["Q4 e-tron", "e-tron", "e-tron GT"],
    "BMW": ["i4", "i7", "iX"],
    "BYD": ["Atto 3", "Dolphin", "Han", "Song Plus", "Tang"],
    "Ford": ["F-150 Lightning", "Mustang Mach-E"],
    "GM": ["Cadillac Lyriq", "Chevrolet Bolt", "Silverado EV"],
    "Honda": ["Prologue", "e:Ny1"],
    "Hyundai": ["Ioniq 5", "Ioniq 6", "Kona Electric"],
    "Kia": ["EV6", "EV9", "Niro EV"],
    "Lucid": ["Air", "Gravity"],
    "MG": ["Comet EV", "MG4", "ZS EV"],
    "Mahindra": ["BE 6", "XUV400"],
    "Mercedes": ["EQB", "EQE", "EQS"],
    "Nissan": ["Ariya", "Leaf"],
    "Renault": ["Megane E-Tech", "Zoe"],
    "Rivian": ["R1S", "R1T"],
    "Tata": ["Nexon EV", "Punch EV", "Tiago EV"],
    "Tesla": ["Model 3", "Model S", "Model X", "Model Y"],
    "Toyota": ["bZ3", "bZ4X"],
    "Volkswagen": ["ID.3", "ID.4", "ID.Buzz"],
    "Volvo": ["EX30", "EX90", "XC40 Recharge"],
}

ALL_BRANDS = sorted(list(BRAND_MODEL_MAP.keys()))

CATEGORICAL_DROPDOWNS = {
    "vehicle_brand": {
        "label": "Vehicle Brand",
        "options": ALL_BRANDS,
        "prefix": "vehicle_brand_",
        "section": "Vehicle Information",
    },
    "vehicle_model": {
        "label": "Vehicle Model",
        "options": [],
        "prefix": "vehicle_model_",
        "section": "Vehicle Information",
    },
    "vehicle_type": {
        "label": "Vehicle Type",
        "options": ["Crossover", "Hatchback", "SUV", "Sedan", "Truck", "Van"],
        "prefix": "vehicle_type_",
        "section": "Vehicle Information",
    },
    "battery_manufacturer": {
        "label": "Battery Manufacturer",
        "options": [
            "BYD Battery", "CATL", "Envision AESC", "Guoxuan",
            "LG Energy Solution", "Northvolt", "Panasonic", "SK On", "Samsung SDI"
        ],
        "prefix": "battery_manufacturer_",
        "section": "Battery Information",
    },
    "battery_chemistry": {
        "label": "Battery Chemistry",
        "options": ["LFP", "LMO", "LTO", "NCA", "NMC"],
        "prefix": "battery_chemistry_",
        "section": "Battery Information",
    },
    "drive_type": {
        "label": "Drive Type",
        "options": ["AWD", "FWD", "RWD"],
        "prefix": "drive_type_",
        "section": "Driving Behaviour",
    },
    "fleet_or_private": {
        "label": "Ownership Type",
        "options": ["Fleet", "Private"],
        "prefix": "fleet_or_private_",
        "section": "Vehicle Information",
    },
    "terrain_type": {
        "label": "Terrain Type",
        "options": ["Coastal", "Desert", "Flat", "Hilly", "Mountainous"],
        "prefix": "terrain_type_",
        "section": "Environmental Conditions",
    },
}

DUMMY_PREFIXES = tuple(cfg["prefix"] for cfg in CATEGORICAL_DROPDOWNS.values())

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Battery Failure Prediction Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# HIGH-CONTRAST BLACK & DARK THEME CUSTOM CSS
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall Dark Background & Default Text */
    .stApp {
        background-color: #090D16 !important;
        color: #F8FAFC !important;
    }

    /* Force Bright Legible Text on All Markdown, Labels, Headings & Form Controls */
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp div[data-testid="stMarkdownContainer"] p,
    .stApp .stRadio label, .stApp .stSelectbox label, .stApp .stSlider label, .stApp .stNumberInput label {
        color: #F8FAFC !important;
    }

    /* Top Navigation Header Box */
    .nav-header-box {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        padding: 1.25rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .nav-header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #3B82F6 !important;
        margin: 0;
    }

    .nav-header-subtitle {
        font-size: 0.9rem;
        color: #9CA3AF !important;
        margin: 0;
    }

    /* Page Main Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC !important;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-size: 1rem;
        color: #94A3B8 !important;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Dark Metric Cards */
    .metric-card-container {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 14px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    .metric-card-container .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #60A5FA !important;
        margin-bottom: 0.2rem;
    }

    .metric-card-container .label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section Headers */
    .section-header {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        padding: 0.65rem 1.1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #F8FAFC !important;
        margin: 1.4rem 0 1rem 0;
    }

    /* Status Result Cards */
    .result-card-success {
        background-color: #064E3B !important;
        border: 1px solid #065F46 !important;
        border-left: 6px solid #10B981 !important;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #ECFDF5 !important;
    }

    .result-card-success h4, .result-card-success p {
        color: #ECFDF5 !important;
    }

    .result-card-error {
        background-color: #7F1D1D !important;
        border: 1px solid #991B1B !important;
        border-left: 6px solid #EF4444 !important;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #FEF2F2 !important;
    }

    .result-card-error h4, .result-card-error p {
        color: #FEF2F2 !important;
    }

    /* Action Buttons */
    div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1.4rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1F2937 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }

    /* Inputs & Form Containers */
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        border-color: #374151 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280 !important;
        padding-top: 2rem;
        padding-bottom: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        border-top: 1px solid #1F2937;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load trained Decision Tree model using joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load cleaned dataset used for training and analytics."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    df = pd.read_csv(path)
    for c in df.columns:
        if c.lower() == "battery_failure":
            global TARGET_COLUMN
            TARGET_COLUMN = c
            break
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str) -> pd.DataFrame:
    """Load precomputed feature importance scores."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature importance file not found at: {path}")
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "feature" not in cols or not any("importance" in c for c in cols):
        raise ValueError(
            "feature_importance.csv must contain a 'Feature' column and an 'Importance' column."
        )
    return df


def safe_load_all():
    """Attempt to load model, dataset, and feature importance file."""
    results = {"model": None, "dataset": None, "feat_imp": None, "errors": []}

    try:
        with st.spinner("Loading Decision Tree Model..."):
            results["model"] = load_model(MODEL_PATH)
    except Exception as e:
        results["errors"].append(f"Model load failed: {e}")

    try:
        results["dataset"] = load_dataset(DATASET_PATH)
    except Exception as e:
        results["errors"].append(f"Dataset load failed: {e}")

    try:
        results["feat_imp"] = load_feature_importance(FEATURE_IMPORTANCE_PATH)
    except Exception as e:
        results["errors"].append(f"Feature importance load failed: {e}")

    return results


RESOURCES = safe_load_all()
MODEL = RESOURCES["model"]
DATASET = RESOURCES["dataset"]
FEATURE_IMPORTANCE = RESOURCES["feat_imp"]


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def get_continuous_feature_columns(df: pd.DataFrame):
    """Return continuous numerical telemetry columns (excluding target & dummies)."""
    if df is None:
        return []
    cols = []
    for c in df.columns:
        if c == TARGET_COLUMN:
            continue
        if c.startswith(DUMMY_PREFIXES):
            continue
        cols.append(c)
    return cols


SECTION_KEYWORDS = {
    "Vehicle Information": ["vehicle", "model", "make", "year", "mileage", "odometer", "vin"],
    "Battery Information": ["battery", "capacity", "voltage", "cell", "soh", "soc", "cycle", "resistance"],
    "Charging Behaviour": ["charge", "charging", "fast_charge", "charger"],
    "Driving Behaviour": ["drive", "driving", "speed", "trip", "distance", "brake", "accel"],
    "Environmental Conditions": ["temp", "temperature", "humidity", "climate", "weather", "ambient", "dust", "altitude"],
    "Maintenance & Diagnostics": ["maintenance", "service", "repair", "inspection", "warranty", "fault", "warning", "bms", "stress", "aging", "thermal"],
}


def assign_section(col_name: str) -> str:
    """Assign numerical feature column to UI section."""
    col_lower = col_name.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in col_lower for kw in keywords):
            return section
    return "Maintenance & Diagnostics"


def render_metric_card(label: str, value: str):
    """Render dark styled HTML metric card."""
    st.markdown(
        f"""
        <div class="metric-card-container">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_recommendations(prediction: int) -> list:
    """Return actionable recommendations based on prediction outcome."""
    if prediction == 0:
        return [
            "Continue routine battery health inspections every 3 to 6 months.",
            "Maintain optimal daily charging threshold between 20% and 80%.",
            "Avoid consecutive high-power fast charging sessions when possible.",
            "Park vehicle in shaded or climate-controlled areas during extreme summer heat.",
        ]
    return [
        "Schedule an immediate comprehensive battery pack diagnostic with an authorized service center.",
        "Suspend fast-charging operations until individual cell voltage imbalance is inspected.",
        "Monitor live BMS telemetry for abnormal thermal rise or rapid voltage degradation.",
        "Review historical charging log files and discharge depth records.",
        "Initiate a full Battery Management System (BMS) cell balancing procedure.",
    ]


# --------------------------------------------------------------------------
# NAVIGATION STRUCTURE
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="nav-header-box">
        <h1 class="nav-header-title"> EV Battery AI Platform</h1>
        <p class="nav-header-subtitle">Production Decision Tree Classifier for EV Battery Failure Risk Assessment</p>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE = st.radio(
    "Navigate Dashboard",
    [
        " Home Overview",
        " Failure Prediction",
        " Telemetry Analytics",
        " Feature Importance",
        " About Project",
    ],
    horizontal=True,
    key="top_navbar",
)

st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0; border: 0; border-top: 1px solid #1F2937;'>", unsafe_allow_html=True)

# Sidebar System Metadata
st.sidebar.markdown("###  System Status")
st.sidebar.markdown("---")
st.sidebar.info(
    "**ML Engine:** Decision Tree Classifier\n\n"
    "**Dataset:** Telemetry Cleaned EV Dataset\n\n"
    "**Developer:** Yuraj Chauhan"
)

if RESOURCES["errors"]:
    with st.sidebar.expander(" Load Warnings", expanded=False):
        for err in RESOURCES["errors"]:
            st.warning(err)


# --------------------------------------------------------------------------
# HOME PAGE OVERVIEW
# --------------------------------------------------------------------------
def render_home():
    st.markdown('<div class="main-header"> Home Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An intelligent Decision Tree platform for real-time electric vehicle '
        'battery health prediction and telemetry analysis.</div>',
        unsafe_allow_html=True,
    )

    if DATASET is not None:
        n_records = DATASET.shape[0]
        n_features = len(get_continuous_feature_columns(DATASET))
    else:
        n_records, n_features = "N/A", "N/A"

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Telemetry Records", f"{n_records:,}" if isinstance(n_records, int) else str(n_records))
    with cols[1]:
        render_metric_card("Continuous Telemetry Features", str(n_features))
    with cols[2]:
        render_metric_card("Target Variable", "Battery Failure")
    with cols[3]:
        render_metric_card("ML Engine", "Decision Tree")

    st.markdown('<div class="section-header"> Executive Summary</div>', unsafe_allow_html=True)
    st.write(
        """
        This production dashboard leverages a **Decision Tree Classifier** trained on high-frequency EV telemetry data
        to predict potential battery failure risks before critical hardware damage occurs.
        Inputs combine vehicle telemetry, cell degradation indices, charging behaviour, environmental stress,
        and maintenance events.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header"> Model Specifications</div>', unsafe_allow_html=True)
        st.write("• **Algorithm:** Decision Tree Classifier (`sklearn.tree`)")
        st.write("• **Optimization:** Hyperparameter tuned for balanced Precision & Recall")
        st.write(f"• **Target Metric:** `{TARGET_COLUMN}` (0: Healthy, 1: Failure Risk)")
    with col_b:
        st.markdown('<div class="section-header"> Data Pipeline</div>', unsafe_allow_html=True)
        if DATASET is not None:
            st.write(f"• **Cleaned Telemetry Size:** {DATASET.shape[0]:,} samples")
            st.write(f"• **Total Features (incl. Categoricals):** {DATASET.shape[1]}")
        else:
            st.warning("Dataset missing.")

    st.markdown('<div class="section-header"> Project Author</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# BATTERY PREDICTION PAGE
# --------------------------------------------------------------------------
def render_prediction():
    st.markdown('<div class="main-header"> Battery Failure Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Select vehicle specifications and operational telemetry parameters to generate an instant battery failure risk assessment.</div>',
        unsafe_allow_html=True,
    )

    if MODEL is None or DATASET is None:
        st.error(" Model or dataset unavailable. Check sidebar notifications.")
        return

    continuous_cols = get_continuous_feature_columns(DATASET)
    if not continuous_cols:
        st.error("No continuous telemetry features detected.")
        return

    st.markdown('<div class="section-header"> Vehicle & Pack Specifications</div>', unsafe_allow_html=True)
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    with col_b1:
        selected_brand = st.selectbox(
            "Vehicle Brand",
            ALL_BRANDS,
            index=ALL_BRANDS.index("Tesla") if "Tesla" in ALL_BRANDS else 0,
            key="sel_brand"
        )

    available_models = BRAND_MODEL_MAP.get(selected_brand, [])
    with col_b2:
        selected_model = st.selectbox(
            "Vehicle Model",
            available_models if available_models else ["Standard"],
            key="sel_model"
        )

    with col_b3:
        selected_type = st.selectbox(
            "Vehicle Type",
            CATEGORICAL_DROPDOWNS["vehicle_type"]["options"],
            index=2,
            key="sel_type"
        )

    with col_b4:
        selected_ownership = st.selectbox(
            "Ownership Type",
            CATEGORICAL_DROPDOWNS["fleet_or_private"]["options"],
            index=1,
            key="sel_ownership"
        )

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        selected_manufacturer = st.selectbox(
            "Battery Manufacturer",
            CATEGORICAL_DROPDOWNS["battery_manufacturer"]["options"],
            index=4,
            key="sel_mfr"
        )
    with col_c2:
        selected_chemistry = st.selectbox(
            "Battery Chemistry",
            CATEGORICAL_DROPDOWNS["battery_chemistry"]["options"],
            index=4,
            key="sel_chem"
        )
    with col_c3:
        selected_drive = st.selectbox(
            "Drive Type",
            CATEGORICAL_DROPDOWNS["drive_type"]["options"],
            index=0,
            key="sel_drive"
        )
    with col_c4:
        selected_terrain = st.selectbox(
            "Primary Terrain",
            CATEGORICAL_DROPDOWNS["terrain_type"]["options"],
            index=2,
            key="sel_terrain"
        )

    grouped = {}
    for col in continuous_cols:
        sec = assign_section(col)
        grouped.setdefault(sec, []).append(col)

    section_order = [
        "Vehicle Information",
        "Battery Information",
        "Charging Behaviour",
        "Driving Behaviour",
        "Environmental Conditions",
        "Maintenance & Diagnostics",
    ]

    user_numeric_inputs = {}

    with st.form("prediction_form"):
        for section in section_order:
            if section not in grouped:
                continue
            st.markdown(f'<div class="section-header"> {section} Telemetry</div>', unsafe_allow_html=True)
            sec_cols = grouped[section]
            w_cols = st.columns(2)
            for i, col in enumerate(sec_cols):
                series = DATASET[col]
                col_lower = col.lower()
                clean_title = col.replace("_", " ").title()

                min_val = float(series.min())
                max_val = float(series.max())
                median_val = float(series.median())

                with w_cols[i % 2]:
                    if "percent" in col_lower or "ratio" in col_lower or "score" in col_lower or "soc" in col_lower or "soh" in col_lower:
                        user_numeric_inputs[col] = st.slider(
                            clean_title,
                            min_value=round(min_val, 2),
                            max_value=round(max_val, 2),
                            value=round(median_val, 2),
                            key=f"num_{col}"
                        )
                    else:
                        user_numeric_inputs[col] = st.number_input(
                            clean_title,
                            min_value=round(min_val, 2),
                            max_value=round(max_val, 2),
                            value=round(median_val, 2),
                            key=f"num_{col}"
                        )

        submitted = st.form_submit_button(" Predict Battery Failure Risk", use_container_width=True)

    if submitted:
        try:
            model_features = list(MODEL.feature_names_in_) if hasattr(MODEL, "feature_names_in_") else list(DATASET.columns)
            input_dict = {feat: 0.0 for feat in model_features}

            for col, val in user_numeric_inputs.items():
                if col in input_dict:
                    input_dict[col] = float(val)

            categorical_selections = [
                ("vehicle_brand_", selected_brand),
                ("vehicle_model_", selected_model),
                ("vehicle_type_", selected_type),
                ("battery_manufacturer_", selected_manufacturer),
                ("battery_chemistry_", selected_chemistry),
                ("drive_type_", selected_drive),
                ("fleet_or_private_", selected_ownership),
                ("terrain_type_", selected_terrain),
            ]

            for prefix, val in categorical_selections:
                target_onehot_col = f"{prefix}{val}"
                if target_onehot_col in input_dict:
                    input_dict[target_onehot_col] = 1.0

            input_df = pd.DataFrame([input_dict])[model_features]

            prediction = int(MODEL.predict(input_df)[0])
            probabilities = MODEL.predict_proba(input_df)[0]
            confidence = max(probabilities) * 100

            st.markdown("---")
            st.markdown('<div class="section-header"> Diagnostic Prediction Result</div>', unsafe_allow_html=True)

            result_cols = st.columns(3)
            with result_cols[0]:
                render_metric_card("Status", "FAILURE RISK DETECTED" if prediction == 1 else "HEALTHY")
            with result_cols[1]:
                render_metric_card("Confidence", f"{confidence:.2f}%")
            with result_cols[2]:
                render_metric_card(
                    "Failure Risk Probability",
                    f"{probabilities[1]*100:.2f}%" if len(probabilities) > 1 else "N/A",
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 0:
                st.markdown(
                    '<div class="result-card-success"><h4> Battery Health: Normal / Safe</h4>'
                    "<p>The Decision Tree model predicts that the battery pack is operating within healthy parameters.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card-error"><h4> Battery Health: Elevated Failure Risk</h4>'
                    "<p>Degradation signals detected! The model predicts high probability of battery failure. Inspect recommendations below.</p></div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-header"> Recommended Actions</div>', unsafe_allow_html=True)
            for rec in get_recommendations(prediction):
                st.write(f"- {rec}")

        except Exception as e:
            st.error(f" Prediction error: {e}")


# --------------------------------------------------------------------------
# TELEMETRY ANALYTICS PAGE
# --------------------------------------------------------------------------
def render_analytics():
    st.markdown('<div class="main-header"> Telemetry Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Comprehensive statistical analysis, distribution plots, and correlation modeling of EV battery telemetry.</div>',
        unsafe_allow_html=True,
    )

    if DATASET is None:
        st.error("Dataset not available.")
        return

    df = DATASET.copy()
    continuous_cols = get_continuous_feature_columns(df)

    st.markdown('<div class="section-header"> Telemetry Overview</div>', unsafe_allow_html=True)
    overview_cols = st.columns(4)
    with overview_cols[0]:
        render_metric_card("Total Samples", f"{df.shape[0]:,}")
    with overview_cols[1]:
        render_metric_card("Telemetry Features", str(len(continuous_cols)))
    with overview_cols[2]:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        render_metric_card("Memory Footprint", f"{mem_mb:.2f} MB")
    with overview_cols[3]:
        render_metric_card("Missing Values", str(int(df.isnull().sum().sum())))

    if TARGET_COLUMN in df.columns:
        st.markdown('<div class="section-header"> Target Distribution (Battery Failure)</div>', unsafe_allow_html=True)
        dist_cols = st.columns(2)
        counts = df[TARGET_COLUMN].value_counts()
        labels_map = {0: "Healthy (0)", 1: "Failure Risk (1)"}
        formatted_labels = [labels_map.get(idx, str(idx)) for idx in counts.index]

        with dist_cols[0]:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(formatted_labels, counts.values, color=["#3B82F6", "#EF4444"], width=0.5)
            ax.set_ylabel("Count of Vehicles")
            ax.set_title("Class Frequency", fontsize=11, fontweight="bold", color="#F8FAFC")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#F8FAFC")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)

        with dist_cols[1]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                counts.values,
                labels=formatted_labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#3B82F6", "#EF4444"],
                explode=(0.04, 0),
                textprops={'color': "#F8FAFC"}
            )
            ax.set_title("Target Share (%)", fontsize=11, fontweight="bold", color="#F8FAFC")
            st.pyplot(fig)

    st.markdown('<div class="section-header"> Continuous Telemetry Statistics</div>', unsafe_allow_html=True)
    desc_df = df[continuous_cols].describe().transpose()
    desc_df = desc_df.rename(columns={"50%": "median"})
    st.dataframe(desc_df[["mean", "std", "min", "25%", "median", "75%", "max"]], use_container_width=True)

    if continuous_cols:
        st.markdown('<div class="section-header"> Telemetry Feature Distribution</div>', unsafe_allow_html=True)
        selected_hist_col = st.selectbox("Select Telemetry Feature for Histogram", continuous_cols, key="hist_select")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df[selected_hist_col], kde=True, ax=ax, color="#3B82F6", bins=30)
        ax.set_title(f"Distribution & Density of '{selected_hist_col}'", fontsize=11, fontweight="bold", color="#F8FAFC")
        ax.set_xlabel(selected_hist_col.replace("_", " ").title(), color="#F8FAFC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

        st.markdown('<div class="section-header"> Telemetry Outlier Boxplot</div>', unsafe_allow_html=True)
        selected_box_col = st.selectbox("Select Telemetry Feature for Boxplot", continuous_cols, key="box_select")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        sns.boxplot(x=df[selected_box_col], ax=ax, color="#60A5FA", fliersize=4)
        ax.set_title(f"Outlier Spread of '{selected_box_col}'", fontsize=11, fontweight="bold", color="#F8FAFC")
        ax.set_xlabel(selected_box_col.replace("_", " ").title(), color="#F8FAFC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

        if TARGET_COLUMN in df.columns:
            st.markdown('<div class="section-header"> Feature Comparison: Healthy vs Failure Risk</div>', unsafe_allow_html=True)
            comp_feat = st.selectbox("Select Feature to Compare across Healthy vs Failure Batteries", continuous_cols, key="comp_select")
            fig, ax = plt.subplots(figsize=(9, 4))
            df_temp = df.copy()
            df_temp["Status"] = df_temp[TARGET_COLUMN].map({0: "Healthy", 1: "Failure Risk"})
            sns.boxplot(data=df_temp, x="Status", y=comp_feat, ax=ax, palette=["#3B82F6", "#EF4444"], width=0.4)
            ax.set_title(f"'{comp_feat}' Distribution Grouped by Failure Status", fontsize=11, fontweight="bold", color="#F8FAFC")
            ax.set_xlabel("", color="#F8FAFC")
            ax.set_ylabel(comp_feat.replace("_", " ").title(), color="#F8FAFC")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)

            st.markdown('<div class="section-header"> Top Correlated Features with Battery Failure</div>', unsafe_allow_html=True)
            corr_series = df[continuous_cols + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN, errors="ignore")
            top_corr = corr_series.abs().sort_values(ascending=False).head(12)
            top_cols = top_corr.index.tolist()

            corr_df_table = pd.DataFrame({
                "Feature": top_cols,
                "Correlation with Target": corr_series[top_cols].values,
                "Absolute Strength": top_corr.values
            }).sort_values(by="Absolute Strength", ascending=False).reset_index(drop=True)

            col_corr_a, col_corr_b = st.columns([1, 1])

            with col_corr_a:
                st.write("**Top 12 Telemetry Correlations**")
                st.dataframe(corr_df_table[["Feature", "Correlation with Target"]], use_container_width=True)

            with col_corr_b:
                fig, ax = plt.subplots(figsize=(6, 5))
                corr_sub = df[top_cols[:8] + [TARGET_COLUMN]].corr()
                sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True, annot_kws={"size": 8})
                ax.set_title("Correlation Heatmap (Top Features)", fontsize=10, fontweight="bold", color="#F8FAFC")
                st.pyplot(fig)


# --------------------------------------------------------------------------
# FEATURE IMPORTANCE PAGE
# --------------------------------------------------------------------------
def render_feature_importance():
    st.markdown('<div class="main-header"> Decision Tree Feature Importance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Gini importance scores assigned by the Decision Tree Classifier during tree splits.</div>',
        unsafe_allow_html=True,
    )

    if FEATURE_IMPORTANCE is None:
        st.error("Feature importance file not available.")
        return

    df = FEATURE_IMPORTANCE.copy()

    feature_col = next(c for c in df.columns if c.lower() == "feature")
    importance_col = next(c for c in df.columns if "importance" in c.lower())

    df_sorted = df.sort_values(by=importance_col, ascending=False).reset_index(drop=True)
    top_20 = df_sorted.head(20)

    st.markdown('<div class="section-header"> Top 20 Most Influential Telemetry Features</div>', unsafe_allow_html=True)
    st.dataframe(top_20, use_container_width=True)

    st.markdown('<div class="section-header"> Feature Importance Bar Chart</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_20[feature_col][::-1], top_20[importance_col][::-1], color="#3B82F6")
    ax.set_title("Top 20 Decision Tree Split Importances", fontsize=11, fontweight="bold", color="#F8FAFC")
    ax.set_xlabel("Importance Score", color="#F8FAFC")
    ax.set_ylabel("Telemetry Feature", color="#F8FAFC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown('<div class="section-header"> All Features Importance Table</div>', unsafe_allow_html=True)
    sort_order = st.radio("Sort order", ["Descending", "Ascending"], horizontal=True)
    ascending = sort_order == "Ascending"
    st.dataframe(
        df.sort_values(by=importance_col, ascending=ascending).reset_index(drop=True),
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------------------------------
def render_about():
    st.markdown('<div class="main-header"> About EV Battery Failure Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detailed overview of system architecture, machine learning methodology, telemetry feature guide, and developer details.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header"> Project Background & Objectives</div>', unsafe_allow_html=True)
    st.write(
        """
        Electric Vehicle (EV) adoption relies heavily on battery pack reliability and safety.
        Lithium-ion battery packs undergo electro-chemical degradation due to charge/discharge cycles,
        extreme operating temperatures, fast-charging frequency, and cell voltage imbalance.

        **Primary Project Goals:**
        - **Early Risk Detection:** Predict imminent EV battery failure before critical physical damage or vehicle breakdown occurs.
        - **BMS Intelligence:** Assist Battery Management Systems (BMS) in flagging high-stress operating conditions.
        - **Actionable Guidance:** Provide maintenance recommendations based on model confidence and predicted failure probability.
        """
    )

    st.markdown('<div class="section-header"> Machine Learning Model & Pipeline</div>', unsafe_allow_html=True)
    col_ml_a, col_ml_b = st.columns(2)
    with col_ml_a:
        st.write("**Model Architecture:**")
        st.write("• **Algorithm:** Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)")
        st.write("• **Split Criterion:** Gini Impurity")
        st.write("• **Input Handling:** Standard One-Hot dummy encoding for categoricals & continuous telemetry scaling")
        st.write("• **Output:** Binary Classification (0: Healthy, 1: Failure Risk)")

    with col_ml_b:
        st.write("**Data Pipeline Steps:**")
        st.write("1. **Telemetry Ingestion:** Cleaned telemetry dataset with 59 continuous indicators.")
        st.write("2. **Categorical Encoding:** Brand, model, chemistry, manufacturer, and drive type mapping.")
        st.write("3. **Model Evaluation:** Evaluated on test set Precision, Recall, and ROC-AUC metrics.")
        st.write("4. **Deployment:** Production Streamlit application for interactive single-sample prediction.")

    st.markdown('<div class="section-header"> Key Telemetry Metrics Guide</div>', unsafe_allow_html=True)
    guide_data = [
        {"Metric": "battery_health_percent", "Category": "Battery Health", "Description": "Current State of Health (SOH) of the battery relative to brand-new capacity."},
        {"Metric": "cell_voltage_std", "Category": "Pack Balance", "Description": "Standard deviation of cell voltages. High std indicates severe cell voltage imbalance."},
        {"Metric": "battery_stress_index", "Category": "Diagnostics", "Description": "Composite score measuring combined thermal, electrical, and mechanical stress."},
        {"Metric": "charging_quality_score", "Category": "Charging", "Description": "Efficiency score evaluating power stability and charging thermal rise."},
        {"Metric": "capacity_loss_percent", "Category": "Degradation", "Description": "Percentage loss of usable kWh capacity from initial factory specification."},
        {"Metric": "cooling_system_health", "Category": "Diagnostics", "Description": "Operational efficiency rating of active liquid cooling and thermal management."},
        {"Metric": "BMS_warning_count", "Category": "Diagnostics", "Description": "Total count of Diagnostic Trouble Codes (DTC) logged by the Battery Management System."},
        {"Metric": "thermal_runaway_risk", "Category": "Safety", "Description": "Predictive index estimating risk of catastrophic thermal breakdown."},
        {"Metric": "depth_of_discharge", "Category": "Usage", "Description": "Average depth to which the battery is discharged during daily operation."},
        {"Metric": "internal_resistance", "Category": "Battery Health", "Description": "Internal electrical resistance (mΩ) of battery cells, increasing with age."},
    ]
    st.dataframe(pd.DataFrame(guide_data), use_container_width=True)

    st.markdown('<div class="section-header"> Technology Stack & Author</div>', unsafe_allow_html=True)
    col_tech_a, col_tech_b = st.columns(2)
    with col_tech_a:
        st.write("**Tech Stack & Libraries:**")
        st.write("• **Language:** Python 3.10+")
        st.write("• **ML Framework:** Scikit-Learn")
        st.write("• **Data Manipulation:** Pandas, NumPy")
        st.write("• **Visualization:** Matplotlib, Seaborn")
        st.write("• **Deployment:** Streamlit Dashboard Framework")
        st.write("• **Model Serialization:** Joblib")

    with col_tech_b:
        st.write("**Author Information:**")
        st.write("• **Developer:** Yuraj Chauhan")
        st.write("• **Project:** EV Battery Failure Prediction")
        st.write("• **Repository:** Decision Tree Classifier Project")
        st.write("• **Deployment File:** `streamlit_app.py`")


# --------------------------------------------------------------------------
# PAGE ROUTER
# --------------------------------------------------------------------------
PAGE_RENDERERS = {
    " Home Overview": render_home,
    " Failure Prediction": render_prediction,
    " Telemetry Analytics": render_analytics,
    " Feature Importance": render_feature_importance,
    " About Project": render_about,
}

try:
    PAGE_RENDERERS[PAGE]()
except Exception as e:
    st.error(f"An unexpected error occurred while rendering this page: {e}")

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown('<div class="footer">EV Battery Failure Prediction Dashboard | Developed by Yuraj Chauhan</div>', unsafe_allow_html=True)