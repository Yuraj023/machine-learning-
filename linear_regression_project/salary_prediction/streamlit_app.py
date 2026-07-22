import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import datetime

try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# -------------------------------
# Paths (relative to this file)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "linear_regression_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data", "cleaned_salaries.csv")

# -------------------------------
# Friendly label mappings
# -------------------------------
EXPERIENCE_MAP = {
    "EN": "Entry Level",
    "MI": "Mid Level",
    "SE": "Senior Level",
    "EX": "Executive Level",
}
EMPLOYMENT_MAP = {
    "FT": "Full Time",
    "PT": "Part Time",
    "CT": "Contract",
    "FL": "Freelance",
}
COMPANY_SIZE_MAP = {
    "S": "Small (< 50 employees)",
    "M": "Medium (50–250 employees)",
    "L": "Large (> 250 employees)",
}
REMOTE_MAP = {
    0: "On-site (No Remote)",
    50: "Hybrid",
    100: "Fully Remote",
}

@st.cache_data
def country_name(code):
    """Convert an ISO alpha-2 country code to a full country name."""
    if not isinstance(code, str):
        return str(code)
    if HAS_PYCOUNTRY:
        try:
            match = pycountry.countries.get(alpha_2=code.upper())
            if match:
                return match.name
        except Exception:
            pass
    return code


def friendly_select(label, options_dict, keys_present):
    """Build a selectbox from a code->label dict, restricted to keys actually in the data,
    displaying the friendly label but returning the underlying code."""
    valid_keys = [k for k in options_dict if k in keys_present]
    extra = [k for k in keys_present if k not in options_dict]
    all_keys = sorted(valid_keys, key=lambda k: options_dict[k]) + sorted(extra)
    return st.selectbox(
        label,
        all_keys,
        format_func=lambda k: options_dict.get(k, k)
    )

# -------------------------------
# Load Model & Dataset
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(
        "Could not find the model or data file. Make sure the app is run from "
        "inside the `salary_prediction` folder so the relative paths resolve correctly.\n\n"
        f"Details: {e}"
    )
    st.stop()

# Detect the salary/target column
SALARY_COL = None
for candidate in ["salary_in_usd", "salary", "salary_usd"]:
    if candidate in df.columns:
        SALARY_COL = candidate
        break

# -------------------------------
# Sidebar (About / Developer / Dataset info only)
# -------------------------------
st.sidebar.title("📘 About This Project")
st.sidebar.write("""
**Salary Prediction App**

Predicts an employee's salary using a **Multiple Linear Regression** model
trained on real-world data science job salary data.

**Features Used**
- Work Year
- Experience Level
- Employment Type
- Job Title
- Employee Residence
- Remote Ratio
- Company Location
- Company Size
""")

st.sidebar.markdown("---")
st.sidebar.write("**📂 Dataset Info**")
st.sidebar.write(f"- Rows: `{df.shape[0]:,}`")
st.sidebar.write(f"- Columns: `{df.shape[1]}`")
st.sidebar.write(f"- Years covered: `{int(df['work_year'].min())} – {int(df['work_year'].max())}`")

st.sidebar.markdown("---")
st.sidebar.write("**👨‍💻 Developer**")
st.sidebar.success("Yuraj Chauhan")

# -------------------------------
# Main dashboard header + navbar (tabs)
# -------------------------------
st.title("💼 Salary Prediction Dashboard")

tab_prediction, tab_analytics = st.tabs(["🏠 Prediction", "📊 Analytics"])

# ===============================================================
# TAB 1: PREDICTION
# ===============================================================
with tab_prediction:

    st.write("Fill in the employee details below and click **Predict Salary**.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Improved Work Year: Defaults to current year, allowing predictions into the near future.
        current_year = datetime.date.today().year
        year_min = int(df["work_year"].min())
        data_max_year = int(df["work_year"].max())
        
        work_year = st.number_input(
            "Work Year", 
            min_value=year_min, 
            max_value=current_year + 5, 
            value=max(data_max_year, current_year),
            step=1
        )

        experience_level = friendly_select(
            "Experience Level", EXPERIENCE_MAP, df["experience_level"].unique()
        )
        employment_type = friendly_select(
            "Employment Type", EMPLOYMENT_MAP, df["employment_type"].unique()
        )
        remote_ratio = friendly_select(
            "Remote Ratio", REMOTE_MAP, df["remote_ratio"].unique()
        )

    with col2:
        job_title = st.selectbox(
            "Job Title (type to search)",
            sorted(df["job_title"].unique())
        )

        residence_keys = sorted(df["employee_residence"].unique(), key=country_name)
        employee_residence = st.selectbox(
            "Employee Residence",
            residence_keys,
            format_func=country_name
        )

        # Auto-sync logic for Company Location
        location_keys = sorted(df["company_location"].unique(), key=country_name)
        default_loc_idx = 0
        
        # If user selects On-site (0) and their residence is in the company locations list, sync them
        if remote_ratio == 0 and employee_residence in location_keys:
            default_loc_idx = location_keys.index(employee_residence)
            st.caption("💡 *Company location auto-synced to residence for on-site role.*")

        company_location = st.selectbox(
            "Company Location",
            location_keys,
            index=default_loc_idx,
            format_func=country_name
        )

        company_size = friendly_select(
            "Company Size", COMPANY_SIZE_MAP, df["company_size"].unique()
        )

    st.markdown("---")

    if st.button("🔮 Predict Salary", use_container_width=True):
        input_data = pd.DataFrame({
            "work_year": [work_year],
            "experience_level": [experience_level],
            "employment_type": [employment_type],
            "job_title": [job_title],
            "employee_residence": [employee_residence],
            "remote_ratio": [remote_ratio],
            "company_location": [company_location],
            "company_size": [company_size]
        })

        try:
            prediction = model.predict(input_data)[0]

            st.success("Prediction Completed Successfully!")

            with st.container(border=True):
                st.subheader("📋 Prediction Summary")
                s1, s2 = st.columns(2)
                with s1:
                    st.write(f"**Job Title:** {job_title}")
                    st.write(f"**Experience Level:** {EXPERIENCE_MAP.get(experience_level, experience_level)}")
                    st.write(f"**Employment Type:** {EMPLOYMENT_MAP.get(employment_type, employment_type)}")
                    st.write(f"**Work Year:** {work_year}")
                with s2:
                    st.write(f"**Employee Residence:** {country_name(employee_residence)}")
                    st.write(f"**Company Location:** {country_name(company_location)}")
                    st.write(f"**Company Size:** {COMPANY_SIZE_MAP.get(company_size, company_size)}")
                    st.write(f"**Remote Ratio:** {REMOTE_MAP.get(remote_ratio, remote_ratio)}")

                st.markdown("---")
                st.metric(label="💰 Predicted Salary (USD)", value=f"${prediction:,.2f}")

        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

# ===============================================================
# TAB 2: ANALYTICS
# ===============================================================
with tab_analytics:

    st.write("Explore trends and patterns across the dataset used to train this model.")
    st.markdown("---")

    if SALARY_COL is None:
        st.warning(
            "Couldn't find a salary column in the dataset (expected one of "
            "`salary_in_usd`, `salary`, `salary_usd`). Some charts below may not render."
        )

    # ---- Dataset overview ----
    st.subheader("🗂️ Dataset Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Total Records", f"{df.shape[0]:,}")
    o2.metric("Unique Job Titles", f"{df['job_title'].nunique():,}")
    o3.metric("Countries (Residence)", f"{df['employee_residence'].nunique():,}")
    if SALARY_COL:
        o4.metric("Avg Salary (USD)", f"${df[SALARY_COL].mean():,.0f}")

    with st.expander("Preview raw data"):
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("---")

    if SALARY_COL:
        # ---- Salary distribution ----
        st.subheader("💵 Salary Distribution")
        fig = px.histogram(df, x=SALARY_COL, nbins=40, title="Distribution of Salaries (USD)")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        c1, c2 = st.columns(2)

        # ---- Salary by experience level ----
        with c1:
            st.subheader("📈 Salary by Experience Level")
            exp_df = df.copy()
            exp_df["experience_level"] = exp_df["experience_level"].map(
                lambda x: EXPERIENCE_MAP.get(x, x)
            )
            fig = px.box(exp_df, x="experience_level", y=SALARY_COL,
                         title="Salary Spread by Experience Level")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Salary by employment type ----
        with c2:
            st.subheader("🧾 Salary by Employment Type")
            emp_df = df.copy()
            emp_df["employment_type"] = emp_df["employment_type"].map(
                lambda x: EMPLOYMENT_MAP.get(x, x)
            )
            fig = px.box(emp_df, x="employment_type", y=SALARY_COL,
                         title="Salary Spread by Employment Type")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        # ---- Salary by company size ----
        with c3:
            st.subheader("🏢 Salary by Company Size")
            size_df = df.copy()
            size_df["company_size"] = size_df["company_size"].map(
                lambda x: COMPANY_SIZE_MAP.get(x, x)
            )
            fig = px.bar(
                size_df.groupby("company_size")[SALARY_COL].mean().reset_index(),
                x="company_size", y=SALARY_COL,
                title="Average Salary by Company Size"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Remote work analysis ----
        with c4:
            st.subheader("🌍 Remote Work Analysis")
            remote_df = df.copy()
            remote_df["remote_ratio"] = remote_df["remote_ratio"].map(
                lambda x: REMOTE_MAP.get(x, x)
            )
            fig = px.bar(
                remote_df.groupby("remote_ratio")[SALARY_COL].mean().reset_index(),
                x="remote_ratio", y=SALARY_COL,
                title="Average Salary by Work Mode"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ---- Top 10 highest paying jobs ----
        st.subheader("🏆 Top 10 Highest-Paying Job Titles")
        top_jobs = (
            df.groupby("job_title")[SALARY_COL]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            top_jobs, x=SALARY_COL, y="job_title", orientation="h",
            title="Top 10 Job Titles by Average Salary"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ---- Country-wise salary analysis ----
        st.subheader("🗺️ Country-wise Salary Analysis")
        country_df = df.copy()
        country_df["country"] = country_df["company_location"].map(country_name)
        top_countries = (
            country_df.groupby("country")[SALARY_COL]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(
            top_countries, x=SALARY_COL, y="country", orientation="h",
            title="Top 15 Countries by Average Salary (Company Location)"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ---- Correlation heatmap ----
        st.subheader("🔥 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Correlation Between Numeric Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns to compute a correlation heatmap.")

    st.markdown("---")

    # ---- Model information ----
    st.subheader("🤖 Model Information")
    m1, m2 = st.columns(2)
    with m1:
        st.write("**Algorithm:** Multiple Linear Regression")
        st.write(f"**Model object type:** `{type(model).__name__}`")
        st.write(f"**Training rows (approx.):** {df.shape[0]:,}")
    with m2:
        st.write("**Features used:**")
        st.write(
            "- Work Year\n"
            "- Experience Level\n"
            "- Employment Type\n"
            "- Job Title\n"
            "- Employee Residence\n"
            "- Remote Ratio\n"
            "- Company Location\n"
            "- Company Size"
        )

st.markdown("---")
st.caption("Machine Learning Project | Multiple Linear Regression | Streamlit")