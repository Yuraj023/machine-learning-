"""
Customer Churn Prediction — Streamlit App
Combined Single-File Version (Informative & Balanced Layout)
"""

from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# FILE PATHS & CONSTANTS
# ==========================================
# Hardcoded absolute paths for the models
MODEL_PATH = r"D:\machine learning\machine_learning\logistic_regression_project\churn_prediction\models\customer_churn_pipeline.joblib"
ENCODER_PATH = r"D:\machine learning\machine_learning\logistic_regression_project\churn_prediction\models\label_encoder.joblib"

APP_ROOT = Path(__file__).resolve().parent
CSS_PATH = APP_ROOT / "styles.css"


# ==========================================
# DATA & MODEL HELPERS
# ==========================================
@st.cache_resource
def load_pipeline():
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_resource
def load_label_encoder():
    path = Path(ENCODER_PATH)
    if not path.exists():
        return None
    return joblib.load(path)


def build_input_dataframe(customer_data):
    return pd.DataFrame([customer_data])


def predict_churn_probability(pipeline, input_df):
    proba = pipeline.predict_proba(input_df)[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


def risk_level(probability):
    if probability >= 0.6:
        return "HIGH", "high"
    if probability >= 0.3:
        return "MEDIUM", "medium"
    return "LOW", "low"


# ==========================================
# CHART HELPERS
# ==========================================
def analytics_bar_chart():
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "ROC-AUC", "Features"],
        "Value": [80.20, 84.89, 20.00],
    })
    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        text="Value",
        color="Metric",
        color_discrete_sequence=["#3b82f6", "#0ea5e9", "#22c55e"],
        title="Model Performance Metrics"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        showlegend=False,
        yaxis=dict(title="Score (%)", range=[0, 100], gridcolor="#334155"),
        xaxis=dict(title="", gridcolor="#334155"),
    )
    return fig


def confusion_matrix_chart():
    matrix = pd.DataFrame(
        [[81.5, 18.5], [25.0, 75.0]],
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )
    fig = px.imshow(
        matrix,
        text_auto=".1f",
        color_continuous_scale=[[0, "#0f172a"], [0.5, "#2563eb"], [1, "#22c55e"]],
        aspect="auto",
    )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        coloraxis_colorbar=dict(title="%", tickfont=dict(color="#e5e7eb")),
    )
    fig.update_xaxes(side="top")
    return fig


def threshold_summary_chart():
    summary = pd.DataFrame({
        "Outcome": ["Churners Caught", "Non-Churn Correctly Kept"],
        "Score": [75.0, 81.5],
    })
    fig = px.line(summary, x="Outcome", y="Score", markers=True, title="Business Impact of Threshold")
    fig.update_traces(line_color="#f59e0b", marker=dict(size=10, color="#f59e0b"))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        yaxis=dict(title="Approx. %", range=[0, 100], gridcolor="#334155"),
        xaxis=dict(title="", gridcolor="#334155"),
    )
    return fig


def gauge_chart(probability):
    color = "#e0435c" if probability >= 0.6 else "#e0a53c" if probability >= 0.3 else "#3ce07a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "#ffffff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9aa0ac"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(60, 224, 122, 0.16)"},
                {"range": [30, 60], "color": "rgba(224, 165, 60, 0.16)"},
                {"range": [60, 100], "color": "rgba(224, 67, 92, 0.16)"},
            ],
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6e6e6"},
    )
    return fig


# ==========================================
# UI COMPONENTS & HTML/CSS RENDERERS
# ==========================================
def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_metric_card(value, label, subtitle=None):
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {subtitle_html}
    </div>
    """


def render_dashboard_result(probability, headline, level, level_class, will_churn):
    recommendations = [
        "Issue an immediate retention discount (e.g., 10-15% off next 3 months).",
        "Proactively contact the customer to gather feedback on service quality.",
        "Recommend switching to a 1-year or 2-year contract with built-in savings.",
    ] if will_churn else [
        "Continue standard engagement — this customer is stable and likely satisfied.",
        "Consider upsell opportunities (e.g., adding Premium Tech Support or Device Protection).",
        "Perform routine periodic check-ins to maintain long-term satisfaction.",
    ]

    return f"""
    <div class="dashboard-card">
        <div class="result-card result-{level_class}">
            <h2>{headline}</h2>
            <span class="risk-badge risk-{level_class}">RISK LEVEL: {level}</span>
            <div style="margin-top:0.85rem; font-size:1.7rem; font-weight:800; color:#f8fafc;">{probability * 100:.2f}% Probability</div>
        </div>
        <div class="action-box">
            <b>Recommended Business Actions</b>
            <ul>
                {''.join(f'<li>{item}</li>' for item in recommendations)}
            </ul>
        </div>
    </div>
    """


def customer_form_fields(st_context):
    st_context.markdown('<div class="section-header">Customer Profile Configuration</div>', unsafe_allow_html=True)
    
    # Switched to a 3-column layout to fix the wide empty spaces on the UI
    with st_context.container():
        col1, col2, col3 = st_context.columns(3)
        
        with col1:
            st_context.markdown("**👤 Demographics & Account**")
            gender = st_context.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st_context.selectbox("Senior Citizen", ["No", "Yes"], help="Senior citizens often have different tech support needs.")
            partner = st_context.selectbox("Partner", ["Yes", "No"], help="Customers with partners are often more financially stable and less likely to churn.")
            dependents = st_context.selectbox("Dependents", ["No", "Yes"], help="Families with dependents tend to rely heavily on stable home internet/phone services.")
            tenure = st_context.number_input("Tenure (Months)", min_value=0, max_value=72, value=12, help="Longer tenure typically indicates higher brand loyalty and lower churn risk.")

        with col2:
            st_context.markdown("**🛠️ Subscribed Services**")
            phone_service = st_context.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st_context.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st_context.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], help="Fiber optic customers historically have higher churn if pricing is too high, while DSL is cheaper but slower.")
            online_security = st_context.selectbox("Online Security", ["No", "Yes", "No internet service"], help="Add-on security services tie the customer closer to the ecosystem.")
            online_backup = st_context.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st_context.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st_context.selectbox("Tech Support", ["No", "Yes", "No internet service"], help="Lack of tech support is a strong indicator of churn when users face network issues.")

        with col3:
            st_context.markdown("**💳 Contract & Billing**")
            streaming_tv = st_context.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st_context.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            st_context.divider() # Visual break for billing specifics
            contract = st_context.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Month-to-month contracts are highly volatile and carry the highest churn risk.")
            paperless_billing = st_context.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st_context.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                help="Automated payments (credit card, bank) usually lower churn risk compared to manual checks."
            )
            monthly_charges = st_context.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5, help="High monthly charges without bundled benefits increase churn likelihood.")
            
            # Auto-calculate total charges based on tenure and monthly to make it easier for the user
            default_total = round(tenure * monthly_charges, 2)
            total_charges = st_context.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total, step=0.5, help="Lifetime value of the customer up to this point.")

    return {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }


def metric_row() -> None:
    left, middle_left, middle_right, right = st.columns(4)
    with left:
        st.markdown(render_metric_card("80.20%", "Accuracy", "Correctly predicted outcomes"), unsafe_allow_html=True)
    with middle_left:
        st.markdown(render_metric_card("0.8489", "ROC-AUC", "True Positive vs False Positive"), unsafe_allow_html=True)
    with middle_right:
        st.markdown(render_metric_card("20", "Features", "Data points per customer"), unsafe_allow_html=True)
    with right:
        st.markdown(render_metric_card("Logistic", "Algorithm", "Scikit-Learn Pipeline"), unsafe_allow_html=True)


def sidebar_panel() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
                <h3>📡 Customer Churn Prediction</h3>
                <p>
                    An AI-powered dashboard that predicts whether a telecom customer is
                    likely to churn using a trained <b>Logistic Regression</b> machine
                    learning pipeline. The application helps businesses identify
                    high-risk customers and supports data-driven customer retention
                    strategies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            render_metric_card(
                "80.20%",
                "Model Accuracy",
                "Reliable baseline performance"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            render_metric_card(
                "0.8489",
                "ROC-AUC Score",
                "Excellent class separation"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            render_metric_card(
                "0.35",
                "Risk Threshold",
                "Scores ≥ 0.35 are flagged as Churn"
            ),
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ How the Prediction Works", expanded=False):
            st.markdown(
                """
                **Prediction Workflow**

                1. Enter customer demographic, service, contract, and billing information.
                2. The saved machine learning pipeline automatically preprocesses the data using:
                   - One-Hot Encoding for categorical features
                   - Standard Scaling for numerical features
                3. The Logistic Regression model calculates the probability of customer churn.
                4. If the predicted probability is **0.35 or higher**, the customer is classified as **Likely to Churn**; otherwise, the customer is predicted to **Stay**.
                5. The application displays the churn probability, risk level, and recommended retention actions to support business decision-making.
                """
            )


# ==========================================
# MAIN TABS & VIEWS
# ==========================================
def render_prediction_tab(pipeline) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Customer Churn Prediction</h1>
            <p>Review a customer profile, instantly score their churn risk, and generate actionable business retention strategies.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_row()
    
    st.markdown("<br>", unsafe_allow_html=True)
    customer_data = customer_form_fields(st)

    st.markdown("<br>", unsafe_allow_html=True)
    action_col_left, action_col_center, action_col_right = st.columns([1, 1, 1])
    with action_col_center:
        predict_clicked = st.button("Predict Customer Churn", width="stretch", type="primary")

    if not predict_clicked:
        st.markdown("---")
        st.caption("Ready for analysis. Modify the customer attributes above and click Predict to view the dashboard.")
        return

    if pipeline is None:
        st.error(f"Cannot run prediction because the trained pipeline is missing at {MODEL_PATH}.")
        return

    input_df = build_input_dataframe(customer_data)
    with st.spinner("Analyzing profile and scoring customer risk..."):
        try:
            churn_probability = predict_churn_probability(pipeline, input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    level, level_class = risk_level(churn_probability)
    CHURN_THRESHOLD = 0.35
    will_churn = churn_probability >= CHURN_THRESHOLD
    headline = "⚠️ Customer is Likely to Churn" if will_churn else "✅ Customer is Likely to Stay"

    st.markdown("---")
    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
    
    st.markdown(
        render_dashboard_result(
            churn_probability,
            headline,
            level,
            level_class,
            will_churn,
        ),
        unsafe_allow_html=True,
    )

    chart_col, insight_col = st.columns([1.1, 0.9])
    with chart_col:
        st.plotly_chart(gauge_chart(churn_probability), width="stretch")
    with insight_col:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="section-header" style="margin-top:0;">Risk Snapshot</div>
                <p style="margin:0 0 0.8rem 0; color:#cbd5e1;">Final model assessment based on the provided 20 features.</p>
                <div class="metric-card">
                    <div class="value">{churn_probability * 100:.2f}%</div>
                    <div class="label">Calculated Probability</div>
                </div>
                <div style="height:0.8rem;"></div>
                <div class="metric-card">
                    <div class="value">{level}</div>
                    <div class="label">Assigned Risk Tier</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with st.expander("View Raw Input DataFrame"):
        st.dataframe(input_df, use_container_width=True)


def render_about_tab() -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Model Intelligence & Metrics</h1>
            <p>A deeper dive into the algorithm's performance signals, tuning decisions, and evaluation context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    with metrics[0]:
        st.markdown(render_metric_card("80.20%", "Accuracy"), unsafe_allow_html=True)
    with metrics[1]:
        st.markdown(render_metric_card("0.8489", "ROC-AUC"), unsafe_allow_html=True)
    with metrics[2]:
        st.markdown(render_metric_card("0.35", "Threshold"), unsafe_allow_html=True)
    with metrics[3]:
        st.markdown(render_metric_card("Logistic", "Classifier"), unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        st.plotly_chart(analytics_bar_chart(), width="stretch")
    with right:
        st.plotly_chart(threshold_summary_chart(), width="stretch")

    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    st.caption("Normalized evaluation snapshot showing True Positives, True Negatives, False Positives, and False Negatives.")
    st.plotly_chart(confusion_matrix_chart(), width="stretch")

    st.markdown('<div class="section-header">Technical Interpretation</div>', unsafe_allow_html=True)
    with st.expander("What do the primary metrics indicate?"):
        st.write(
            "**Accuracy (80.20%)** shows the overall correctness of the model across all predictions. "
            "**ROC-AUC (0.8489)** is an excellent score indicating the model is highly capable of distinguishing between customers who will churn and those who will stay."
        )
    with st.expander("Why was the decision threshold adjusted to 0.35?"):
        st.write(
            "In churn prediction, missing a customer who is going to leave (False Negative) is usually more expensive than accidentally offering a retention discount to a happy customer (False Positive). "
            "By lowering the threshold from the default 0.50 to **0.35**, the model becomes more aggressive, prioritizing the identification of at-risk customers over pure accuracy."
        )
    with st.expander("How does the pipeline manage the data?"):
        st.write(
            "The model utilizes a Scikit-Learn `Pipeline`. When you click predict, the raw text inputs (like 'Month-to-month' or 'Fiber optic') are automatically One-Hot Encoded, and numerical values (like Tenure and Monthly Charges) are Standard Scaled before being fed into the Logistic Regression equation. This ensures deployment consistency."
        )


# ==========================================
# MAIN EXECUTION
# ==========================================
def main() -> None:
    load_css()
    sidebar_panel()
    
    # Load Models (Cached for performance)
    pipeline = load_pipeline()
    _ = load_label_encoder() 

    st.markdown(
        """
        <div class="navbar-shell">
            <div class="navbar-title">Telecom Customer Churn Predictor</div>
            <div class="navbar-subtitle">Analyze behavior, predict risk, and retain customers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_tab, about_tab = st.tabs(["📊 Live Prediction", "🧠 Model Architecture & Metrics"])

    with prediction_tab:
        render_prediction_tab(pipeline)

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()