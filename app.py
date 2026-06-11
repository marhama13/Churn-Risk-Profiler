import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Profiler",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ── */
.main { background: #F0F4FF; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #94A3B8 !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    color: #CBD5E1 !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #334155;
    border-color: #3B82F6;
}

/* ── Header banner ── */
.iq-header {
    background: linear-gradient(135deg, #1D4ED8 0%, #0891B2 100%);
    border-radius: 16px;
    padding: 28px 36px;
    color: white;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 4px 24px rgba(29,78,216,0.25);
}
.iq-header h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
.iq-header p  { margin: 4px 0 0; font-size: 0.95rem; opacity: 0.85; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi-card {
    flex: 1;
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.kpi-label { font-size: 0.75rem; color: #64748B; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600; margin-bottom: 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #0F172A; line-height: 1; }
.kpi-sub   { font-size: 0.72rem; color: #94A3B8; margin-top: 4px; }

/* ── Section header ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #E2E8F0;
}

/* ── Chart card ── */
.chart-card {
    background: white;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    padding: 20px;
    margin-bottom: 20px;
}

/* ── Risk badge ── */
.risk-high   { background:#FEF2F2; color:#DC2626; border:1.5px solid #FECACA; border-radius:10px; padding:12px 18px; font-weight:700; font-size:1.1rem; }
.risk-medium { background:#FFFBEB; color:#D97706; border:1.5px solid #FDE68A; border-radius:10px; padding:12px 18px; font-weight:700; font-size:1.1rem; }
.risk-low    { background:#F0FDF4; color:#16A34A; border:1.5px solid #BBF7D0; border-radius:10px; padding:12px 18px; font-weight:700; font-size:1.1rem; }

/* ── Model accuracy table ── */
.acc-best { background: #EFF6FF; font-weight: 700; }

/* ── Input labels ── */
label { font-size: 0.85rem !important; font-weight: 500 !important; color: #374151 !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #0891B2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-size: 0.95rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
    box-shadow: 0 3px 12px rgba(37,99,235,0.3);
}
.stButton > button:hover { opacity: 0.88; }

/* ── Expander ── */
.streamlit-expanderHeader { font-weight: 600; color: #1E293B; }

/* ── Footer ── */
.iq-footer { text-align:center; color:#94A3B8; font-size:0.78rem; margin-top:40px; padding: 16px 0; border-top: 1px solid #E2E8F0; }

/* ── Metric overrides ── */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Palette for charts ──────────────────────────────────────────────────────
PALETTE = ["#2563EB", "#0891B2", "#7C3AED", "#059669", "#DC2626", "#D97706"]
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#374151"),
    margin=dict(l=8, r=8, t=40, b=8),
    xaxis=dict(showgrid=False, linecolor="#E2E8F0"),
    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
)

def apply_layout(fig):
    fig.update_layout(**CHART_LAYOUT)
    return fig

# ─── Data & model loading ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("telco.csv")

@st.cache_resource
def train_all_models(df):
    data = df.copy()
    if "customerID" in data.columns:
        data.drop("customerID", axis=1, inplace=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data.dropna(inplace=True)
    le = LabelEncoder()
    for c in data.columns:
        if data[c].dtype == "object":
            data[c] = le.fit_transform(data[c])
    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, max_depth=5, eval_metric="logloss"),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    results = {n: accuracy_score(y_test, m.predict(X_test)) for n, m in models.items()}
    best_name = max(results, key=results.get)
    return data, X, scaler, X_test, y_test, models, best_name

df   = load_data()
data, X, scaler, X_test, y_test, models, best_name = train_all_models(df)
best_model = models[best_name]

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="iq-header">
  <div style="font-size:2.6rem;line-height:1">📊</div>
  <div>
    <h1>Churn Risk Profiler</h1>
    <p>Customer Retention Analytics Platform</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI strip ───────────────────────────────────────────────────────────────
churn_pct  = data["Churn"].mean() * 100
avg_charge = data["MonthlyCharges"].mean()
avg_tenure = data["tenure"].mean()
total_rev  = (data["MonthlyCharges"] * data["tenure"]).sum()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Customers", f"{len(data):,}")
with k2:
    st.metric("Churn Rate", f"{churn_pct:.1f}%", delta=f"-{churn_pct:.1f}% target")
with k3:
    st.metric("Avg Monthly Charges", f"${avg_charge:.2f}")
with k4:
    st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar nav ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px;">
      <div style="font-size:2rem">📊</div>
      <div style="font-size:1.1rem;font-weight:700;color:#F1F5F9 !important;margin-top:4px;">Churn Risk Profiler</div>
      <div style="font-size:0.72rem;color:#64748B !important;margin-top:2px;">v1.0 · Telco Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["🏠  Dashboard", "📈  Analytics", "🧪  Model Lab", "🎯  Prediction Center", "ℹ️  About"],
        label_visibility="visible",
    )

    st.markdown("<hr style='border-color:#334155;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#1E293B;border-radius:10px;padding:12px;border:1px solid #334155;">
      <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Best Model</div>
      <div style="font-size:0.9rem;font-weight:600;color:#3B82F6;">{best_name}</div>
      <div style="font-size:0.75rem;color:#64748B;margin-top:2px;">
        {accuracy_score(y_test, best_model.predict(X_test))*100:.1f}% accuracy
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Pages ───────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    st.markdown('<p class="section-title">Business Overview</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            data, x="MonthlyCharges", nbins=40,
            title="Monthly Charges Distribution",
            color_discrete_sequence=[PALETTE[0]],
        )
        apply_layout(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_counts = data["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "count"]
        churn_counts["Label"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig2 = px.pie(
            churn_counts, names="Label", values="count",
            title="Churn vs. Retained Customers",
            color_discrete_sequence=[PALETTE[3], PALETTE[4]],
            hole=0.45,
        )
        apply_layout(fig2)
        fig2.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(
            data, x="Churn", y="MonthlyCharges",
            title="Monthly Charges by Churn Status",
            color="Churn", color_discrete_sequence=[PALETTE[3], PALETTE[4]],
            labels={"Churn": "Churned (1=Yes)"},
        )
        apply_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.scatter(
            data.sample(min(500, len(data)), random_state=1),
            x="tenure", y="MonthlyCharges",
            color="Churn", opacity=0.55,
            title="Tenure vs. Monthly Charges",
            color_continuous_scale=["#059669", "#DC2626"],
        )
        apply_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)

elif "Analytics" in page:
    st.markdown('<p class="section-title">Customer Behaviour Analytics</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        data_plot = data.copy()
        data_plot["TenureGroup"] = pd.cut(data_plot["tenure"], bins=[0,12,24,36,48,60,72],
                                           labels=["0-12","12-24","24-36","36-48","48-60","60-72"])
        trend = data_plot.groupby("TenureGroup", observed=True)["Churn"].mean().reset_index()
        trend["ChurnPct"] = (trend["Churn"] * 100).round(1)

        fig = px.bar(
            trend, x="TenureGroup", y="ChurnPct",
            title="Churn Rate by Tenure Cohort (%)",
            color="ChurnPct",
            color_continuous_scale=["#3B82F6", "#DC2626"],
            text="ChurnPct",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rf_imp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_imp.fit(X, data["Churn"])
        imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf_imp.feature_importances_
        }).sort_values("Importance", ascending=True).tail(12)

        fig2 = px.bar(
            imp, x="Importance", y="Feature", orientation="h",
            title="Top 12 Churn Driver Features",
            color="Importance",
            color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        )
        apply_layout(fig2)
        fig2.update_traces(texttemplate="%{x:.3f}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Contract & Billing Breakdown</p>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        contract_churn = data.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn["ChurnPct"] = (contract_churn["Churn"] * 100).round(1)
        contract_churn["Label"] = contract_churn["Contract"].map({0: "Month-to-Month", 1: "One Year", 2: "Two Year"})
        fig3 = px.bar(
            contract_churn, x="Label", y="ChurnPct",
            title="Churn Rate by Contract Type (%)",
            color="ChurnPct", color_continuous_scale=["#059669", "#DC2626"],
            text="ChurnPct",
        )
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        apply_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        corr = data[["tenure","MonthlyCharges","TotalCharges","Churn"]].corr()
        fig4 = px.imshow(
            corr, text_auto=".2f",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        apply_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)

elif "Model Lab" in page:
    st.markdown('<p class="section-title">Model Performance Comparison</p>', unsafe_allow_html=True)

    results_rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
        results_rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred) * 100, 2),
            "Precision (Churn)": round(report["1"]["precision"] * 100, 2),
            "Recall (Churn)": round(report["1"]["recall"] * 100, 2),
            "F1 Score (Churn)": round(report["1"]["f1-score"] * 100, 2),
        })
    results_df = pd.DataFrame(results_rows).sort_values("Accuracy", ascending=False)

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("**Model Scorecard**")
        st.dataframe(
            results_df.style
              .highlight_max(subset=["Accuracy","F1 Score (Churn)"], color="#DBEAFE")
              .format("{:.2f}%", subset=["Accuracy","Precision (Churn)","Recall (Churn)","F1 Score (Churn)"]),
            use_container_width=True, hide_index=True,
        )

    with col2:
        metrics = ["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1 Score (Churn)"]
        fig_radar = go.Figure()
        for _, row in results_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill="toself",
                name=row["Model"],
                opacity=0.65,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
            title="Model Comparison — Radar",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            margin=dict(l=40, r=40, t=50, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<p class="section-title" style="margin-top:8px">Best Model Deep-Dive: ' + best_name + '</p>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        with st.expander("📋 Full Classification Report", expanded=True):
            pred = best_model.predict(X_test)
            report = classification_report(y_test, pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

    with col4:
        with st.expander("🔲 Confusion Matrix", expanded=True):
            cm = confusion_matrix(y_test, best_model.predict(X_test))
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Retained", "Churned"],
                y=["Retained", "Churned"],
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            apply_layout(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

elif "Prediction" in page:
    st.markdown('<p class="section-title">Customer Churn Risk Assessment</p>', unsafe_allow_html=True)

    st.info(f"Predictions powered by **{best_name}** — highest accuracy across all trained models.", icon="🤖")

    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        st.markdown("**👤 Customer Profile**")
        inputs["gender"]        = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x else "Female")
        inputs["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inputs["Partner"]       = st.selectbox("Has Partner", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inputs["Dependents"]    = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inputs["tenure"]        = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**📱 Services**")
        inputs["PhoneService"]    = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inputs["Contract"]        = st.selectbox("Contract Type", [0, 1, 2],
                                                  format_func=lambda x: ["Month-to-Month", "One Year", "Two Year"][x])
        inputs["PaperlessBilling"]= st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col3:
        st.markdown("**💳 Billing**")
        inputs["MonthlyCharges"] = st.slider("Monthly Charges ($)", 0, 150, 70)
        inputs["TotalCharges"]   = st.slider("Total Charges ($)", 0, 9000, 1000)
        st.markdown("<br>", unsafe_allow_html=True)

    for col in X.columns:
        if col not in inputs:
            inputs[col] = 0

    input_df = pd.DataFrame([inputs])[X.columns]

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_clicked = st.button("⚡ Run Risk Assessment")

    if predict_clicked:
        scaled = scaler.transform(input_df)
        prob   = best_model.predict_proba(scaled)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1, 1, 1.5])

        with r1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 36, "family": "Inter"}},
                title={"text": "Churn Risk Score", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 40],   "color": "#D1FAE5"},
                        {"range": [40, 70],  "color": "#FEF9C3"},
                        {"range": [70, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": "#DC2626", "width": 3}, "value": 70},
                },
            ))
            gauge.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(gauge, use_container_width=True)

        with r2:
            if prob >= 0.7:
                st.markdown(f'<div class="risk-high">🔴 HIGH RISK<br><small>Immediate action needed</small></div>', unsafe_allow_html=True)
            elif prob >= 0.4:
                st.markdown(f'<div class="risk-medium">🟠 MEDIUM RISK<br><small>Monitor & engage proactively</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">🟢 LOW RISK<br><small>Customer likely to stay</small></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Risk Score", f"{prob*100:.1f}%")
            st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")

        with r3:
            st.markdown("**📋 Recommended Actions**")
            recs = []
            if inputs["tenure"] < 12:
                recs.append(("🚀", "Assign a dedicated onboarding specialist for first-year customers."))
            if inputs["MonthlyCharges"] > 80:
                recs.append(("💰", "Offer a 10–15% loyalty discount or bundle upgrade."))
            if inputs["Contract"] == 0:
                recs.append(("📄", "Pitch a 1-year or 2-year contract with a free month incentive."))
            if inputs["SeniorCitizen"] == 1:
                recs.append(("👴", "Assign a dedicated customer care agent for senior customers."))
            if prob >= 0.7:
                recs.append(("⚡", "Trigger high-priority win-back campaign immediately."))
            if not recs:
                recs.append(("✅", "No immediate action required. Continue standard engagement."))
            for icon, rec in recs:
                st.markdown(f"**{icon}** {rec}")

elif "About" in page:
    st.markdown('<p class="section-title">About RetentionIQ</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **RetentionIQ** is a full-stack churn prediction and analytics platform built for telecom operators.
        It combines classical ML models with an interactive analytics suite to help retention teams
        identify at-risk customers and act before they churn.

        #### Models Trained
        - Logistic Regression (baseline)
        - Decision Tree (interpretable)
        - Random Forest (ensemble)
        - XGBoost (gradient boosting)

        #### Data
        Trained on the IBM Telco Customer Churn dataset — 7,000+ customer records
        with demographics, services, billing, and churn labels.
        """)

    with c2:
        st.markdown("""
        #### Tech Stack
        | Layer | Library |
        |---|---|
        | App framework | Streamlit |
        | ML models | scikit-learn, XGBoost |
        | Visualisations | Plotly |
        | Data processing | pandas, NumPy |

        #### How to run
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```
        """)

    st.success("Place `telco.csv` inside a `data/` folder next to `app.py` before running.", icon="💡")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<div class='iq-footer'>RetentionIQ © 2026 · Built with Streamlit & scikit-learn</div>",
            unsafe_allow_html=True)
