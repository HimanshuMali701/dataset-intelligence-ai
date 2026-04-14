import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.data_profiler import profile_dataset
from src.quality_checker import run_quality_checks
from src.scoring import calculate_health_score
from src.suggestion_engine import generate_suggestions
from src.visualization import (
    plot_missing_values,
    plot_correlation,
    plot_target_distribution,
    plot_numeric_distribution,
)
from src.llm_agent import (
    ask_dataset_question,
    suggest_ml_model,
    generate_preprocessing_code,
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Dataset Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM CSS ----------------
# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    /* ---------- THEME VARIABLES ---------- */
    :root {
        --card-bg: var(--background-color);
        --card-border: rgba(0,0,0,0.06);
        --card-shadow: 0 4px 18px rgba(0,0,0,0.05);
    }

    [data-theme="dark"] {
        --card-bg: rgba(30,30,30,0.6);
        --card-border: rgba(255,255,255,0.08);
        --card-shadow: 0 4px 18px rgba(0,0,0,0.4);
    }

    /* ---------- APP BACKGROUND ---------- */
    .stApp {
        background: var(--background-color);
    }

    /* ---------- HERO ---------- */
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #1f4b99 0%,
            #2d6cdf 45%,
            #5b8def 100%
        );
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(31, 75, 153, 0.18);
    }

    .hero h1 {
        font-size: 2rem;
        margin: 0;
    }

    .hero p {
        margin: 0.3rem 0 0 0;
        opacity: 0.95;
        font-size: 0.95rem;
    }

    /* ---------- SECTION CARD ---------- */
    .section-card {
        background: var(--card-bg);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        margin-bottom: 1rem;
    }

    /* ---------- METRICS ---------- */
    .stMetric {
        background: var(--card-bg);
        border-radius: 14px;
        border: 1px solid var(--card-border);
        padding: 0.8rem;
        box-shadow: var(--card-shadow);
    }

    /* ---------- EXPANDERS ---------- */
    div[data-testid="stExpander"] {
        background: var(--card-bg);
        border-radius: 14px;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
    }

    /* ---------- BUTTON ---------- */
    .stButton > button {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.15);
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px 10px 0 0;
        padding-left: 16px;
        padding-right: 16px;
    }

    /* ---------- CHAT ---------- */
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 0.6rem;
        border: 1px solid var(--card-border);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model_suggestion" not in st.session_state:
    st.session_state.model_suggestion = None

if "preprocessing_code" not in st.session_state:
    st.session_state.preprocessing_code = None

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>🤖 AI Dataset Copilot</h1>
        <p>Upload a dataset, inspect data quality, get ML guidance, generate preprocessing code, and ask natural-language questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
    help="Supported formats: CSV, XLSX",
)

if not uploaded_file:
    st.info("Upload a dataset to begin.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = load_data(uploaded_file)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Model Settings")
st.sidebar.caption("Choose the target column before asking for ML recommendations.")

target_column = st.sidebar.selectbox(
    "Select Target Column (optional)",
    ["None"] + list(df.columns),
)

if target_column == "None":
    target_column = None

st.sidebar.divider()
st.sidebar.caption("Quick dataset info")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

# ---------------- PROFILE / QUALITY / SCORE ----------------
profile = profile_dataset(df)
quality = run_quality_checks(df)
health_score, score_details = calculate_health_score(df, quality)
suggestions = generate_suggestions(df, quality, target_column)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Quality", "Visuals", "AI Tools", "Chat"]
)

# ---------------- OVERVIEW TAB ----------------
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", profile["rows"])
    c2.metric("Columns", profile["columns"])
    c3.metric("Duplicate Rows", profile["duplicate_rows"])

    st.markdown("**Missing Values Summary**")
    missing_df = pd.DataFrame(
        profile["missing_values"].items(),
        columns=["Column", "Missing Values"],
    )
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Health Score")
    score_cols = st.columns(5)
    score_cols[0].metric("Overall", f"{health_score} / 100")
    score_cols[1].metric("Completeness", score_details["completeness"])
    score_cols[2].metric("Duplicates", score_details["duplicates"])
    score_cols[3].metric("Correlation", score_details["correlation"])
    score_cols[4].metric("Outliers", score_details["outliers"])
    st.caption(f"Imbalance score: {score_details['imbalance']}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- QUALITY TAB ----------------
with tab2:
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Missing Value Issues")
        if len(quality["missing"]) > 0:
            st.dataframe(
                quality["missing"].reset_index().rename(
                    columns={"index": "Column", 0: "Missing %", "missing": "Missing %"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No missing values detected.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Constant Columns")
        if quality["constant_columns"]:
            st.write(quality["constant_columns"])
        else:
            st.success("No constant columns detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Duplicate Rows")
        st.metric("Duplicates", quality["duplicates"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Highly Correlated Features")
        if quality["high_correlation"]:
            st.write(quality["high_correlation"])
        else:
            st.success("No high correlation detected.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Class Imbalance")
        if quality["imbalance"]:
            st.json(quality["imbalance"])
        else:
            st.success("No major imbalance detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Outliers")
    if quality["outliers"]:
        st.json(quality["outliers"])
    else:
        st.success("No major outliers detected.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- VISUALS TAB ----------------
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Auto EDA Dashboard")
    st.caption("Charts are shown in a responsive grid with controlled figure sizes.")
    st.markdown("</div>", unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        fig = plot_missing_values(df)
        if fig:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Missing Values")
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        fig = plot_target_distribution(df, target_column)
        if fig:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Target Distribution")
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with v2:
        fig = plot_correlation(df)
        if fig:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Correlation Heatmap")
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Numeric Feature Distributions")
    figs = plot_numeric_distribution(df)
    if figs:
        grid = st.columns(2)
        for i, fig in enumerate(figs):
            with grid[i % 2]:
                st.pyplot(fig, use_container_width=True)
    else:
        st.info("No numeric columns available for distribution plots.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- AI TOOLS TAB ----------------
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Recommendations")
    if suggestions:
        for s in suggestions:
            st.write(f"• {s}")
    else:
        st.success("Dataset looks clean. No major suggestions.")
    st.markdown("</div>", unsafe_allow_html=True)

    a1, a2 = st.columns([1, 2])

    with a1:
        if st.button("Suggest ML Model", use_container_width=True):
            st.session_state.model_suggestion = suggest_ml_model(
                profile,
                quality,
                target_column,
                health_score,
                df,
            )

    with a2:
        if st.session_state.model_suggestion:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Model Recommendation")
            st.markdown(st.session_state.model_suggestion)
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Generate Preprocessing Code", use_container_width=True):
        st.session_state.preprocessing_code = generate_preprocessing_code(
            profile,
            quality,
            target_column,
            df,
        )

    if st.session_state.preprocessing_code:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Preprocessing Code")
        for title, code in st.session_state.preprocessing_code:
            st.subheader(title)
            st.code(code, language="python")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHAT TAB ----------------
with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Ask AI about Dataset")
    st.caption("Ask follow-up questions like: Is this dataset good for prediction? Which columns should I drop? What preprocessing should I apply?")
    st.markdown("</div>", unsafe_allow_html=True)

    question = st.chat_input("Ask about the dataset...")

    if question:
        answer = ask_dataset_question(
            question,
            profile,
            quality,
            health_score,
        )

        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])