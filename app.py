import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e2e8f0; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; }
    .metric-card {
        background: #f8fafc; border-radius: 12px; padding: 1.2rem;
        border-left: 5px solid #3b82f6; margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .approved-box {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981; border-radius: 15px;
        padding: 2rem; text-align: center;
    }
    .rejected-box {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444; border-radius: 15px;
        padding: 2rem; text-align: center;
    }
    .section-header {
        background: #f1f5f9; padding: 0.6rem 1rem; border-radius: 8px;
        margin: 1.2rem 0 0.8rem; font-weight: 600; color: #1e40af;
        border-left: 4px solid #3b82f6;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white; border: none; border-radius: 10px;
        padding: 0.8rem 2rem; font-size: 1.1rem; font-weight: 600;
        width: 100%; transition: all 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59,130,246,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("loan_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

try:
    model, FEATURE_NAMES = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model files: {e}\nMake sure `loan_model.pkl` and `feature_names.pkl` are in the same directory.")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏦 Loan Approval Predictor</h1>
    <p>AI-powered instant loan eligibility assessment using XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — About ───────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/100/bank.png", width=80)
    st.markdown("### About This App")
    st.markdown("""
    This app uses a **XGBoost classifier** trained on 4,269 loan applications to predict
    whether a loan will be **Approved** or **Rejected**.

    **Key features used:**
    - 📊 CIBIL Credit Score
    - 💰 Annual Income
    - 🏠 Total Assets
    - 💳 Loan Amount & Term
    - 👨‍👩‍👧 Dependents & Profile

    **Model Performance:**
    - ✅ Test Accuracy: ~98%
    - ✅ AUC-ROC: ~99%
    """)
    st.divider()
    st.markdown("**Developed with** 🐍 Python, Scikit-learn, XGBoost, Streamlit")

# ── Main Form ─────────────────────────────────────────────────
if model_loaded:
    st.markdown("### 📋 Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-header'>👤 Personal Info</div>", unsafe_allow_html=True)
        no_of_dependents = st.slider("Number of Dependents", 0, 5, 2)
        education        = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed    = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        st.markdown("<div class='section-header'>💰 Financial Details</div>", unsafe_allow_html=True)
        income_annum  = st.number_input("Annual Income (₹)", min_value=100000, max_value=10000000,
                                         value=5000000, step=100000,
                                         format="%d", help="Enter annual income in Rupees")
        loan_amount   = st.number_input("Loan Amount (₹)", min_value=500000, max_value=40000000,
                                         value=15000000, step=500000, format="%d")
        loan_term     = st.select_slider("Loan Term (Years)", options=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20], value=10)
        cibil_score   = st.slider("CIBIL Score", 300, 900, 650,
                                   help="Credit score between 300-900. Higher is better.")

    with col3:
        st.markdown("<div class='section-header'>🏠 Asset Values (₹)</div>", unsafe_allow_html=True)
        residential_assets = st.number_input("Residential Assets (₹)", 0, 50000000, 5000000,  500000, format="%d")
        commercial_assets  = st.number_input("Commercial Assets (₹)",  0, 50000000, 3000000,  500000, format="%d")
        luxury_assets      = st.number_input("Luxury Assets (₹)",      0, 50000000, 8000000,  500000, format="%d")
        bank_asset         = st.number_input("Bank Assets (₹)",        0, 50000000, 2000000,  500000, format="%d")

    st.divider()

    # ── Predict Button ─────────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔍 Predict Loan Approval")

    if predict_clicked:
        # Build feature vector
        total_assets       = residential_assets + commercial_assets + luxury_assets + bank_asset
        loan_to_income     = loan_amount / (income_annum + 1)
        asset_to_loan      = total_assets / (loan_amount + 1)
        income_per_dep     = income_annum / (no_of_dependents + 1)
        education_enc      = 1 if education == "Graduate" else 0
        self_employed_enc  = 1 if self_employed == "Yes" else 0

        feature_values = {
            'no_of_dependents':          no_of_dependents,
            'education':                 education_enc,
            'self_employed':             self_employed_enc,
            'income_annum':              income_annum,
            'loan_amount':               loan_amount,
            'loan_term':                 loan_term,
            'cibil_score':               cibil_score,
            'residential_assets_value':  residential_assets,
            'commercial_assets_value':   commercial_assets,
            'luxury_assets_value':       luxury_assets,
            'bank_asset_value':          bank_asset,
            'total_assets':              total_assets,
            'loan_to_income_ratio':      loan_to_income,
            'asset_to_loan_ratio':       asset_to_loan,
            'income_per_dependent':      income_per_dep
        }

        # Align with training feature order
        input_df = pd.DataFrame([{f: feature_values.get(f, 0) for f in FEATURE_NAMES}])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        prob_approved = probability[1]
        prob_rejected = probability[0]

        st.divider()
        st.markdown("## 🎯 Prediction Result")
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class='approved-box'>
                    <h2 style='color:#065f46; margin:0;'>✅ LOAN APPROVED</h2>
                    <p style='font-size:1.2rem; color:#047857;'>
                        Approval Probability: <strong>{prob_approved*100:.1f}%</strong>
                    </p>
                    <p style='color:#6b7280;'>The applicant meets the loan eligibility criteria.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='rejected-box'>
                    <h2 style='color:#7f1d1d; margin:0;'>❌ LOAN REJECTED</h2>
                    <p style='font-size:1.2rem; color:#b91c1c;'>
                        Rejection Probability: <strong>{prob_rejected*100:.1f}%</strong>
                    </p>
                    <p style='color:#6b7280;'>The application does not meet eligibility criteria.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Probability Breakdown")
            st.metric("Approved",  f"{prob_approved*100:.1f}%")
            st.metric("Rejected",  f"{prob_rejected*100:.1f}%")
            st.progress(float(prob_approved))

        # Summary table
        st.divider()
        st.markdown("#### 📊 Application Summary")
        summary_data = {
            "Feature": ["Annual Income", "Loan Amount", "CIBIL Score", "Loan Term",
                         "Total Assets", "Loan-to-Income Ratio", "Asset-to-Loan Ratio"],
            "Value": [
                f"₹{income_annum:,}",
                f"₹{loan_amount:,}",
                str(cibil_score),
                f"{loan_term} years",
                f"₹{total_assets:,}",
                f"{loan_to_income:.2f}x",
                f"{asset_to_loan:.2f}x"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Tips
        st.divider()
        if prediction == 0:
            st.markdown("#### 💡 Tips to Improve Eligibility")
            tips = []
            if cibil_score < 600:
                tips.append("📈 **Improve your CIBIL score** to at least 700 by clearing existing dues.")
            if loan_to_income > 5:
                tips.append("📉 **Reduce loan amount** or increase income — loan-to-income ratio is high.")
            if asset_to_loan < 1:
                tips.append("🏠 **Build more assets** to improve your asset-to-loan coverage ratio.")
            if not tips:
                tips.append("⏳ Consider applying after improving your financial profile over the next 6-12 months.")
            for tip in tips:
                st.markdown(f"- {tip}")

else:
    st.warning("Model files not found. Please ensure `loan_model.pkl` and `feature_names.pkl` are in the app directory.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>
    🏦 Loan Approval Predictor | XGBoost ML Model | Trained on 4,269 loan applications
    <br>⚠️ For demonstration purposes only. Not intended for actual financial decisions.
</p>
""", unsafe_allow_html=True)
