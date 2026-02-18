import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import run_pipeline
import json

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="Social Support AI",
    page_icon="🏛️",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .approve-box {
        background: #dcfce7; border: 2px solid #16a34a;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .decline-box {
        background: #fee2e2; border: 2px solid #dc2626;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .agent-msg {
        background: #1e293b; color: #94a3b8;
        font-family: monospace; font-size: 13px;
        padding: 8px 14px; border-radius: 6px; margin: 4px 0;
    }
    .section-title {
        font-size: 18px; font-weight: 700;
        color: #1e293b; margin: 20px 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────
st.title("🏛️ Social Support AI — Abu Dhabi Government")
st.caption("AI-powered social support eligibility assessment · Powered by LangGraph + Ollama (llama3)")
st.divider()

# ── LAYOUT ────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

# ── LEFT — APPLICATION FORM ───────────────────────────
with col_form:
    st.markdown('<div class="section-title">📋 Applicant Information</div>', unsafe_allow_html=True)

    with st.form("application_form"):
        # Personal
        st.subheader("Personal Details")
        name        = st.text_input("Full Name", placeholder="Ahmed Al Mansouri")
        nationality = st.selectbox("Nationality", ["Emirati", "Expat"])
        age         = st.number_input("Age", min_value=18, max_value=80, value=35)

        st.divider()

        # Employment
        st.subheader("Employment")
        employment_status = st.selectbox(
            "Employment Status",
            ["employed", "unemployed", "part_time", "self_employed"]
        )
        years_employed = st.number_input("Years Employed", min_value=0, max_value=40, value=0)
        monthly_income = st.number_input("Monthly Income (AED)", min_value=0.0, value=1200.0, step=100.0)
        monthly_expenses = st.number_input("Monthly Expenses (AED)", min_value=0.0, value=1400.0, step=100.0)

        st.divider()

        # Family
        st.subheader("Family & Housing")
        family_size  = st.number_input("Family Size", min_value=1, max_value=15, value=5)
        dependents   = st.number_input("Number of Dependents", min_value=0, max_value=15, value=3)
        housing_type = st.selectbox("Housing Type", ["rented", "owned", "family", "government"])

        st.divider()

        # Financial
        st.subheader("Financial Profile")
        total_assets      = st.number_input("Total Assets (AED)", min_value=0.0, value=5000.0, step=1000.0)
        total_liabilities = st.number_input("Total Liabilities (AED)", min_value=0.0, value=8000.0, step=1000.0)
        credit_score      = st.slider("Credit Score", min_value=300, max_value=850, value=420)

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Submit Application",
            use_container_width=True,
            type="primary"
        )

# ── RIGHT — RESULTS ───────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">🤖 AI Assessment</div>', unsafe_allow_html=True)

    if not submitted:
        st.info("Fill in the form on the left and click **Submit Application** to begin the AI assessment.")
        st.markdown("""
        **The AI pipeline will:**
        - 🔍 Extract & structure your data
        - ✅ Validate for inconsistencies  
        - ⚖️ Run ML eligibility model (91% accuracy)
        - 📋 Generate decision + recommendations
        """)

    else:
        # Build applicant dict
        applicant = {
            "name":               name,
            "nationality":        nationality,
            "age":                age,
            "employment_status":  employment_status,
            "years_employed":     years_employed,
            "monthly_income":     monthly_income,
            "monthly_expenses":   monthly_expenses,
            "family_size":        family_size,
            "dependents":         dependents,
            "housing_type":       housing_type,
            "total_assets":       total_assets,
            "total_liabilities":  total_liabilities,
            "credit_score":       credit_score
        }

        # Run pipeline with live agent status
        st.markdown("**🔄 Running AI Pipeline...**")
        status_box = st.empty()

        with st.spinner("Agents processing your application..."):
            # Show live agent steps
            steps = [
                "📤 Agent 1: Extracting & structuring data...",
                "✅ Agent 2: Validating data consistency...",
                "⚖️ Agent 3: Running ML eligibility model...",
                "📋 Agent 4: Generating decision & recommendations..."
            ]
            for step in steps:
                status_box.markdown(f'<div class="agent-msg">{step}</div>', unsafe_allow_html=True)

            result = run_pipeline(applicant)

        status_box.empty()

        # ── AGENT LOG ──────────────────────────────────
        with st.expander("🔍 View Agent Pipeline Log", expanded=False):
            for msg in result.get("messages", []):
                st.markdown(f'<div class="agent-msg">{msg}</div>', unsafe_allow_html=True)

        st.divider()

        # ── DECISION CARD ──────────────────────────────
        final    = result.get("final_output", {})
        ml       = result.get("ml_result", {})
        decision = final.get("final_decision", "DECLINE")

        if decision == "APPROVE":
            st.markdown(f"""
            <div class="approve-box">
                <h2>✅ APPROVED</h2>
                <p style="font-size:18px;font-weight:600;">Confidence: {ml.get('confidence', 0)}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decline-box">
                <h2>⚠️ NOT ELIGIBLE</h2>
                <p style="font-size:18px;font-weight:600;">Confidence: {ml.get('confidence', 0)}%</p>
            </div>
            """, unsafe_allow_html=True)

        # ── DECISION REASON ────────────────────────────
        st.markdown("**📝 Decision Explanation:**")
        st.info(final.get("decision_reason", ""))

        # ── VALIDATION FLAGS ───────────────────────────
        validation = result.get("validation", {})
        flags      = validation.get("flags", [])
        if flags:
            st.markdown("**⚠️ Validation Flags:**")
            for flag in flags:
                st.warning(flag)

        # ── SHAP EXPLANATION ───────────────────────────
        explanation = ml.get("explanation", {})
        if explanation:
            st.markdown("**📊 Why this decision? (SHAP Feature Impact)**")
            import pandas as pd
            shap_df = pd.DataFrame(
                list(explanation.items()),
                columns=["Feature", "Impact"]
            ).sort_values("Impact", ascending=False)
            st.bar_chart(shap_df.set_index("Feature"))

        # ── ENABLEMENT RECS ────────────────────────────
        recs = final.get("enablement_recommendations", [])
        if recs:
            st.markdown("**🎓 Economic Enablement Recommendations:**")
            for rec in recs:
                st.success(f"✅ {rec}")

        # ── NEXT STEPS ─────────────────────────────────
        next_steps = final.get("next_steps", "")
        if next_steps:
            st.markdown("**➡️ Next Steps:**")
            st.write(next_steps)

