
import streamlit as st
import pandas as pd
import joblib



@st.cache_resource
def load_model():
    return joblib.load("child_recode_model.pkl")

pipeline = load_model()


st.title("Infant Mortality Risk Prediction")

st.markdown(
    """
    This tool predicts the risk of infant mortality (0-11 months) based on prenatal factors.
    Please provide accurate information about the pregnancy and household context.
    """
)



st.header("💰 Socioeconomic Status")
v170 = st.checkbox("Does the mother have a bank account?")
v190 = st.selectbox("What is the household's wealth level?", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])


st.header("👨‍👩‍👧 Household & Pregnancy Details")
v136 = st.number_input("How many people live in the household?", min_value=1, max_value=30, value=5, step=1)
bord = st.number_input("What is the birth order of this child?", min_value=1, max_value=20, value=1, step=1)
m14 = st.number_input("How many antenatal visits did the mother have?", min_value=0, max_value=50, value=4, step=1)






import pandas as pd
import numpy as np
import streamlit as st

if st.button("🔮 Predict Risk"):
    
    load_model.clear()  
    pipeline = load_model()

    with st.spinner("Predicting infant mortality risk..."):

        bool_inputs = {
            'v170 - has an account in a bank or other financial institution': int(v170)
        }

        num_inputs = {
            'v136 - number of household members (listed)': v136,
            'bord - birth order number': bord,
            'm14 - number of antenatal visits during pregnancy': m14
        }

        # === DO NOT TOUCH: keep your reverse mapping exactly as provided ===
        reverse_mapping = {
            "Poorest": "Richest",
            "Poorer": "Richer",
            "Middle": "Middle",
            "Richer": "Poorer",
            "Richest": "Poorest"
        }

        ordinal_inputs = {
            'v190 - wealth index combined': reverse_mapping[v190]
        }

        all_inputs = {**bool_inputs, **num_inputs, **ordinal_inputs}
        input_df = pd.DataFrame([all_inputs])

        y_pred = pipeline.predict(input_df)[0]
        y_prob = float(pipeline.predict_proba(input_df)[0][1])

    if y_prob < 0.30:
        risk_level = "Low Risk"
        color = "green"
    elif y_prob < 0.70:
        risk_level = "Medium Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"

    st.markdown(f"### Risk Level: **<span style='color:{color}'>{risk_level}</span>**", unsafe_allow_html=True)
    st.write(f"Estimated probability of infant mortality: **{y_prob*100:.2f}%**")

    # ======================
    # Recommendations section
    # ======================
    recs = []

    # Base actions per risk level
    if risk_level == "Low Risk":
        recs.extend([
            "Maintain ANC schedule (target ≥ 4 total visits).",
            "Continue routine monitoring and health education."
        ])
    elif risk_level == "Medium Risk":
        recs.extend([
            "Schedule the next ANC visit immediately to reach ≥ 4 total visits.",
            "Plan a follow-up check in 2–4 weeks (nutrition/infection screening)."
        ])
    else:  # High Risk
        recs.extend([
            "Urgent referral for same-week ANC and physician assessment.",
            "Arrange transport/caregiver support; increase follow-up frequency."
        ])

    # Rule-based triggers from your EDA thresholds
    if bord > 4:
        recs.append("High childbirth order (>4): provide family planning counseling and close ANC follow-up.")
    if m14 < 3:
        recs.append("Low ANC (<3 visits): book the next ANC now and target at least 4 visits total.")
    if v136 < 3:
        recs.append("Small household (<3 members): mobilize family/community support and link to social programs.")
    if int(v170) == 0:
        recs.append("No bank account: connect to financial services or cash support to reduce barriers to care.")

    # Deduplicate while preserving order
    recs = list(dict.fromkeys(recs))

    st.subheader("✅ Recommended Next Steps")
    for r in recs:
        st.markdown(f"- {r}")

    # Debug
    st.subheader("🔍 Debug: Inputs Sent to Model")
    st.dataframe(input_df)
    st.write("Column types:", input_df.dtypes)
