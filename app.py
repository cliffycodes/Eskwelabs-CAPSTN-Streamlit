
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


    # Rule-based triggers from your EDA thresholds
    if bord > 4:
        recs.append("Mother has more than 4 childbirths. Assess the mother's situation and provide family planning counseling. Advise on proper prenatal and antenatal care.")
    if m14 < 3:
        recs.append("Mother has fewer than 3 antenatal care visits. Assess why this is the case and advise next steps or connect her to programs that can help increase visits.")
    if v136 < 3:
        recs.append("Mother lives in a household with fewer than 3 members. Explore her support system and link her to community or social programs for additional assistance.")
    if int(v170) == 0:
        recs.append("Mother does not have a bank account. Assess financial barriers and connect her to financial services or cash support programs to reduce obstacles to care.")

    # Wealth index recommendations (based on original selection `v190`, not the mapped value)
    if v190 in ["Poorest", "Poorer"]:
        recs.extend([
            "Costs may be a barrier for antenatal care, discuss free or low-cost services and community transport options.",
            "Suggest local programs or organizations that may provided financial support."
        ])


    # Deduplicate while preserving order
    recs = list(dict.fromkeys(recs))

    st.subheader("✅ Recommended Next Steps")
    for r in recs:
        st.markdown(f"- {r}")

    # # Debug
    # st.subheader("🔍 Debug: Inputs Sent to Model")
    # st.dataframe(input_df)
    # st.write("Column types:", input_df.dtypes)
