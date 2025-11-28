
import streamlit as st
import pandas as pd
import joblib

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource  
def load_model():
    return joblib.load("child_recode_model.pkl")

pipeline = load_model()

# -------------------------
# FRONT END / QUESTIONNAIRE
# -------------------------
st.title("Infant Mortality Risk Prediction")

st.markdown(
    """
    This tool predicts the risk of infant mortality (0-11 months) based on prenatal factors.
    Please provide accurate information about the pregnancy and household context.
    """
)

# 🩺 Prenatal Care Indicators (Boolean)
st.header("🩺 Prenatal Care Indicators")
m42c = st.checkbox("During pregnancy: Was blood pressure taken?")
v170 = st.checkbox("Does the mother have an account in a bank or financial institution?")

# 👨‍👩‍👧 Household & Pregnancy Details (Numeric)
st.header("👨‍👩‍👧 Household & Pregnancy Details")
v136 = st.number_input("Number of household members (listed):", min_value=1, max_value=30, value=5, step=1)
bord = st.number_input("Birth order number:", min_value=1, max_value=20, value=1, step=1)
m14 = st.number_input("Number of antenatal visits during pregnancy:", min_value=0, max_value=50, value=4, step=1)

# 💰 Socioeconomic Status (Ordinal)
st.header("💰 Socioeconomic Status")
v190 = st.selectbox("Wealth index combined:", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])

# -------------------------
# Prediction Button
# -------------------------
if st.button("🔮 Predict Risk"):
    
    load_model.clear()  # clears @st.cache_resource for this function
    pipeline = load_model()

    with st.spinner("Predicting infant mortality risk..."):

        # Boolean inputs (convert to int)
        bool_inputs = {
            'm42c - during pregnancy: blood pressure taken': int(m42c),
            'v170 - has an account in a bank or other financial institution': int(v170)
        }

        # Numeric inputs
        num_inputs = {
            'v136 - number of household members (listed)': v136,
            'bord - birth order number': bord,
            'm14 - number of antenatal visits during pregnancy': m14
        }

        
        # Reverse wealth index meaning
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

        # Combine all inputs
        all_inputs = {**bool_inputs, **num_inputs, **ordinal_inputs}
        input_df = pd.DataFrame([all_inputs])

        # Predict
        y_pred = pipeline.predict(input_df)[0]
        y_prob = pipeline.predict_proba(input_df)[0][1]

    # -------------------------
    # Risk Categorization
    # -------------------------
    if y_prob < 0.33:
        risk_level = "Low Risk"
        color = "green"
    elif y_prob < 0.66:
        risk_level = "Medium Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"

    
    st.markdown(f"### Risk Level: **<span style='color:{color}'>{risk_level}</span>**", unsafe_allow_html=True)
    st.write(f"Estimated probability of infant mortality: **{y_prob*100:.2f}%**")


    st.subheader("🔍 Debug: Inputs Sent to Model")
    st.dataframe(input_df)
    st.write("Column types:", input_df.dtypes)
