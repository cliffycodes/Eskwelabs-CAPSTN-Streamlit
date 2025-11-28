
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


    st.subheader("🔍 Debug: Inputs Sent to Model")
    st.dataframe(input_df)
    st.write("Column types:", input_df.dtypes)
