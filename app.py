import streamlit as st
import pandas as pd
import joblib
import time
import os

# -------------------------
# APP VARIABLES FOR TWEAKING/UPDATES 
# -------------------------
app_new_min = 1
app_new_max = 5


# -------------------------
# LOAD MODELS WITH FRONT END 
# -------------------------
# @st.cache_resource
# def load_model():
#     model_path = os.path.join("model", "final_knn_pipeline_raw.pkl")
#     return joblib.load(model_path)

# pipeline = load_model()

@st.cache_resource
def load_model():
    return joblib.load("final_knn_pipeline_raw.pkl")

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
m42d = st.checkbox("During pregnancy: Was urine sample taken?")
m57a = st.checkbox("Antenatal care provided at respondent's home?")
v170 = st.checkbox("Does the mother have an account in a bank or financial institution?")

# 👨‍👩‍👧 Household & Pregnancy Details (Numeric)
st.header("👨‍👩‍👧 Household & Pregnancy Details")

v136 = st.number_input(
    "Number of household members (listed):",
    min_value=1, max_value=30, value=5, step=1
)

bord = st.number_input(
    "Birth order number:",
    min_value=1, max_value=20, value=1, step=1
)

m14 = st.number_input(
    "Number of antenatal visits during pregnancy:",
    min_value=0, max_value=50, value=4, step=1
)

# 💰 Socioeconomic Status (Ordinal)
st.header("💰 Socioeconomic Status")

v190 = st.selectbox(
    "Wealth index combined:",
    ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
)

# -------------------------
# Variable Conversions
# -------------------------
# Numerical : Scale to old ranges + StandardScaling
# Categorical: One Hot encoding
# Boolean: Keep as is



#------------ Numerical----------- #


### Scaling conversion for numerical values (from 1-5 to the old actual data range) ###
# Function for conversion from new min max (1-5) to old min max 
def rescale_to_original(y_input, old_min, old_max, new_min=1.0, new_max=5.0):
\
    # cap the input between new_min and new_max
    y_capped = max(new_min, min(y_input, new_max))

    # linear rescale back to original
    x_original = old_min + (y_capped - new_min) * (old_max - old_min) / (new_max - new_min)
    return x_original








# -------------------------
# Prediction button
# -------------------------
if st.button("🔮 Predict"):
    with st.spinner("Predicting proficiency..."):

        # -------------------------
        # Rescale numerical inputs
        # -------------------------
        HISEI_CONVERTED     = rescale_to_original(HISEI, 11.01, 88.70, app_new_min, app_new_max)
        HOMEPOS_CONVERTED   = rescale_to_original(HOMEPOS, -7.863, 2.714, app_new_min, app_new_max)
        ICTRES_CONVERTED    = rescale_to_original(ICTRES, -5.060, 5.143, app_new_min, app_new_max)
        BULLIED_CONVERTED   = rescale_to_original(BULLIED, -1.228, 4.694, app_new_min, app_new_max)
        INFOSEEK_CONVERTED  = rescale_to_original(INFOSEEK, -2.421, 2.599, app_new_min, app_new_max)
        SCHRISK_CONVERTED   = rescale_to_original(SCHRISK, -0.639, 3.649, app_new_min, app_new_max)
        DISCLIM_CONVERTED   = rescale_to_original(DISCLIM, -2.493, 1.851, app_new_min, app_new_max)
        BELONG_CONVERTED    = rescale_to_original(BELONG, -3.258, 2.756, app_new_min, app_new_max)
        FAMSUP_CONVERTED    = rescale_to_original(FAMSUP, -3.063, 1.958, app_new_min, app_new_max)
        CREATAS_CONVERTED   = rescale_to_original(CREATAS, -1.121, 4.353, app_new_min, app_new_max)
        CREATFAM_CONVERTED  = rescale_to_original(CREATFAM, -2.789, 2.239, app_new_min, app_new_max)
        OPENART_CONVERTED   = rescale_to_original(OPENART, -2.815, 1.903, app_new_min, app_new_max)
        CREATSCH_CONVERTED  = rescale_to_original(CREATSCH, -2.623, 2.814, app_new_min, app_new_max)
        CREATOOS_CONVERTED  = rescale_to_original(CREATOOS, -0.821, 4.774, app_new_min, app_new_max)
        COGACRCO_CONVERTED  = rescale_to_original(COGACRCO, -2.862, 3.720, app_new_min, app_new_max)
        EXPOFA_CONVERTED    = rescale_to_original(EXPOFA, -2.085, 2.640, app_new_min, app_new_max)
        MATHPERS_CONVERTED  = rescale_to_original(MATHPERS, -3.096, 2.849, app_new_min, app_new_max)
        WORKPAY_CONVERTED   = rescale_to_original(WORKPAY, 0.0, 10.0, app_new_min, app_new_max)

        # -------------------------
        # Group Inputs by Type
        # -------------------------
        bool_inputs = {
            "REPEAT": REPEAT,
            "MISSSC": MISSSC,
        }

        cat_inputs = {
            "MISCED": MISCED,   
            "FISCED": FISCED,
            "LANGN": LANGN,
            "school_type": school_type,
            "urban_rural_proxy": urban_rural_proxy,
            "OCOD1_major_label": OCOD1_major_label,
        }

        num_inputs = {
            "ICTRES": ICTRES_CONVERTED,
            "HISEI": HISEI_CONVERTED,
            "BULLIED": BULLIED_CONVERTED,
            "INFOSEEK": INFOSEEK_CONVERTED,
            "CREATAS": CREATAS_CONVERTED,
            "HOMEPOS": HOMEPOS_CONVERTED,
            "MATHPERS": MATHPERS_CONVERTED,
            "CREATFAM": CREATFAM_CONVERTED,
            "OPENART": OPENART_CONVERTED,
            "SCHRISK": SCHRISK_CONVERTED,
            "WORKPAY": WORKPAY_CONVERTED,
            "CREATSCH": CREATSCH_CONVERTED,
            "DISCLIM": DISCLIM_CONVERTED,
            "BELONG": BELONG_CONVERTED,
            "CREATOOS": CREATOOS_CONVERTED,
            "COGACRCO": COGACRCO_CONVERTED,
            "EXPOFA": EXPOFA_CONVERTED,
            "FAMSUP": FAMSUP_CONVERTED,
        }

        all_inputs = {**bool_inputs, **cat_inputs, **num_inputs}
        input_df = pd.DataFrame([all_inputs])

        # -------------------------
        # Predict
        # -------------------------
        y_pred = pipeline.predict(input_df)[0]         # ✅ scalar
        y_prob = pipeline.predict_proba(input_df)[0][1] # ✅ probability of class 1

    # ✅ Display results after spinner closes
    if y_pred == 1:   # ✅ fixed: no extra indexing
        st.success(f"✅ Predicted: **Proficient**\n\nThis student has a **{y_prob*100:.2f}%** chance of being proficient.")
    else:
        st.error(f"❌ Predicted: **Not Proficient**\n\nThis student has only a **{y_prob*100:.2f}%** chance of being proficient.")

    # # -------------------------
    # # Debug: show all inputs
    # # -------------------------
    # st.subheader("🔍 Debug: Inputs Sent to Model")
    # st.dataframe(input_df)  # shows values and column names
    # st.write("Column types:", input_df.dtypes)  # shows data types

