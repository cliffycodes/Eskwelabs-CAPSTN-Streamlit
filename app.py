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

st.title("Student PISA 2022 Proficiency Prediction")

st.markdown(
    """
    This project allows you to predict a student's proficiency level in the PISA 2022 assessment 
    based on their personal, family, school, and social-emotional factors. 
    Please answer the questions below as accurately as possible to generate the prediction.
    """
)

# 🌍 Demographics & Background
st.header("🌍 Demographics & Background")

LANGN = st.selectbox(
    "What language does the student speak most often at home?",
    [
        "Filipino/Tagalog", "Cebuano/Bisaya", "Hiligaynon/Ilonggo", "Bikolano",
        "English", "Kapampangan", "Waray", "Pangasinan", "Other/Minor",
        "Maguindanao", "Ilocano", "Tausug", "Maranao", "Chavacano"
    ]
)

HISEI = st.number_input(
    "What is the highest occupational level of the student's parents? "
    "(1 = Elementary occupations such as service or manual jobs, "
    "5 = Professional/managerial occupations)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

MISCED = st.selectbox(
    "What is the mother’s highest education level?",
    [
        "Less than primary", "Primary education", "Lower secondary", "Upper secondary",
        "Post-secondary non-tertiary", "Short-cycle tertiary", "Bachelor’s or equivalent",
        "Master’s or equivalent", "Doctoral or equivalent"
    ]
)

FISCED = st.selectbox(
    "What is the father’s highest education level?",
    [
        "Less than primary", "Primary education", "Lower secondary", "Upper secondary",
        "Post-secondary non-tertiary", "Short-cycle tertiary", "Bachelor’s or equivalent",
        "Master’s or equivalent", "Doctoral or equivalent"
    ]
)

HOMEPOS = st.number_input(
    "How many household items or resources does the student’s family have at home? "
    "(This includes durable goods such as a study desk, books, internet connection, etc. "
    "1 = Very few possessions, 5 = Many possessions)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

ICTRES = st.number_input(
    "How many technological resources are available to the student at home or at school? "
    "(1 = Very limited access, 5 = Many technological resources)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

# 💬 Social-Emotional & School Climate
st.header("💬 Social-Emotional & School Climate")

BULLIED = st.number_input(
    "How often has the student experienced bullying at school? "
    "(1 = Never, 5 = Very frequently)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

INFOSEEK = st.number_input(
    "How often does the student actively seek out information for learning? "
    "(1 = Rarely, 5 = Very frequently)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

SCHRISK = st.number_input(
    "How safe does the student feel at school? "
    "(1 = Very safe, 5 = Very unsafe/disorderly environment)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

DISCLIM = st.number_input(
    "How would the student describe the classroom’s disciplinary climate? "
    "(1 = Very orderly, 5 = Very disorderly)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

BELONG = st.number_input(
    "How strongly does the student feel a sense of belonging at school? "
    "(1 = Feels very isolated, 5 = Strong sense of belonging)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

FAMSUP = st.number_input(
    "How supportive is the student’s family of their learning? "
    "(1 = Very unsupportive, 5 = Very supportive)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

# 💡 Cognitive, Creativity & Attitudes
st.header("💡 Cognitive, Creativity & Attitudes")

CREATAS = st.number_input(
    "How creative is the student in completing assigned schoolwork? "
    "(1 = Very creative, 5 = Not creative at all)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

CREATFAM = st.number_input(
    "How much does the student’s family encourage creativity? "
    "(1 = Not at all, 5 = Very much)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

OPENART = st.number_input(
    "How open is the student to artistic and cultural experiences? "
    "(1 = Not open, 5 = Very open)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

CREATSCH = st.number_input(
    "How many opportunities for creativity does the student have at school? "
    "(1 = Very few, 5 = Many opportunities)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

CREATOOS = st.number_input(
    "How much does the student engage in creative activities outside of school? "
    "(1 = Very active in creativity outside school, 5 = Not active at all)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

COGACRCO = st.number_input(
    "How often does the student participate in reasoning-based learning activities? "
    "(1 = Rarely, 5 = Very frequently)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

EXPOFA = st.number_input(
    "How much exposure does the student have to future-oriented activities "
    "(e.g., career talks, internships)? (1 = Very little, 5 = A lot)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

MATHPERS = st.number_input(
    "How perseverant is the student in solving mathematics tasks? "
    "(1 = Gives up easily, 5 = Keeps trying until successful)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1
)

# 🕓 Student Background & Engagement
st.header("🕓 Student Background & Engagement")

WORKPAY = st.number_input(
    "How many hours does the student spend in paid work outside school? "
    "(1 = No paid work, 5 = Works many hours per week)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

REPEAT = st.checkbox("Has the student ever repeated a grade?")

MISSSC = st.checkbox("Has the student missed several days of school (absenteeism)?")

# 🏫 School Context & Career Orientation
st.header("🏫 School Context & Career Orientation")

school_type = st.selectbox(
    "What type of school does the student attend?",
    ["Public", "Private (Govt-Dependent)", "Private (Independent)"]
)

urban_rural_proxy = st.selectbox(
    "Where is the student’s school located?",
    ["Urban", "Rural", "Semi-urban"]
)

OCOD1_major_label = st.selectbox(
    "What career or occupational cluster is the student most interested in?",
    [
        "No response", "Elementary Occupations", "Service & Sales Workers", "Professionals",
        "Technicians & Associate Professionals", "Managers", "Skilled Agricultural Workers",
        "Craft & Related Trades Workers", "Clerical Support Workers", "Plant & Machine Operators"
    ]
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

