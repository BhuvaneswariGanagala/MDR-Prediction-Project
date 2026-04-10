import streamlit as st
import pandas as pd
import joblib
import shap

# ===============================
# 🔥 LOAD MODEL + COLUMNS
# ===============================
model = joblib.load("xgb_model.pkl")
columns = joblib.load("columns.pkl")

# ===============================
# 🔥 PAGE CONFIG
# ===============================
st.set_page_config(page_title="🧬 Antibiotic Predictor", layout="wide")

st.title("🧬 Antibiotic Resistance Predictor")
st.markdown("Predict resistance & understand **WHY (Explainable AI using SHAP)**")

# ===============================
# 🔥 USER INPUT FORM
# ===============================
st.sidebar.header("🧾 Patient Details")

age = st.sidebar.number_input("Age", 0, 120, 30)
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
hospital = st.sidebar.selectbox("Hospital History", ["No", "Yes"])
infection = st.sidebar.slider("Infection Frequency", 0, 10, 1)
gender = st.sidebar.selectbox("Gender", ["M", "F"])

bacteria = st.sidebar.selectbox(
    "Bacteria Type",
    [
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Enterobacter spp.",
        "Proteus mirabilis"
    ]
)

# Convert inputs
diabetes = 1 if diabetes == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0
hospital = 1 if hospital == "Yes" else 0

# ===============================
# 🔥 PREPARE INPUT DATA
# ===============================
input_df = pd.DataFrame([{
    'Age': age,
    'Diabetes': diabetes,
    'Hypertension': hypertension,
    'Hospital_before': hospital,
    'Infection_Freq': infection,
    'Gender': gender,
    'Bacteria': bacteria
}])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)
input_df = input_df.astype(int)

# ===============================
# 🔥 PREDICTION BUTTON
# ===============================
if st.button("🔍 Predict"):

    st.subheader("📊 Prediction Results")

    probs = model.predict_proba(input_df)

    drugs = ['CIP', 'AN', 'ofx', 'Co-trimoxazole', 'Furanes']

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    results = {}

    for i, drug in enumerate(drugs):

        prob_array = probs[i]

        if len(prob_array.shape) == 2:
            prob = prob_array[0][1]
        else:
            prob = prob_array[1]

        label = "Resistant ❌" if prob > 0.5 else "Sensitive ✅"
        results[drug] = label

        with cols[i]:
            st.metric(
                label=drug,
                value=label,
                delta=f"{prob:.2f} confidence"
            )

    # ===============================
    # 🔥 SHAP EXPLANATION
    # ===============================
    st.subheader("🧠 Why this prediction? (SHAP Explanation)")

    shap_explainers = [
        shap.TreeExplainer(est) for est in model.estimators_
    ]

    feature_names_map = {
        "Hospital_before": "Hospital history",
        "Infection_Freq": "Frequent infections",
        "Diabetes": "Diabetes",
        "Hypertension": "Blood pressure",
        "Gender_M": "Male",
        "Gender_F": "Female"
    }

    for i, drug in enumerate(drugs):

        st.markdown(f"### 🔹 {drug}")

        explainer = shap_explainers[i]
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            if results[drug] == "Resistant ❌":
                values = shap_values[1][0]
            else:
                values = shap_values[0][0]
        else:
            values = shap_values[0]

        feature_contrib = list(zip(columns, values))
        feature_contrib = sorted(
            feature_contrib,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feat, val in feature_contrib[:3]:

            name = feature_names_map.get(feat, feat)

            if val > 0:
                st.write(f"🟢 {name} increased resistance")
            else:
                st.write(f"🔵 {name} reduced resistance")

# import streamlit as st
# import pandas as pd
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import google.generativeai as genai

# # ================================
# # 🔧 PAGE CONFIG
# # ================================
# st.set_page_config(page_title="AMR Predictor", layout="wide")

# st.title("🧬 Antibiotic Resistance Predictor")
# st.write("App started successfully ✅")

# # ================================
# # 🔥 LOAD MODEL + FEATURES
# # ================================
# @st.cache_resource
# def load_assets():
#     model = joblib.load('xgb_model.pkl')
#     features = joblib.load('columns.pkl')
#     return model, features

# model, features = load_assets()

# # ================================
# # 🔑 GEMINI CONFIG (FIXED)
# # ================================
# genai.configure(api_key="AIzaSyBv8Sr5EOp5aufWHtC5k7l8i98QbYsCpCg")   # 🔥 put your key
# llm = genai.GenerativeModel("models/gemini-2.5-flash")

# # =========================
# # 🔁 FALLBACK FUNCTION
# # =========================
# def fallback_suggestions(patient_data, tested_drugs, sensitive_drugs, error_msg=None):

#     bacteria = patient_data['Bacteria'].lower()

#     if "coli" in bacteria:
#         suggestions = [
#             "Nitrofurantoin → effective for urinary E.coli infections",
#             "Fosfomycin → good for resistant E.coli",
#             "Amoxicillin-clavulanate → common alternative"
#         ]
#     elif "klebsiella" in bacteria:
#         suggestions = [
#             "Cefepime → broad-spectrum cephalosporin",
#             "Meropenem → used for resistant infections",
#             "Piperacillin-tazobactam → strong hospital antibiotic"
#         ]
#     else:
#         suggestions = [
#             "Ceftriaxone → broad-spectrum antibiotic",
#             "Levofloxacin → alternative fluoroquinolone",
#             "Doxycycline → commonly used antibiotic"
#         ]

#     text = "⚠️ LLM unavailable. Showing rule-based suggestions:\n\n"

#     for s in suggestions:
#         text += f"- {s}\n"

#     if sensitive_drugs:
#         text += "\n💊 Already effective drugs:\n"
#         for d in sensitive_drugs:
#             text += f"- {d}\n"

#     if error_msg:
#         text += f"\n(LLM Error: {error_msg})"

#     return text

# # =========================
# # 🤖 LLM FUNCTION
# # =========================
# def get_llm_suggestions(patient_data, tested_drugs, sensitive_drugs):

#     prompt = f"""
# Patient details:
# Age: {patient_data['Age']}
# Gender: {patient_data['Gender']}
# Diabetes: {patient_data['Diabetes']}
# Hypertension: {patient_data['Hypertension']}
# Hospital history: {patient_data['Hospital_before']}
# Infection frequency: {patient_data['Infection_Freq']}
# Bacteria: {patient_data['Bacteria']}

# Model results:
# Sensitive drugs: {sensitive_drugs}
# Tested drugs: {tested_drugs}

# Task:
# Suggest NEW antibiotics NOT in tested drugs list.
# Do NOT repeat any of these: {tested_drugs}
# Give 3-5 alternative antibiotics with short reasoning.
# """

#     try:
#         response = llm.generate_content(prompt)

#         if response and hasattr(response, "text") and response.text.strip():
#             return response.text
#         else:
#             raise ValueError("Empty response")

#     except Exception as e:
#         return fallback_suggestions(patient_data, tested_drugs, sensitive_drugs, str(e))

# # ================================
# # 📋 SIDEBAR INPUTS
# # ================================
# st.sidebar.header("📋 Patient Information")

# age = st.sidebar.number_input("Age", 0, 120, 45)
# gender = st.sidebar.selectbox("Gender", ["M", "F"])
# diabetes = st.sidebar.toggle("Diabetes")
# hypertension = st.sidebar.toggle("Hypertension")
# hospital = st.sidebar.toggle("Hospitalized recently")
# inf_freq = st.sidebar.slider("Infection Frequency", 0, 10, 1)
# bacteria = st.sidebar.text_input("Bacteria", "Escherichia coli")

# # Normalize bacteria
# bacteria = bacteria.strip().lower()
# bacteria_map = {
#     "ecoli": "Escherichia coli",
#     "e coli": "Escherichia coli",
#     "e.coli": "Escherichia coli",
#     "klebsiella": "Klebsiella pneumoniae"
# }
# bacteria = bacteria_map.get(bacteria, bacteria.title())

# # ================================
# # 🔍 MAIN PREDICTION
# # ================================
# st.markdown("### 🦠 Prediction Results")

# if st.sidebar.button("Run Analysis"):

#     # --------------------------
#     # PREPARE INPUT
#     # --------------------------
#     data = {
#         'Age': age,
#         'Diabetes': int(diabetes),
#         'Hypertension': int(hypertension),
#         'Hospital_before': int(hospital),
#         'Infection_Freq': inf_freq,
#         'Gender': gender,
#         'Bacteria': bacteria
#     }

#     input_df = pd.DataFrame([data])
#     input_df = pd.get_dummies(input_df)
#     input_df = input_df.reindex(columns=features, fill_value=0)

#     # --------------------------
#     # PREDICTIONS
#     # --------------------------
#     probs = model.predict_proba(input_df)

#     drugs = ['CIP', 'AN', 'ofx', 'Co-trimoxazole', 'Furanes']
#     cols = st.columns(len(drugs))

#     sensitive_list = []

#     for i, drug in enumerate(drugs):

#         prob_array = probs[i]

#         if len(prob_array.shape) == 2:
#             prob = prob_array[0][1]
#         else:
#             prob = prob_array[1]

#         is_resistant = prob > 0.5

#         with cols[i]:
#             st.metric(drug, "Resistant ❌" if is_resistant else "Sensitive ✅")
#             st.progress(float(prob))
#             st.caption(f"{prob:.1%}")

#             if not is_resistant:
#                 sensitive_list.append(drug)

#     # ================================
#     # 🧠 SHAP EXPLANATION
#     # ================================
#     st.divider()
#     st.subheader("🧠 Model Reasoning")

#     try:
#         explainer = shap.TreeExplainer(model.estimators_[0])
#         shap_values = explainer.shap_values(input_df)

#         fig, ax = plt.subplots()

#         shap.plots.waterfall(
#             shap.Explanation(
#                 values=shap_values[0],
#                 base_values=explainer.expected_value,
#                 data=input_df.iloc[0],
#                 feature_names=features
#             ),
#             show=False
#         )

#         st.pyplot(fig)

#     except Exception as e:
#         st.warning(f"SHAP error: {e}")

#     # ================================
#     # 🤖 LLM + FALLBACK
#     # ================================
#     st.divider()
#     st.subheader("🤖 AI Suggested Treatments")

#     patient_data = {
#         'Age': age,
#         'Gender': gender,
#         'Diabetes': int(diabetes),
#         'Hypertension': int(hypertension),
#         'Hospital_before': int(hospital),
#         'Infection_Freq': inf_freq,
#         'Bacteria': bacteria
#     }

#     tested_drugs = drugs

#     with st.spinner("Analyzing..."):
#         output = get_llm_suggestions(patient_data, tested_drugs, sensitive_list)

#     st.text(output)

# else:
#     st.info("👈 Enter details and click Run Analysis")
# # import streamlit as st
# # import pandas as pd
# # import joblib
# # import shap
# # import matplotlib.pyplot as plt
# # import google.generativeai as genai
# # import os

# # # ================================
# # # 🔧 PAGE CONFIG
# # # ================================
# # st.set_page_config(page_title="AMR Predictor", layout="wide")

# # st.title("🧬 Antibiotic Resistance Predictor")
# # st.write("App started successfully ✅")

# # # ================================
# # # 🔥 LOAD MODEL + FEATURES
# # # ================================
# # @st.cache_resource
# # def load_assets():
# #     model = joblib.load('xgb_model.pkl')
# #     features = joblib.load('columns.pkl')
# #     return model, features

# # model, features = load_assets()

# # # ================================
# # # 🔑 GEMINI CONFIG (SAFE WAY)
# # # ================================
# # genai.configure(api_key=os.getenv("AIzaSyCos10LoEVtpjwu7Mipb6k3QWFZUsdxYkw"))
# # llm = genai.GenerativeModel("models/gemini-2.5-flash")

# # # ================================
# # # 📋 SIDEBAR INPUTS
# # # ================================
# # st.sidebar.header("📋 Patient Information")

# # age = st.sidebar.number_input("Age", 0, 120, 45)

# # gender = st.sidebar.selectbox("Gender", ["M", "F"])

# # diabetes = st.sidebar.toggle("Diabetes")
# # hypertension = st.sidebar.toggle("Hypertension")
# # hospital = st.sidebar.toggle("Hospitalized recently")

# # inf_freq = st.sidebar.slider("Infection Frequency (past year)", 0, 10, 1)

# # bacteria = st.sidebar.text_input("Bacteria Species", "Escherichia coli")

# # # ================================
# # # 🧪 NORMALIZE BACTERIA
# # # ================================
# # bacteria = bacteria.strip().lower()

# # bacteria_map = {
# #     "ecoli": "Escherichia coli",
# #     "e coli": "Escherichia coli",
# #     "e.coli": "Escherichia coli",
# #     "klebsiella": "Klebsiella pneumoniae"
# # }

# # bacteria = bacteria_map.get(bacteria, bacteria.title())

# # # ================================
# # # 🔍 MAIN PREDICTION
# # # ================================
# # st.title("🦠 Antibiotic Multiresistance Prediction")
# # st.markdown("Predicting resistance for: **CIP, AN, OFX, Co-trimoxazole, Furanes**")

# # if st.sidebar.button("Run Analysis"):

# #     # --------------------------
# #     # PREPARE INPUT
# #     # --------------------------
# #     data = {
# #         'Age': age,
# #         'Diabetes': int(diabetes),
# #         'Hypertension': int(hypertension),
# #         'Hospital_before': int(hospital),
# #         'Infection_Freq': inf_freq,
# #         'Gender': gender,
# #         'Bacteria': bacteria
# #     }

# #     input_df = pd.DataFrame([data])

# #     # Encoding
# #     input_df = pd.get_dummies(input_df)

# #     # Align with training features
# #     input_df = input_df.reindex(columns=features, fill_value=0)

# #     # --------------------------
# #     # PREDICTIONS
# #     # --------------------------
# #     probs = model.predict_proba(input_df)

# #     drugs = ['CIP', 'AN', 'ofx', 'Co-trimoxazole', 'Furanes']

# #     cols = st.columns(len(drugs))
# #     sensitive_list = []

# #     for i, drug in enumerate(drugs):

# #         prob_array = probs[i]

# #         # Safe probability extraction
# #         if len(prob_array.shape) == 2:
# #             prob = prob_array[0][1]
# #         else:
# #             prob = prob_array[1]

# #         is_resistant = prob > 0.5

# #         with cols[i]:
# #             st.metric(label=drug, value="Resistant ❌" if is_resistant else "Sensitive ✅")
# #             st.progress(float(prob))
# #             st.caption(f"Confidence: {prob:.1%}")

# #             if not is_resistant:
# #                 sensitive_list.append(drug)

# #     # ================================
# #     # 🧠 SHAP EXPLANATION
# #     # ================================
# #     st.divider()
# #     st.subheader("🧠 Model Reasoning (SHAP)")

# #     try:
# #         # Use first antibiotic (CIP)
# #         explainer = shap.TreeExplainer(model.estimators_[0])
# #         shap_values = explainer.shap_values(input_df)

# #         fig, ax = plt.subplots()

# #         shap.plots.waterfall(
# #             shap.Explanation(
# #                 values=shap_values[0],
# #                 base_values=explainer.expected_value,
# #                 data=input_df.iloc[0],
# #                 feature_names=features
# #             ),
# #             show=False
# #         )

# #         st.pyplot(fig)

# #     except Exception as e:
# #         st.warning(f"SHAP error: {e}")

# #     # ================================
# #     # 🤖 GEMINI INSIGHTS
# #     # ================================
# #     st.divider()
# #     st.subheader("🤖 AI Clinical Insights")

# #     with st.spinner("Analyzing alternative treatments..."):

# #         try:
# #             prompt = f"""
# #             Patient details:
# #             Bacteria: {bacteria}
# #             Age: {age}
# #             Hospital history: {hospital}

# #             Model Results:
# #             Sensitive drugs: {sensitive_list}
# #             Resistant drugs: {[d for d in drugs if d not in sensitive_list]}

# #             Task:
# #             Suggest 3 alternative antibiotics NOT in tested list.
# #             Give short medical reasoning.
# #             """

# #             response = llm.generate_content(prompt)

# #             st.info(response.text)

# #         except Exception as e:
# #             st.error(f"Gemini error: {e}")

# # else:
# #     st.info("👈 Enter patient details in sidebar and click 'Run Analysis'")