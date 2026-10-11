
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# Load trained model and scaler
# ----------------------------------
model = joblib.load("best_cvd_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="CardioCare Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------------
# Custom styling
# ----------------------------------

# ----------------------------------
# Custom styling
# ----------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
    color: #243447;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #17365d !important;
    font-weight: 800;
}

h2, h3 {
    color: #24476b !important;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #243447;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e1e8f0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #17365d !important;
}

.stButton > button {
    background-color: #d62839;
    color: white !important;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #a91d2b;
    color: white !important;
}


/* Sidebar readability */
section[data-testid="stSidebar"] {
    background-color: #252833;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #f4f7fb !important;
}

/* Keep input fields readable */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    color: #ffffff !important;
}


/* Hide Streamlit's top toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Remove the empty top header space */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hide the Deploy button and menu */
.stDeployButton {
    display: none !important;
}


</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Sidebar: Patient inputs
# ----------------------------------
st.sidebar.title("👤 Patient Details")
st.sidebar.caption("Enter the available dataset values.")

age = st.sidebar.number_input(
    "Age", min_value=1, max_value=120, value=50
)

gender = st.sidebar.selectbox(
    "Gender (dataset code)", [0, 1]
)

chestpain = st.sidebar.number_input(
    "Chest Pain (code)", min_value=0, max_value=3, value=1
)

restingBP = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=0, max_value=250, value=120
)

serumcholestrol = st.sidebar.number_input(
    "Serum Cholesterol",
    min_value=0, max_value=600, value=200
)

fastingbloodsugar = st.sidebar.selectbox(
    "Fasting Blood Sugar (code)", [0, 1]
)

restingrelectro = st.sidebar.number_input(
    "Resting ECG (code)", min_value=0, max_value=2, value=0
)

maxheartrate = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=50, max_value=250, value=150
)

exerciseangia = st.sidebar.selectbox(
    "Exercise Angina (code)", [0, 1]
)

oldpeak = st.sidebar.number_input(
    "Oldpeak", min_value=0.0, max_value=10.0, value=1.0
)

slope = st.sidebar.number_input(
    "Slope (code)", min_value=0, max_value=2, value=1
)

noofmajorvessels = st.sidebar.number_input(
    "Number of Major Vessels",
    min_value=0, max_value=3, value=0
)

# ----------------------------------
# Patient overview
# ----------------------------------
st.subheader(" Patient Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Age", f"{age} years")
col2.metric("Blood Pressure", restingBP)
col3.metric("Cholesterol", serumcholestrol)
col4.metric("Maximum Heart Rate", maxheartrate)

st.divider()

# ----------------------------------
# Prepare model input
# Keep the same feature order used
# during model training
# ----------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "chestpain": [chestpain],
    "restingBP": [restingBP],
    "serumcholestrol": [serumcholestrol],
    "fastingbloodsugar": [fastingbloodsugar],
    "restingrelectro": [restingrelectro],
    "maxheartrate": [maxheartrate],
    "exerciseangia": [exerciseangia],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "noofmajorvessels": [noofmajorvessels]
})

# ----------------------------------
# Prediction section
# ----------------------------------
st.subheader(" CVD Prediction")

if st.button("❤️ Predict CVD", use_container_width=True):

    try:
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]

        # Assumes model class 1 represents CVD.
        # Locate the probability using the trained class labels.
        classes = list(model.classes_)
        cvd_index = classes.index(1)
        no_cvd_index = classes.index(0)

        cvd_probability = probabilities[cvd_index] * 100
        no_cvd_probability = probabilities[no_cvd_index] * 100

        left, right = st.columns(2)

        with left:
            st.markdown("### Model Result")

            if prediction == 1:
                st.error("⚠️ CVD predicted by the model")
            else:
                st.success("✅ No CVD predicted by the model")

            st.metric(
                "Model-estimated CVD probability",
                f"{cvd_probability:.2f}%"
            )

        with right:
            st.markdown("### Probability Distribution")

            chart_data = pd.DataFrame({
                "Outcome": ["No CVD", "CVD"],
                "Probability (%)": [
                    no_cvd_probability,
                    cvd_probability
                ]
            })

            st.bar_chart(
                chart_data.set_index("Outcome")
            )

    except Exception as error:
        st.error(f"Prediction error: {error}")
        st.info(
            "Check that the model, scaler, and input features "
            "match the training notebook."
        )

st.divider()

# ----------------------------------
# Model performance
# ----------------------------------
st.subheader(" Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.markdown("**Algorithm**")
m1.markdown("### Random Forest")
m2.metric("Accuracy", "98.50%")
m3.metric("F1 Score", "98.71%")
m4.metric("ROC-AUC", "99.90%")

st.caption(
    "Performance figures reported by the training notebook; "
    "they are not evidence of clinical effectiveness."
)

# ----------------------------------
# Medical disclaimer
# ----------------------------------
st.warning(
    "Educational demonstration only. This model is not a "
    "medical diagnostic tool. Its predictions must not be used "
    "to make medical decisions or replace advice from a qualified "
    "healthcare professional."
)

st.markdown("---")
st.caption("❤️ CardioCare | Machine Learning Project")
