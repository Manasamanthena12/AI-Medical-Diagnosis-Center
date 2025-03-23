import streamlit as st
import pickle
import pandas as pd
import time
import base64

# Set page config for wide layout, title, and favicon
st.set_page_config(
    page_title="AI Medical Diagnosis Center",
    layout="wide",
    page_icon="🩺"  # Stethoscope emoji as favicon
)

# Function to set background image
def set_background(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            .stButton>button {{
                background-color: #4682B4;  /* Steel blue */
                color: white;
                border-radius: 10px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stButton.small>button {{
                font-size: 14px;
                padding: 5px 10px;
            }}
            .stNumberInput>div>input, .stSelectbox>div>select {{
                border-radius: 5px;
                border: 1px solid #ccc;
                font-weight: bold;
                color: black;
            }}
            .title {{
                text-align: center;
                font-size: 50px;
                font-weight: 900;
                font-family: 'Arial Black', Arial, sans-serif;
                color: #000000;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                background-color: rgba(255, 255, 255, 0.7);
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
                margin-bottom: 50px;
            }}
            .disease-card>button {{
                background-color: #4682B4;  /* Steel blue */
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 18px;
                padding: 15px;
                width: 100%;
                transition: background-color 0.2s;
            }}
            .disease-card>button:hover {{
                background-color: #1E90FF;  /* Dodger blue */
            }}
            .disease-card.selected>button {{
                background-color: #1E90FF;  /* Dodger blue */
            }}
            .disease-title {{
                text-align: center;
                font-size: 32px;
                font-weight: bold;
                color: black;
                margin-bottom: 20px;
            }}
            .description, .prediction-result {{
                font-size: 20px;
                font-weight: bold;
                color: black;
                text-align: center;
                margin-top: 20px;
            }}
            .input-label {{
                font-size: 16px;
                font-weight: bold;
                color: black;
                margin-top: 10px;
            }}
            .example-text {{
                font-size: 14px;
                font-weight: bold;
                color: black;
                margin-left: 10px;
            }}
            .content {{
                margin-top: 30px;
            }}
            .error-message {{
                font-size: 16px;
                font-weight: bold;
                color: red;
                text-align: center;
                margin-top: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #f0f2f6;
            }
            .stButton>button {
                background-color: #4682B4;  /* Steel blue */
                color: white;
                border-radius: 10px;
                font-weight: bold;
                margin: 10px 0;
            }
            .stButton.small>button {
                font-size: 14px;
                padding: 5px 10px;
            }
            .stNumberInput>div>input, .stSelectbox>div>select {
                border-radius: 5px;
                border: 1px solid #ccc;
                font-weight: bold;
                color: black;
            }
            .title {
                text-align: center;
                font-size: 50px;
                font-weight: 900;
                font-family: 'Arial Black', Arial, sans-serif;
                color: #000000;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                background-color: rgba(255, 255, 255, 0.7);
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
                margin-bottom: 50px;
            }
            .disease-card>button {
                background-color: #4682B4;  /* Steel blue */
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 18px;
                padding: 15px;
                width: 100%;
                transition: background-color 0.2s;
            }
            .disease-card>button:hover {
                background-color: #1E90FF;  /* Dodger blue */
            }
            .disease-card.selected>button {
                background-color: #1E90FF;  /* Dodger blue */
            }
            .disease-title {
                text-align: center;
                font-size: 32px;
                font-weight: bold;
                color: black;
                margin-bottom: 20px;
            }
            .description, .prediction-result {
                font-size: 20px;
                font-weight: bold;
                color: black;
                text-align: center;
                margin-top: 20px;
            }
            .input-label {
                font-size: 16px;
                font-weight: bold;
                color: black;
                margin-top: 10px;
            }
            .example-text {
                font-size: 14px;
                font-weight: bold;
                color: black;
                margin-left: 10px;
            }
            .content {
                margin-top: 30px;
            }
            .error-message {
                font-size: 16px;
                font-weight: bold;
                color: red;
                text-align: center;
                margin-top: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.warning("Background image not found. Using default background color.")

# Configuration for diseases, models, features, descriptions, and inputs
CONFIG = {
    "Diabetes": {
        "model": "models/diabetes_model.sav",
        "features": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        "description": "Assesses diabetes risk using metrics like glucose levels, BMI, and number of pregnancies.",
        "inputs": [
            {"label": "Number of Pregnancies", "type": "number", "min_value": 0, "step": 1, "example": "3"},
            {"label": "Glucose Level", "type": "number", "min_value": 0, "step": 1, "example": "120"},
            {"label": "Blood Pressure value", "type": "number", "min_value": 0, "step": 1, "example": "80"},
            {"label": "Skin Thickness value", "type": "number", "min_value": 0, "step": 1, "example": "20"},
            {"label": "Insulin Level", "type": "number", "min_value": 0, "step": 1, "example": "80"},
            {"label": "BMI value", "type": "number", "min_value": 0.0, "step": 0.1, "example": "32.0"},
            {"label": "Diabetes Pedigree Function value", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.5"},
            {"label": "Age of the Person", "type": "number", "min_value": 0, "step": 1, "example": "45"}
        ]
    },
    "Heart Disease": {
        "model": "models/heart_disease_model.sav",
        "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        "description": "Predicts the likelihood of heart disease based on factors like age, cholesterol, and chest pain type.",
        "inputs": [
            {"label": "Age", "type": "number", "min_value": 0, "step": 1, "example": "55"},
            {"label": "Sex (1 = Male, 0 = Female)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Chest Pain types (0-3)", "type": "number", "min_value": 0, "max_value": 3, "step": 1, "example": "2"},
            {"label": "Resting Blood Pressure", "type": "number", "min_value": 0, "step": 1, "example": "130"},
            {"label": "Serum Cholesterol in mg/dl", "type": "number", "min_value": 0, "step": 1, "example": "240"},
            {"label": "Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Resting Electrocardiographic results (0-2)", "type": "number", "min_value": 0, "max_value": 2, "step": 1, "example": "1"},
            {"label": "Maximum Heart Rate achieved", "type": "number", "min_value": 0, "step": 1, "example": "150"},
            {"label": "Exercise Induced Angina (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "ST depression induced by exercise", "type": "number", "min_value": 0.0, "step": 0.1, "example": "1.0"},
            {"label": "Slope of the peak exercise ST segment (0-2)", "type": "number", "min_value": 0, "max_value": 2, "step": 1, "example": "1"},
            {"label": "Major vessels colored by fluoroscopy (0-3)", "type": "number", "min_value": 0, "max_value": 3, "step": 1, "example": "0"},
            {"label": "Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", "type": "number", "min_value": 0, "max_value": 2, "step": 1, "example": "2"}
        ]
    },
    "Parkinsons Disease": {
        "model": "models/parkinsons_model.sav",
        "features": ['MDVP:Fo', 'MDVP:Fhi', 'MDVP:Flo', 'MDVP:Jitter', 'MDVP:Jitter.1', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer.1', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
        "description": "Identifies Parkinson’s disease using voice measurements like jitter and shimmer.",
        "inputs": [
            {"label": "MDVP:Fo", "type": "number", "min_value": 0.0, "step": 0.1, "example": "119.992"},
            {"label": "MDVP:Fhi", "type": "number", "min_value": 0.0, "step": 0.1, "example": "157.302"},
            {"label": "MDVP:Flo", "type": "number", "min_value": 0.0, "step": 0.1, "example": "74.997"},
            {"label": "MDVP:Jitter", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.00784"},
            {"label": "MDVP:Jitter(Abs)", "type": "number", "min_value": 0.0, "step": 0.00001, "example": "0.00007"},
            {"label": "MDVP:RAP", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.00370"},
            {"label": "MDVP:PPQ", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.00554"},
            {"label": "Jitter:DDP", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.01109"},
            {"label": "MDVP:Shimmer", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.04374"},
            {"label": "MDVP:Shimmer(dB)", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.426"},
            {"label": "Shimmer:APQ3", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.02182"},
            {"label": "Shimmer:APQ5", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.03130"},
            {"label": "MDVP:APQ", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.02971"},
            {"label": "Shimmer:DDA", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.06545"},
            {"label": "NHR", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.02211"},
            {"label": "HNR", "type": "number", "min_value": 0.0, "step": 0.01, "example": "21.033"},
            {"label": "RPDE", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.414783"},
            {"label": "DFA", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.815285"},
            {"label": "spread1", "type": "number", "min_value": -10.0, "step": 0.01, "example": "-4.813031"},
            {"label": "spread2", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.266482"},
            {"label": "D2", "type": "number", "min_value": 0.0, "step": 0.01, "example": "2.301442"},
            {"label": "PPE", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.284654"}
        ]
    },
    "Lung Cancer": {
        "model": "models/lung_cancer_model.sav",
        "features": ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'],
        "description": "Evaluates lung cancer risk based on symptoms like smoking, coughing, and chest pain.",
        "inputs": [
            {"label": "Gender (1 = Male, 0 = Female)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Age", "type": "number", "min_value": 0, "step": 1, "example": "62"},
            {"label": "Smoking (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Yellow Fingers (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Anxiety (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Peer Pressure (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Chronic Disease (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Fatigue (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Allergy (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Wheezing (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Alcohol Consuming (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Coughing (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Shortness of Breath (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "1"},
            {"label": "Swallowing Difficulty (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "0"},
            {"label": "Chest Pain (1 = Yes, 0 = No)", "type": "selectbox", "options": [1, 0], "example": "1"}
        ]
    },
    "Breast Cancer": {
        "model": "models/breast_cancer_model.sav",
        "features": [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
            'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ],
        "description": "Detects breast cancer using tumor characteristics like radius, texture, and smoothness.",
        "inputs": [
            {"label": "Mean Radius", "type": "number", "min_value": 0.0, "step": 0.1, "example": "15.0"},
            {"label": "Mean Texture", "type": "number", "min_value": 0.0, "step": 0.1, "example": "20.0"},
            {"label": "Mean Perimeter", "type": "number", "min_value": 0.0, "step": 0.1, "example": "90.0"},
            {"label": "Mean Area", "type": "number", "min_value": 0.0, "step": 1.0, "example": "600.0"},
            {"label": "Mean Smoothness", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.1"},
            {"label": "Mean Compactness", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.1"},
            {"label": "Mean Concavity", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.1"},
            {"label": "Mean Concave Points", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.05"},
            {"label": "Mean Symmetry", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.18"},
            {"label": "Mean Fractal Dimension", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.06"},
            {"label": "Radius Error", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.4"},
            {"label": "Texture Error", "type": "number", "min_value": 0.0, "step": 0.01, "example": "1.2"},
            {"label": "Perimeter Error", "type": "number", "min_value": 0.0, "step": 0.01, "example": "2.8"},
            {"label": "Area Error", "type": "number", "min_value": 0.0, "step": 0.1, "example": "40.0"},
            {"label": "Smoothness Error", "type": "number", "min_value": 0.0, "step": 0.0001, "example": "0.007"},
            {"label": "Compactness Error", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.02"},
            {"label": "Concavity Error", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.03"},
            {"label": "Concave Points Error", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.01"},
            {"label": "Symmetry Error", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.02"},
            {"label": "Fractal Dimension Error", "type": "number", "min_value": 0.0, "step": 0.0001, "example": "0.003"},
            {"label": "Worst Radius", "type": "number", "min_value": 0.0, "step": 0.1, "example": "17.0"},
            {"label": "Worst Texture", "type": "number", "min_value": 0.0, "step": 0.1, "example": "25.0"},
            {"label": "Worst Perimeter", "type": "number", "min_value": 0.0, "step": 0.1, "example": "110.0"},
            {"label": "Worst Area", "type": "number", "min_value": 0.0, "step": 1.0, "example": "800.0"},
            {"label": "Worst Smoothness", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.12"},
            {"label": "Worst Compactness", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.25"},
            {"label": "Worst Concavity", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.3"},
            {"label": "Worst Concave Points", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.15"},
            {"label": "Worst Symmetry", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.3"},
            {"label": "Worst Fractal Dimension", "type": "number", "min_value": 0.0, "step": 0.001, "example": "0.08"}
        ]
    },
    "Brain Disease": {
        "model": "models/alzheimers_model.sav",
        "features": ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'],
        "description": "Predicts brain disease risk using brain metrics like MMSE score and brain volume.",
        "inputs": [
            {"label": "Age", "type": "number", "min_value": 0, "step": 1, "example": "85"},
            {"label": "Years of Education", "type": "number", "min_value": 0, "step": 1, "example": "12"},
            {"label": "Socioeconomic Status (SES, 1-5)", "type": "number", "min_value": 1, "max_value": 5, "step": 1, "example": "3", "default_value": 1},
            {"label": "MMSE Score (0-30)", "type": "number", "min_value": 0, "max_value": 30, "step": 1, "example": "18"},
            {"label": "eTIV", "type": "number", "min_value": 0.0, "step": 1.0, "example": "1500.0"},
            {"label": "nWBV", "type": "number", "min_value": 0.0, "step": 0.01, "example": "0.65"},
            {"label": "ASF", "type": "number", "min_value": 0.0, "step": 0.01, "example": "1.2"}
        ]
    },
    "Kidney Disease": {
        "model": "models/kidney_model.sav",
        "features": ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'],
        "description": "Detects chronic kidney disease using blood metrics like serum creatinine and hemoglobin.",
        "inputs": [
            {"label": "Age", "type": "number", "min_value": 0, "step": 1, "example": "60"},
            {"label": "Blood Pressure (mmHg)", "type": "number", "min_value": 0, "step": 1, "example": "90"},
            {"label": "Specific Gravity (sg)", "type": "number", "min_value": 1.0, "step": 0.005, "example": "1.020", "default_value": 1.0},
            {"label": "Albumin (al, 0-5)", "type": "number", "min_value": 0, "step": 1, "example": "1"},
            {"label": "Sugar (su, 0-5)", "type": "number", "min_value": 0, "step": 1, "example": "0"},
            {"label": "Blood Glucose Random (bgr, mg/dL)", "type": "number", "min_value": 0, "step": 1, "example": "150"},
            {"label": "Blood Urea (bu, mg/dL)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "50.0"},
            {"label": "Serum Creatinine (sc, mg/dL)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "3.0"},
            {"label": "Sodium (sod, mEq/L)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "135.0"},
            {"label": "Potassium (pot, mEq/L)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "4.0"},
            {"label": "Hemoglobin (hemo, g/dL)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "10.0"},
            {"label": "Packed Cell Volume (pcv)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "40.0"},
            {"label": "White Blood Cell Count (wbcc, cells/cumm)", "type": "number", "min_value": 0, "step": 100, "example": "8000"},
            {"label": "Red Blood Cell Count (rbcc, millions/cumm)", "type": "number", "min_value": 0.0, "step": 0.1, "example": "4.5"}
        ]
    }
}

# Load models
try:
    models = {disease: pickle.load(open(config["model"], 'rb')) for disease, config in CONFIG.items()}
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Navigation between pages
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Page 1: Home Page
if st.session_state.page == "home":
    set_background('background.jpg')
    st.markdown('<div class="title">AI MEDICAL DIAGNOSIS CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("### Select a Disease to Predict")
    cols = st.columns(4)
    for idx, disease_name in enumerate(CONFIG.keys()):
        col = cols[idx % 4]
        with col:
            card_class = "disease-card selected" if st.session_state.selected_disease == disease_name else "disease-card"
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            if st.button(disease_name, key=f"card_{disease_name}", use_container_width=True):
                st.session_state.selected_disease = disease_name
                st.session_state.page = "details"
                st.session_state.prediction_result = None  # Reset prediction result
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Disease Details Page
elif st.session_state.page == "details" and st.session_state.selected_disease:
    set_background('background_details.jpg')
    st.markdown('<div class="content">', unsafe_allow_html=True)
    disease = st.session_state.selected_disease
    st.markdown(f'<div class="disease-title">{disease} Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="description">{CONFIG[disease]["description"]}</div>', unsafe_allow_html=True)

    # Dynamically generate input fields
    inputs = {}
    col1, col2 = st.columns(2)
    for idx, input_config in enumerate(CONFIG[disease]["inputs"]):
        col = col1 if idx % 2 == 0 else col2
        with col:
            st.markdown(
                f'<div class="input-label">{input_config["label"]} <span class="example-text">(e.g., {input_config["example"]})</span></div>',
                unsafe_allow_html=True
            )
            key = f"{disease.lower().replace(' ', '_')}_{input_config['label'].lower().replace(' ', '_')}"
            if input_config["type"] == "number":
                # Determine the type based on step
                step = input_config.get("step", 1)
                is_float = isinstance(step, float)
                default_value = input_config.get("default_value", 0.0 if is_float else 0)
                value = st.number_input(
                    "",
                    min_value=input_config.get("min_value", 0.0 if is_float else 0),
                    max_value=input_config.get("max_value"),
                    step=step,
                    value=default_value,
                    label_visibility="collapsed",
                    key=key
                )
            elif input_config["type"] == "selectbox":
                value = st.selectbox(
                    "",
                    input_config["options"],
                    label_visibility="collapsed",
                    key=key
                )
            inputs[input_config["label"]] = value

    # Button for prediction
    if st.button(f"Predict {disease}", use_container_width=True):
        # Validate inputs
        all_valid = True
        for label, value in inputs.items():
            # Allow 0 for selectbox inputs (binary), but not for number inputs
            input_type = next((input_config["type"] for input_config in CONFIG[disease]["inputs"] if input_config["label"] == label), "number")
            if input_type == "number" and (value == 0 or value == 0.0):
                all_valid = False
                break
        
        if not all_valid:
            st.markdown('<div class="error-message">Please fill in all numerical fields (except binary inputs) with valid non-zero values.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Predicting..."):
                time.sleep(1)  # Simulate a delay for the spinner to be visible
                input_data = [inputs[label] for label in [input_config["label"] for input_config in CONFIG[disease]["inputs"]]]
                input_df = pd.DataFrame([input_data], columns=CONFIG[disease]["features"])
                try:
                    prediction = models[disease].predict(input_df)[0]
                    result = f"{disease} Detected" if prediction == 1 else f"No {disease}"
                    st.session_state.prediction_result = result
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    # Display prediction result
    if st.session_state.prediction_result:
        st.markdown(f'<div class="prediction-result">🎉 Prediction: {st.session_state.prediction_result} 🎉</div>', unsafe_allow_html=True)

    # "Back to Home" button at the end
    st.markdown('<div class="stButton small">', unsafe_allow_html=True)
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.session_state.selected_disease = None
        st.session_state.prediction_result = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)