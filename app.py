import streamlit as st
import pickle
import numpy as np

# Load the trained Naive Bayes model
with open("final_model.sav", "rb") as f:
    gnb_model = pickle.load(f)

# Apply custom CSS styling
with open("styles.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Feature names for input
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

# Feature descriptions for tooltips
feature_info = {
    'age': 'Age of the patient in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting ECG results (0–2)',
    'thalach': 'Max heart rate achieved',
    'exang': 'Exercise-induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression by exercise',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Number of major vessels (0–3)',
    'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
}

# Layout
col1, col2 = st.columns([1, 2])
with col1:
    st.image("heart.png", use_column_width=True)
with col2:
    st.title('Heart Disease Detection')
    st.write('This app predicts the likelihood of heart disease based on user inputs.')

# Input section
col3, col4, col5 = st.columns(3)
inputs = []
for i, feature_name in enumerate(feature_names):
    col = [col3, col4, col5][i % 3]
    with col:
        val = st.number_input(
            f'{feature_name.capitalize()}',
            min_value=0.0,
            step=1.0,
            help=feature_info.get(feature_name, "")
        )
        inputs.append(val)

# Prediction
def predict(features):
    features_array = np.array(features).reshape(1, -1)
    prob = gnb_model.predict_proba(features_array)[0][1]
    return prob

if st.button('Predict'):
    prediction = predict(inputs)
    if prediction > 0.5:
        st.error(f'⚠️ High probability of heart disease! Consult a doctor. Probability: {prediction:.2f}')
    else:
        st.success(f'✅ No heart disease detected. Probability: {prediction:.2f}')
