import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Load the trained model
model, scaler = pickle.load(open('iris_model.pkl', 'rb'))

# Load Iris dataset to get target names
iris = load_iris()
target_names = iris.target_names

# Streamlit app title & description
st.title("🌹 Iris Flower Species Predictor")
st.write(
    """
    Adjust the sliders below to set the flower's measurements.
    Click **Predict Species** to see what type of Iris flower it is!
    """
)

# Sliders for user input
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width  = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width  = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict Species"):
    # Format input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_species = target_names[prediction[0]]
    st.success(f"🌼 The predicted species is **{predicted_species.capitalize()}**.")

# cd "D:\Courses\summers_2025" streamlit run app.py