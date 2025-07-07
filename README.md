# Iris-Classification-Task1

Iris Flower Species Classifier:
A simple machine learning project that predicts the species of an Iris flower based on its sepal and petal measurements.
Built using Logistic Regression, Scikit-Learn, and an interactive Streamlit app for live predictions.

Dataset:
The classic Iris dataset (sklearn.datasets.load_iris).
-Features: sepal length & width, petal length & width.
-Target: species → Setosa, Versicolor, Virginica.

How it works:
1️- Train the model
Task_1.py loads the dataset, scales features, trains a logistic regression model, and saves both the model and scaler as iris_model.pkl.

2️- Launch the Streamlit app
app.py loads the saved model & scaler.

User adjusts sliders for measurements.
App predicts the flower species instantly.
