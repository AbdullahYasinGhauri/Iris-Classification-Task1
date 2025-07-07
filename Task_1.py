# Iris Classification based on measurements

# Importing libraries
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loading Iris Dataset
iris = load_iris()
x = iris.data
y = iris.target

# Splitting the dataset into the training set and the test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

# Scaling our features
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Modelling our Regression (logistic)
model = LogisticRegression()
model.fit(x_train, y_train)

# Making Predictions
y_pred = model.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))

with open('iris_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

print("Model & Scaler saved as iris_model.pkl")