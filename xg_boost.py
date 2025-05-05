# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# Importing the dataset
dataset = pd.read_csv(r"your_path")
X = dataset.iloc[:, 3:-1].values  # Features (columns 3 to second-last)
y = dataset.iloc[:, -1].values    # Target (last column: "Exited")

# Encoding categorical data
# Label Encoding the "Gender" column (column index 2)
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Male: 1, Female: 0

# One-Hot Encoding the "Geography" column (column index 1)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)
X = ct.fit_transform(X)  # This already returns a NumPy array

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training XGBoost
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Confusion Matrix & Accuracy
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Training set accuracy (bias)
bias = classifier.score(X_train, y_train)
print("Training Accuracy (Bias):", bias)

# k-Fold Cross Validation (Stratified)
accuracies = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(accuracies.mean() * 100))
