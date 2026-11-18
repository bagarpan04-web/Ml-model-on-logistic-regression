"""
THE HONEST VERSION - train/test split, baseline, and predict_proba.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

students = pd.read_csv("students.csv")

X = students[["hours_studied", "attendance", "prev_marks"]]
y = students["passed"]

# Baseline: the score you get for free by saying "fail" to everyone
print("Share who passed :", students["passed"].mean())
print("Baseline accuracy:", round((1 - students["passed"].mean()) * 100, 1), "%")
print()

# Hide 25% of the students from the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

model = LogisticRegression().fit(X_train, y_train)

correct = (model.predict(X_test) == y_test).sum()
print("Trained on", len(X_train), "| tested on", len(X_test))
print("Test accuracy    :", round(correct / len(y_test) * 100, 1), "%")   # len(y_test), not len(students)
print()

# The model's formula
print("Coefficients:")
for name, weight in zip(X.columns, model.coef_[0]):
    print(f"  {name:<15} {weight:+.3f}")
print("  intercept       ", round(model.intercept_[0], 3))
print()

# Ask it about one person (with confidence)
one = pd.DataFrame([[8.0, 90, 70]], columns=X.columns)
print("Prediction for 8 hrs, 90 att, 70 marks:", model.predict(one)[0])
print("P(fail), P(pass):", model.predict_proba(one).round(3)[0])

edge = pd.DataFrame([[5.0, 75, 60]], columns=X.columns)
print("Borderline student P(fail), P(pass):", model.predict_proba(edge).round(3)[0])
