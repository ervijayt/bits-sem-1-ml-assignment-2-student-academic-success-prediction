# Train 5 classification models on the Student Dropout dataset and save them.
# Run:  python train_models.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# 1. Load the data (the UCI file is separated by ';')
df = pd.read_csv("data/students_dropout.csv", sep=";")
df.columns = df.columns.str.strip()  # one column name has an extra tab at the end

X = df.drop(columns=["Target"])
y = df["Target"]
print("Dataset shape:", df.shape)
print(y.value_counts())

# 2. Split into 80% train and 20% test, and save the test part for the app
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

test_data = X_test.copy()
test_data["Target"] = y_test
test_data.to_csv("test_data.csv", index=False)

# 3. The 5 models. Each one scales the features first (StandardScaler).
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

file_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# 4. Train, evaluate and save each model
results = []
for name, classifier in models.items():
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 3 classes -> "weighted" average for precision/recall/F1, one-vs-rest for AUC
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    joblib.dump(model, "model/" + file_names[name])
    print("Saved:", name)

# 5. Save and print the comparison table
results = pd.DataFrame(results).round(4)
results.to_csv("model/metrics.csv", index=False)
print(results.to_string(index=False))
