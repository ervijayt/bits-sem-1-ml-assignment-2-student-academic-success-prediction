# Streamlit app to test the 5 trained models.
# Run:  streamlit run app.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.title("Predict Students' Dropout and Academic Success")
st.write("Classes: **Dropout**, **Enrolled**, **Graduate** (UCI dataset 697)")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# ---------- Sidebar: upload data and choose a model ----------
st.sidebar.header("1. Upload test data")
with open("test_data.csv", "rb") as f:
    st.sidebar.download_button("Download sample test_data.csv", f, file_name="test_data.csv")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.radio("Model", list(model_files.keys()))

# ---------- Comparison table from training ----------
st.subheader("Comparison of all 5 models (from training)")
st.dataframe(pd.read_csv("model/metrics.csv"), hide_index=True)

# ---------- Load the data ----------
if uploaded_file is not None:
    # sep=None detects ',' or ';' automatically; utf-8-sig removes a hidden BOM character
    data = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
    st.write("Using uploaded file:", uploaded_file.name)
else:
    data = pd.read_csv("test_data.csv")
    st.write("No file uploaded, using the sample **test_data.csv**")

data.columns = data.columns.str.strip()
st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
st.dataframe(data.head())

# ---------- Predict ----------
model = joblib.load("model/" + model_files[model_name])
X = data[model.feature_names_in_]  # same columns, same order as in training
y_pred = model.predict(X)

st.header(f"Results: {model_name}")

if "Target" not in data.columns:
    st.warning("No 'Target' column in the file, so only predictions are shown.")
    st.dataframe(pd.DataFrame({"Predicted": y_pred}))
    st.stop()

y_true = data["Target"]
y_prob = model.predict_proba(X)

# ---------- Evaluation metrics ----------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
col2.metric("AUC", f"{roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'):.4f}")
col3.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
col4.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
col5.metric("F1", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

# ---------- Confusion matrix and classification report ----------
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))

# ---------- Predictions ----------
st.subheader("Predictions")
st.dataframe(pd.DataFrame({"Actual": y_true, "Predicted": y_pred}))
