# Predict Students' Dropout and Academic Success: ML Classification and Streamlit App

**M.Tech (AIML) – Machine Learning (S1-26_AIMLZG565) – Assignment 2**

- **GitHub repository:** `https://github.com/ervijayt/bits-sem-1-ml-assignment-2-student-academic-success-prediction`
- **Live Streamlit app:** `<ADD YOUR STREAMLIT APP LINK HERE>`

---

## a. Problem Statement

Higher-education institutions lose many students to dropout or delayed graduation. The goal is to **predict each student's academic outcome**
(**Dropout**, **Enrolled** or **Graduate**) at the end of the normal course duration, using information known at enrolment
(demographics, socio-economic background and prior academic path) and performance in the first two semesters.
Early identification of at-risk students lets the institution intervene with tutoring, counselling or financial support.

This is a **multi-class (3-class) classification** problem. Five classical ML classifiers are trained on the same data and compared on six
evaluation metrics. The trained models are served through an interactive Streamlit web app.

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, ID 697 – <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success> |
| Origin | Polytechnic Institute of Portalegre (Portugal), undergraduate students enrolled 2008/09 – 2018/19 |
| Instances | **4,424** (no missing values) |
| Features | **36** input features + 1 target (`Target`) |
| Task | Multi-class classification (3 classes) |
| Class distribution | Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%), so the classes are **imbalanced** |
| File format | `;`-separated CSV (`data/students_dropout.csv`) |

**Feature groups**

| Group | Features |
|---|---|
| Demographic | Marital status, Nacionality, Displaced, Gender, Age at enrollment, International, Educational special needs |
| Socio-economic | Mother's/Father's qualification, Mother's/Father's occupation, Debtor, Tuition fees up to date, Scholarship holder |
| Academic path at enrolment | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification (+ grade), Admission grade |
| Academic performance | Curricular units 1st & 2nd semester: credited, enrolled, evaluations, approved, grade, without evaluations (12 features) |
| Macro-economic | Unemployment rate, Inflation rate, GDP |

Categorical attributes are already **integer-encoded** in the source file, so the only preprocessing needed is feature **standardisation**
(`StandardScaler`). It is fitted on the training split only, inside each model's `Pipeline`, so there is no data leakage.

**Train / test split:** 80% / 20%, stratified on `Target`, `random_state=42`, giving 3,539 training and 885 test rows.
The 885-row test split is saved as [`test_data.csv`](test_data.csv). It includes the `Target` column so the app can compute metrics.

## c. GitHub Repository Link

`<ADD YOUR GITHUB REPO LINK HERE>`

Repository structure:

```
ml-assignment-2/
│-- app.py                 # Streamlit web application
│-- train_models.py        # Trains, evaluates and saves all 5 models
│-- requirements.txt
│-- README.md
│-- test_data.csv          # Held-out 20% test split (features + Target)
│-- data/
│   └── students_dropout.csv   # Original UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.csv            # Comparison table (generated)
```

## d. Models Used

All five models are `sklearn` pipelines of the form `StandardScaler → classifier`, trained on the same split:

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000)` (multinomial) |
| Decision Tree | `DecisionTreeClassifier(max_depth=8, min_samples_leaf=10)` |
| K-Nearest Neighbor | `KNeighborsClassifier(n_neighbors=15, weights="distance")` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | `SVC(kernel="rbf", C=1.0, probability=True)` |

**Metric definitions (multi-class):** Accuracy; **AUC** = one-vs-rest ROC-AUC, weighted by class support;
**Precision, Recall, F1** = weighted averages over the three classes; **MCC** = multi-class Matthews Correlation Coefficient.

### Comparison Table (held-out test set, n = 885)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7684** | **0.8961** | **0.7500** | **0.7684** | **0.7531** | **0.6150** |
| Decision Tree | 0.7198 | 0.8665 | 0.7153 | 0.7198 | 0.7167 | 0.5403 |
| KNN | 0.7017 | 0.8392 | 0.6852 | 0.7017 | 0.6721 | 0.5009 |
| Naive Bayes | 0.6588 | 0.8118 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Support Vector Machine | 0.7582 | 0.8865 | 0.7465 | 0.7582 | 0.7454 | 0.5986 |

Per-class recall on the test set (Dropout / Enrolled / Graduate): LR 0.77 / 0.33 / 0.93; DT 0.68 / 0.41 / 0.86;
KNN 0.62 / 0.20 / 0.94; NB 0.67 / 0.18 / 0.83; SVM 0.73 / 0.35 / 0.93.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on **all six metrics** (Acc 0.768, AUC 0.896, MCC 0.615). Train and test accuracy are almost the same (0.773 vs 0.768), so it generalises well with no overfitting. After scaling, the strongest predictors (approved units and grades in semesters 1–2, tuition fees up to date, scholarship) relate to the outcome almost linearly, which suits a linear model. It identifies Dropouts well (recall 0.77) but recalls only 33% of the minority *Enrolled* class. |
| Decision Tree | Mid-table (Acc 0.720, MCC 0.540). Limiting the depth to 8 with at least 10 samples per leaf controls overfitting, but a train/test gap remains (0.808 vs 0.720). Its axis-aligned splits give the **best recall on the hard *Enrolled* class (0.41)** of all five models, at the cost of lower Graduate and Dropout recall. It is the most interpretable model, since its rules can be read directly. |
| KNN | Acc 0.702 and MCC 0.501. Distance weighting means it memorises the training set (train accuracy 1.00), which signals high variance. With 36 dimensions the distances become less informative, and the many integer-coded categorical features (Course, Occupation, and so on) are not truly metric. It favours the majority class: Graduate recall is very high (0.94) but Enrolled recall is only 0.20, which gives the lowest F1 apart from NB. |
| Naive Bayes | Weakest model (Acc 0.659, MCC 0.428), though it still has a reasonable AUC (0.812). GaussianNB assumes features are independent and normally distributed. Both assumptions fail here: the semester features are strongly correlated (enrolled, approved and grade move together), and many features are coded categories. It is fast and stable (train 0.675 vs test 0.659) but has high bias, and Enrolled recall is only 0.18. |
| Support Vector Machine | A close second (Acc 0.758, AUC 0.887, MCC 0.599), within about 1 percentage point of Logistic Regression on every metric. The RBF kernel captures some non-linearity but adds little over the linear boundary, and it overfits slightly (train 0.830 vs test 0.758). It is slower to train and predict, and probability estimates need internal Platt scaling. |
| **Overall winner** | **Logistic Regression.** It has the highest Accuracy, AUC, F1 and MCC, the smallest train/test gap, and it is simple, fast and interpretable. SVM is a close runner-up. All models struggle with the *Enrolled* class: it is the minority class (18%) and lies "between" Dropout and Graduate. Class weighting, resampling (SMOTE) or ensemble methods would be the next steps to improve it. |

---

## How to Run

```bash
cd ml-assignment-2            # scripts use paths relative to this folder
pip install -r requirements.txt
python train_models.py        # retrains models, regenerates model/*.joblib, model/metrics.csv, test_data.csv
python -m streamlit run app.py
```

### Streamlit App Features
- **Dataset upload (CSV):** upload test data (`,` or `;` separated). If nothing is uploaded, the app uses `test_data.csv`, which can also be downloaded from the sidebar.
- **Model selection (radio buttons):** choose any of the 5 trained models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the uploaded data (when it has a `Target` column), plus a comparison table of all models.
- **Confusion matrix and classification report** for the selected model, plus a table of actual vs predicted classes.

### Deployment (Streamlit Community Cloud)
1. Push this folder to a GitHub repository.
2. Go to <https://streamlit.io/cloud>, sign in with GitHub and click **New app**.
3. Select the repository and branch, set **Main file path** to `app.py`, then click **Deploy**.

> The `.joblib` models are tied to the `scikit-learn` version used to train them (pinned in `requirements.txt`).
> If you retrain with a different version (for example on BITS Virtual Lab), update the pin to match `python -c "import sklearn; print(sklearn.__version__)"`.
