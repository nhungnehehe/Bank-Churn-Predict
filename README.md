
# ChurnCast – Customer Churn Prediction for Banks

This project is a machine learning application that predicts customer churn for banks, using multiple classification models. The system includes a Python-based GUI developed with PyQt6 and supports model training, evaluation, and prediction from both individual and bulk input.

---

## 📌 Project Overview

**Goal:** Build a software tool that integrates ML models to predict whether a bank customer is likely to leave, helping banks improve customer retention strategies.

**Models Implemented:**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (tuned with RandomizedSearchCV)

**Best model selected:** `XGBoost` achieved the highest accuracy (90.4%) and F1-score (0.901).

---

## 🧠 Features

- Load data and preprocess (One-hot encoding, StandardScaler, SMOTE)
- Train and test multiple ML models
- Evaluate models with Accuracy, Precision, Recall, F1-score
- Visualize confusion matrix and churn distributions
- Predict churn for new customer entries or CSV batch files
- GUI Interface with login/reset password functions

---

## 🖥 Interface Preview

| Login Screen | Prediction Screen | Visualization |
|--------------|-------------------|----------------|
| ![login](screenshots/login.png) | ![predict](screenshots/predict.png) | ![charts](screenshots/chart.png) |

> *(Add your screenshots in a `screenshots/` folder for these images to show)*

---

## 🛠 Tools & Technologies

- Python (Pandas, Scikit-learn, XGBoost, SMOTE)
- PyQt6 (GUI)
- MySQL (Database)
- GitHub, VS Code
- `joblib` for saving models

---

## 📂 Folder Structure


