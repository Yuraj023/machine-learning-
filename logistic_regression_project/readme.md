#  Customer Churn Prediction using Logistic Regression

A machine learning project that predicts whether a telecom customer is likely to churn (leave the service) using **Logistic Regression**. The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and deployment with **Streamlit**.

---

##  Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Acquiring a new customer is often more expensive than retaining an existing one. This project helps identify customers who are at a higher risk of leaving by analyzing customer demographics, subscribed services, contract details, and billing information.

The trained model predicts the probability of churn and provides an easy-to-use web interface for making real-time predictions.

---

##  Objectives

- Understand the Customer Churn dataset
- Perform data cleaning and preprocessing
- Explore the dataset using Exploratory Data Analysis (EDA)
- Build a Logistic Regression classification model
- Evaluate model performance using multiple metrics
- Deploy the trained model using Streamlit
- Create a professional portfolio-ready machine learning project

---

##  Project Structure

```text
Customer_Churn_Prediction/
│
├── dataset/
│   ├── Telco_customer_churn.xlsx
│   └── cleaned_customer_churn.csv
│
├── models/
│   ├── customer_churn_pipeline.joblib
│   └── label_encoder.joblib
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning_Preprocessing.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   └── 04_Feature_Engineering_Model_Training.ipynb
│
├── outputs/
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

##  Dataset Information

| Feature | Details |
|----------|---------|
| Dataset | IBM Telco Customer Churn |
| Problem Type | Binary Classification |
| Target Variable | Churn Label |
| Records | 7,043 Customers |
| Features | Customer Demographics, Services, Contracts & Billing |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Plotly
- Streamlit

---

## ⚙ Machine Learning Workflow

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
One-Hot Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train-Test Split
      │
      ▼
Logistic Regression
      │
      ▼
Model Evaluation
      │
      ▼
Model Deployment (Streamlit)
```

---

##  Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **80.20%** |
| ROC-AUC Score | **0.8489** |
| Decision Threshold | **0.35** |
| Classification Type | Binary Classification |

### Model Summary

- ✅ Data Cleaning Completed
- ✅ Target Leakage Removed
- ✅ Logistic Regression Pipeline Implemented
- ✅ Model Saved using Joblib
- ✅ Streamlit Application Developed
- ✅ Deployment Ready

---

##  Streamlit Application Features

- Predict customer churn in real time
- Modern and responsive user interface
- Churn probability visualization
- Risk level indicator
- Business recommendations
- Model information dashboard
- Performance metrics overview

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/Yuraj023/machine-learning-.git
```

Move to the project folder:

```bash
cd logistic_regression/Customer_Churn_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

---

## 📷 Application Preview

### Prediction Dashboard

> *(Add a screenshot here)*

---

### About Model Dashboard

> *(Add a screenshot here)*

---

##  Learning Outcomes

This project demonstrates:

- Binary Classification using Logistic Regression
- Data Cleaning & Feature Engineering
- Pipeline-based Machine Learning
- Model Evaluation Techniques
- Deployment using Streamlit
- Building production-ready machine learning applications

---

##  Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature importance visualization
- SHAP explainability
- Cloud deployment
- REST API using FastAPI
- Docker containerization

---

##  Author

**Yuraj Chauhan**

Aspiring AI Engineer | Machine Learning | Deep Learning | Generative AI | LLMs

GitHub: https://github.com/Yuraj023

---

⭐ If you found this project helpful, consider giving it a star on GitHub!