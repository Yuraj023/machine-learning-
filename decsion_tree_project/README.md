# 🔋 EV Battery Failure Prediction using Decision Tree

A Machine Learning project that predicts whether an Electric Vehicle (EV) battery is likely to fail based on battery health, charging behaviour, driving patterns, environmental conditions, and maintenance history.

This project demonstrates a complete end-to-end Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, evaluation, and deployment using Streamlit.

---

## 📌 Project Overview

Electric Vehicle batteries degrade over time due to multiple operational and environmental factors. Predicting battery failure in advance helps reduce maintenance costs, improve vehicle reliability, and enhance safety.

This project builds a **Decision Tree Classification Model** capable of predicting battery failure using historical battery and vehicle data.

---

## 🎯 Problem Statement

Predict whether an EV battery will fail based on historical operational data.

**Target Variable**

```
battery_failure

0 → Healthy Battery

1 → Battery Failure
```

This is a **Binary Classification** problem.

---

# 📊 Dataset Information

| Property | Value |
|-----------|-------|
| Dataset | EV Battery Failure Dataset |
| Records | ~200,000 |
| Total Columns | 70 |
| Features Used | 67 |
| Target Column | battery_failure |
| Problem Type | Binary Classification |

---

## Dataset Features

The dataset contains information related to:

- Vehicle Specifications
- Battery Specifications
- Charging Behaviour
- Driving Behaviour
- Environmental Conditions
- Battery Health Indicators
- Maintenance History
- Electrical Measurements

Example Features

```
battery_health_percent

state_of_charge

cycle_count

battery_capacity_kwh

remaining_capacity

pack_voltage

cell_temperature_avg

charging_cycles_last_month

battery_stress_index

thermal_health_score

maintenance_score

predicted_remaining_life_cycles
```

---

# Dataset Challenges

### Missing Values

Almost every feature contained approximately **3–5% missing values**, which were handled during preprocessing.

---

### High Cardinality Columns

The following identifier columns were removed before model training:

```
vehicle_id

battery_serial
```

These columns have no predictive value and may cause overfitting.

---

### Imbalanced Dataset

Target Distribution

| Class | Percentage |
|---------|-----------:|
| Healthy Battery | 90.04% |
| Battery Failure | 9.96% |

Because the dataset is imbalanced, evaluation focuses on more than just accuracy.

Metrics used include:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

SMOTE was applied during model optimization to improve minority class prediction.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Joblib
- Streamlit

---

# 📂 Project Structure

```
decision_tree/
│
├── assets/
│   └── ev_battery_failure_dataset.csv
│
├── notebooks/
│   ├── 01_Data_Loading_Inspection.ipynb
│   ├── 02_Data_Cleaning_Preprocessing.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Decision_Tree_Model.ipynb
│   ├── 05_Model_Optimization.ipynb
│   └── 06_Streamlit_App.ipynb
│
├── model/
│   ├── best_decision_tree.pkl
│   ├── feature_importance.csv
│   └── project_report.json
│
├── outputs/
│   └── cleaned_ev_battery_dataset.csv
│
├── app/
│   └── streamlit_app.py
│
├── README.md
│
└── requirements.txt
```

---

# 🔄 Machine Learning Workflow

### 1. Data Loading

- Import dataset
- Inspect data
- Explore data types
- Check missing values
- Check duplicates

---

### 2. Data Cleaning

- Remove identifier columns
- Handle missing values
- Encode categorical variables
- Prepare dataset for training

---

### 3. Exploratory Data Analysis

- Target Distribution
- Feature Distribution
- Correlation Analysis
- Outlier Detection
- Relationship with Target Variable

---

### 4. Model Building

- Train-Test Split
- Decision Tree Classifier
- Model Training
- Prediction

---

### 5. Model Optimization

- SMOTE
- Hyperparameter Tuning
- Cross Validation
- Feature Importance
- Model Comparison

---

### 6. Deployment

Interactive Streamlit dashboard including:

- Home Dashboard
- Battery Prediction
- Dataset Analytics
- Feature Importance
- Project Information

---

# 🌳 Decision Tree Concepts Covered

This project explains the complete Decision Tree algorithm including:

- Decision Tree Fundamentals
- Root Node
- Leaf Node
- Decision Node
- Entropy
- Gini Index
- Information Gain
- Splitting Criteria
- Tree Depth
- Overfitting
- Underfitting
- Pruning
- Hyperparameter Tuning

---

# 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- Cross Validation

---

# 📊 Feature Importance

The trained model ranks the importance of every feature.

The Streamlit application visualizes the **Top 20 Most Important Features**, helping understand which battery characteristics contribute most to failure prediction.

---

# 💻 Streamlit Dashboard

The application includes multiple pages:

🏠 Home

🔋 Battery Failure Prediction

📊 Dataset Analytics

🌳 Feature Importance

ℹ️ About Project

Users can:

- Enter battery information
- Predict battery failure
- View prediction confidence
- Explore dataset statistics
- Analyze feature importance

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Yuraj023/machine-learning-.git
```

Move into the project

```bash
cd decision_tree
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

# 📦 Requirements

Main libraries used:

```
pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

streamlit

joblib
```

---

# 📚 Learning Outcomes

This project demonstrates practical knowledge of:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Decision Tree Classification
- Model Evaluation
- Handling Imbalanced Data
- Hyperparameter Optimization
- Feature Importance Analysis
- Streamlit Deployment
- Machine Learning Project Structure

---

# 📌 Future Improvements

Potential future enhancements include:

- Random Forest Classifier
- XGBoost Classifier
- LightGBM
- CatBoost
- SHAP Explainability
- Real-time Prediction API using FastAPI
- Docker Containerization
- Cloud Deployment (AWS / Azure)

---

https://github.com/Yuraj023/machine-learning-

---

## ⭐ If you found this project helpful, consider giving the repository a star!