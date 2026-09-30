# Credit Card Fraud Detection

A machine learning-based credit card fraud detection system using **Random Forest, SMOTE, and decision threshold optimization**, with a Streamlit web application for transaction prediction.

## Project Overview

Credit card fraud detection is a binary classification problem where transactions are classified as either:

- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction

The dataset is highly imbalanced, with fraudulent transactions representing only a small portion of the total transactions. To address this class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

The final model is deployed through a **Streamlit web application** that accepts transaction features and estimates the probability of fraud.

---

## Dataset

The project uses a credit card transaction dataset containing:

- **11,958 transactions**
- **30 input features**
- **1 target variable (`Class`)**

Class distribution:

| Class | Transactions |
|------:|-------------:|
| Legitimate | 11,906 |
| Fraudulent | 52 |

The input features are:

- `Time`
- `V1` to `V28`
- `Amount`

The `V1`–`V28` features are anonymized numerical features.

---

## Machine Learning Pipeline

The implemented pipeline is:

```text
Raw Transaction Data
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
StandardScaler
        ↓
SMOTE
        ↓
Random Forest Classifier
        ↓
Fraud Probability
        ↓
Decision Threshold = 0.4
        ↓
Fraudulent / Legitimate