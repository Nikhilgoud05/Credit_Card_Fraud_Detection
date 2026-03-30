# Credit Card Fraud Detection

This repository contains a comprehensive Machine Learning pipeline designed to identify fraudulent credit card transactions. The project addresses the critical challenge of **class imbalance**—where fraudulent transactions are significantly rarer than legitimate ones—to build a model that prioritizes the detection of actual fraud (Recall) without overwhelming the system with false alarms (Precision).

## 📊 Dataset Overview

The dataset contains transactions made by credit cards in September 2013 by European cardholders.[cite: 1]

  - **Total Transactions**: 284,807[cite: 1]
  - **Fraudulent Transactions**: 492 (0.172% of the total)[cite: 1]
  - **Features**: Features V1-V28 are numerical variables resulting from a **PCA transformation** to protect user identities. 'Time' and 'Amount' are the only features not transformed.[cite: 1]
  - **Target**: `Class` (1 for Fraud, 0 for Legit).[cite: 1]

## 🛠️ Technical Workflow

### 1\. Data Preprocessing & Analysis

  - **Missing Values**: The dataset was verified to have zero missing values across all columns.[cite: 1]
  - **Exploratory Data Analysis (EDA)**: Statistical measures were compared between legit and fraudulent transactions. It was noted that the mean transaction amount for fraud ($122.21) is higher than for legit transactions ($88.29).[cite: 1]
  - **Feature Scaling**: Applied `StandardScaler` to normalize features for improved model convergence.[cite: 2]

### 2\. Handling Imbalanced Data

To prevent the model from being biased toward legitimate transactions, two distinct strategies were implemented:

  - **Under-sampling**: Created a balanced sub-dataset by randomly selecting 492 legit transactions to match the number of fraud cases.[cite: 1]
  - **SMOTE (Synthetic Minority Over-sampling Technique)**: Over-sampled the minority class by creating synthetic examples in the training set, resulting in an equal distribution of both classes (227,451 samples each).[cite: 2]

### 3\. Model Training & Comparison

The project evaluated several high-performance algorithms:

  - **Logistic Regression**: Used as a baseline classifier.[cite: 1, 2]
  - **Random Forest**: An ensemble method to capture non-linear relationships.[cite: 1]
  - **XGBoost**: A gradient-boosted decision tree framework optimized for speed and performance.[cite: 1, 2]

## 📈 Performance Metrics

Since accuracy is misleading in imbalanced datasets, the following metrics were prioritized on the test data:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (SMOTE)** | 97.3% | 0.06 | **0.95** | 0.11 |
| **XGBoost (SMOTE)** | **99.5%** | **0.26** | 0.89 | **0.40** |

*Note: While Logistic Regression achieved higher Recall, XGBoost provided a much better balance between Precision and Recall (F1-Score).*[cite: 2]

## 🚀 How to Run

1.  Clone the repository.
2.  Ensure you have the `creditcard.csv` dataset in the root directory.
3.  Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
    ```
4.  Run the Jupyter Notebook to train the models and view the visualizations.

## 👤 Author

**Kollu Nikhil Goud**

  - GitHub: [@Nikhilgoud05](https://www.google.com/search?q=https://github.com/Nikhilgoud05)


### Suggested Follow-up

Would you like me to help you add a section to this README on how to interpret the **Confusion Matrix** results for recruiters?
