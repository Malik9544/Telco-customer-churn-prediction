# Customer Churn Prediction Using Machine Learning

This project presents a complete data science pipeline to predict customer churn for a telecom service provider. The goal is to identify churn-prone customers, understand behavioral drivers, and provide actionable strategies to improve retention.


## Problem Statement

Customer churn is a key metric for subscription-based businesses. Understanding and predicting churn enables better customer retention and revenue stability. This project aims to:

- Predict which customers are likely to churn
- Identify major factors contributing to churn
- Recommend targeted interventions to reduce churn rates

---

## Dataset Overview

The dataset includes customer demographic and service details:

- Demographics: Gender, Partner, Dependents
- Services: Internet, Streaming, Online Security, etc.
- Account Information: Tenure, Contract type, Payment method, Monthly charges
- Target Variable: Churn (Yes/No)

---

## Exploratory Data Analysis (EDA)

Visualizations and summaries were used to explore:

- Churn rates across customer segments
- Tenure and monthly charge distribution
- Categorical feature influence (e.g., Contract type)
- Correlation analysis

Key findings:

- Customers on month-to-month contracts churn more
- Short-tenure customers are at higher churn risk
- Higher monthly charges are linked with higher churn

---

## Data Preprocessing

Steps included:

- Encoding categorical features using Label and One-Hot encoding
- Handling missing values and irrelevant entries
- Scaling numerical variables
- Class imbalance addressed using class weights in model training

---

## Model Development

Three classification models were developed:

- Logistic Regression
- Random Forest
- XGBoost

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

XGBoost performed best across metrics.

---

## Model Interpretability

SHAP (SHapley Additive Explanations) was used to interpret predictions:

- Global feature importance rankings
- Individual customer prediction explanations
- Summary and force plots

Most influential features:

- Monthly Charges
- Contract Type
- Tenure
- Internet Service

---

## Business Insights

1. **Short-tenure customers churn frequently**  
   - Introduce onboarding benefits or loyalty programs during the early customer lifecycle.

2. **Month-to-month contract customers are unstable**  
   - Offer incentives to switch to long-term plans.

3. **High charges increase churn probability**  
   - Provide flexible pricing or targeted discount options for price-sensitive users.

---

## Visual Assets

All visuals (heatmaps, SHAP plots, pie charts, etc.) are stored in the `/charts` directory for easy review or presentation purposes.

---

## Deliverables

- Cleaned Jupyter Notebook
- Encoded and processed dataset
- Visualization assets
- Insight summary slide
- README documentation

---

## Technologies Used

- Python (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap)
- Jupyter Notebook
- SHAP for model explainability

---

## Business Value

This end-to-end machine learning pipeline can help telecom companies:

- Proactively reduce churn
- Improve customer lifetime value
- Target the right users with the right offers

---

## Future Enhancements

- Build a web dashboard for real-time churn prediction
- Integrate churn scores into CRM for sales/marketing use
- Monitor and retrain models on updated datasets periodically

---

## Author

Muhammad Mudasir  
LinkedIn: # Customer Churn Prediction Using Machine Learning

This project presents a complete data science pipeline to predict customer churn for a telecom service provider. The goal is to identify churn-prone customers, understand behavioral drivers, and provide actionable strategies to improve retention.

---

## Problem Statement

Customer churn is a key metric for subscription-based businesses. Understanding and predicting churn enables better customer retention and revenue stability. This project aims to:

- Predict which customers are likely to churn
- Identify major factors contributing to churn
- Recommend targeted interventions to reduce churn rates

---

## Dataset Overview

The dataset includes customer demographic and service details:

- Demographics: Gender, Partner, Dependents
- Services: Internet, Streaming, Online Security, etc.
- Account Information: Tenure, Contract type, Payment method, Monthly charges
- Target Variable: Churn (Yes/No)

---

## Exploratory Data Analysis (EDA)

Visualizations and summaries were used to explore:

- Churn rates across customer segments
- Tenure and monthly charge distribution
- Categorical feature influence (e.g., Contract type)
- Correlation analysis

Key findings:

- Customers on month-to-month contracts churn more
- Short-tenure customers are at higher churn risk
- Higher monthly charges are linked with higher churn

---

## Data Preprocessing

Steps included:

- Encoding categorical features using Label and One-Hot encoding
- Handling missing values and irrelevant entries
- Scaling numerical variables
- Class imbalance addressed using class weights in model training

---

## Model Development

Three classification models were developed:

- Logistic Regression
- Random Forest
- XGBoost

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

XGBoost performed best across metrics.

---

## Model Interpretability

SHAP (SHapley Additive Explanations) was used to interpret predictions:

- Global feature importance rankings
- Individual customer prediction explanations
- Summary and force plots

Most influential features:

- Monthly Charges
- Contract Type
- Tenure
- Internet Service

---

## Business Insights

1. **Short-tenure customers churn frequently**  
   - Introduce onboarding benefits or loyalty programs during the early customer lifecycle.

2. **Month-to-month contract customers are unstable**  
   - Offer incentives to switch to long-term plans.

3. **High charges increase churn probability**  
   - Provide flexible pricing or targeted discount options for price-sensitive users.

---

## Visual Assets

All visuals (heatmaps, SHAP plots, pie charts, etc.) are stored in the `/charts` directory for easy review or presentation purposes.

---

## Deliverables

- Cleaned Jupyter Notebook
- Encoded and processed dataset
- Visualization assets
- Insight summary slide
- README documentation

---

## Technologies Used

- Python (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap)
- Jupyter Notebook
- SHAP for model explainability

---

## Business Value

This end-to-end machine learning pipeline can help telecom companies:

- Proactively reduce churn
- Improve customer lifetime value
- Target the right users with the right offers

---

## Future Enhancements

- Build a web dashboard for real-time churn prediction
- Integrate churn scores into CRM for sales/marketing use
- Monitor and retrain models on updated datasets periodically

---

## Author

Muhammad Mudasir  
LinkedIn: www.linkedin.com/in/mudasir-malik-000a7822a


GitHub: [github.com](https://github.com)
