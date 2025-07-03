# Medical-Insurance-Cost-Prediction

This project presents an end-to-end machine learning solution to predict individual medical insurance costs based on demographic and lifestyle factors such as age, gender, BMI, smoking status, and number of dependents.
# Objective
To help individuals and insurers estimate medical insurance charges using regression models built on real-world health data.
# Dataset Description
The dataset includes features such as:

•	age, sex, bmi, children, smoker, region, and charges (target)

•	Charges represent the billed amount for health insurance based on the individual's profile
# Problem Statement

The rising cost of healthcare makes it critical for insurance providers to accurately estimate individual medical expenses. Traditional pricing models often overlook the complex interactions between personal and lifestyle factors such as age, BMI, smoking habits, and number of dependents.

This project aims to build a predictive system that estimates medical insurance charges based on these factors. Accurate predictions will help both individuals plan better and insurance companies offer fair, personalized pricing.
# Proposed Solution
We propose an end-to-end regression-based machine learning system that:

•	Utilizes demographic and health-related features (age, gender, BMI, children, smoking status, region)

•	Cleans, processes, and engineers the dataset to extract meaningful patterns

•	Trains multiple regression models (including Linear Regression, Random Forest, XGBoost)

•	Tracks model performance and selects the best using MLflow

•	Deploys the best model in an interactive Streamlit application

The final app allows users to input their information and get an instant estimate of their expected medical insurance costs, while also exploring visual EDA insights from the dataset.

# Approach

Step 1: Data Preprocessing

•	Handled missing/duplicate values and encoded categorical features

•	Engineered features like BMI category and smoker-region interactions

Step 2: Regression Modeling

•	Performed EDA to explore 18 key questions

•	Trained 5+ regression models including Linear Regression, Random Forest, and XGBoost

•	Used MLflow for experiment tracking and model registry

Step 3: Streamlit App Development

•	Developed a user-friendly web interface with:

o	EDA Dashboard (18 interactive visual insights)

o	Prediction Form to estimate charges from user input

# Skills Applied

•	Python, Pandas, NumPy, Seaborn, Matplotlib

•	Streamlit, Scikit-learn, XGBoost, MLflow

•	EDA, Regression Modeling, Feature Engineering

•	Model Evaluation (RMSE, MAE, R²)

# Outcome

The app provides personalized insurance cost estimates based on user input, backed by data-driven insights. The ML pipeline is modular, explainable, and production-ready for deployment.
