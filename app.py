import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# Page config
st.set_page_config(page_title="Medical Insurance Dashboard", layout="wide")

# Load the data
df = pd.read_csv("D:/Project/Guvi_Project/Medical Insurance Cost Prediction/medical_insurance.csv")
model = joblib.load('C:/Users/hp/saved_insurance_models/XGBoost_20250703_170025.joblib')

# Optional: Add BMI category
def classify_bmi(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif 18.5 <= bmi < 25:
        return 'normal'
    elif 25 <= bmi < 30:
        return 'overweight'
    else:
        return 'obese'
df['bmi_category'] = df['bmi'].apply(classify_bmi)

# Sidebar menu
menu = st.sidebar.selectbox(
    "📌 Select Section",
    ["Introduction", "EDA", "Prediction"]
)

# 1️⃣ Introduction Page
if menu == "Introduction":
    st.title("🏥 Medical Insurance Cost Analysis & Prediction")
    st.markdown("""
    This project presents an end-to-end machine learning solution to predict individual medical insurance costs based on demographic and lifestyle factors such as age, gender, BMI, smoking status, and number of dependents.
    
    #### Objective:
    
    To help individuals and insurers estimate medical insurance charges using regression models built on real-world health data.
    
    #### Dataset Description:
    
    The dataset includes features such as:
    
        •	age, sex, bmi, children, smoker, region, and charges (target)
        
        •	Charges represent the billed amount for health insurance based on the individual's profile

    #### Problem Statement:
    
    With healthcare costs on the rise, insurance providers need accurate tools to estimate individual medical expenses. Traditional models often miss the complex relationships between factors like age, BMI, smoking status, and number of dependents. This project addresses that gap by building a predictive system to estimate insurance charges more accurately, helping individuals plan and enabling insurers to offer fair, personalized pricing.
    
    #### Proposed Solution:
    
    We developed an end-to-end machine learning pipeline that processes health and demographic data to train multiple regression models (e.g., Linear Regression, Random Forest, XGBoost). The best model is tracked and selected using MLflow, then deployed in a Streamlit app. Users can explore interactive EDA insights and input their details to instantly predict their expected medical insurance cost.
    
    #### Skills Applied:
    
        •	Python, Pandas, NumPy, Seaborn, Matplotlib
        
        •	Streamlit, Scikit-learn, XGBoost, MLflow
        
        •	EDA, Regression Modeling, Feature Engineering
        
        •	Model Evaluation (RMSE, MAE, R²)
    
    """)

# 2️⃣ EDA Section
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    question = st.sidebar.selectbox("Choose a question to explore", [
        "1. Distribution of medical insurance charges",
        "2. Age distribution of individuals",
        "3. Count of smokers vs non-smokers",
        "4. Average BMI in the dataset",
        "5. Regions with most policyholders",
        "6. How do charges vary with age?",
        "7. Is there a difference in average charges between smokers and non-smokers?",
        "8. Does BMI impact insurance charges?",
        "9. Do men or women pay more on average?",
        "10. Is there a correlation between the number of children and the insurance charges?",
        "11. How does smoking status combined with age affect medical charges?",
        "12. How do age, BMI, and smoking status together affect insurance cost?",
        "13. Are there outliers in the charges column? Who are the individuals paying the highest costs?",
        "14. Are there extreme BMI values that could skew predictions?",
        "15. What is the correlation between numeric features like age, BMI, number of children, and charges?",
        "16. Which features have the strongest correlation with the target variable (charges)?"
    ])

    # 1. Charges Distribution
    if question == "1. Distribution of medical insurance charges":
        st.subheader("1️⃣ Distribution of Medical Insurance Charges")
        fig, ax = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax)
        ax.set_title("Charges Distribution")
        st.pyplot(fig)

    # 2. Age Distribution
    elif question == "2. Age distribution of individuals":
        st.subheader("2️⃣ Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, bins=20, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    # 3. Smokers vs Non-Smokers
    elif question == "3. Count of smokers vs non-smokers":
        st.subheader("3️⃣ Smokers vs Non-Smokers")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='smoker', palette='Set2', ax=ax)
        ax.set_title("Smokers vs Non-Smokers")
        st.pyplot(fig)

    # 4. Average BMI
    elif question == "4. Average BMI in the dataset":
        st.subheader("4️⃣ Average BMI")
        avg_bmi = round(df['bmi'].mean(), 2)
        st.metric("📌 Average BMI", avg_bmi)

        fig, ax = plt.subplots()
        sns.histplot(df['bmi'], kde=True, ax=ax)
        ax.axvline(avg_bmi, color='red', linestyle='--', label=f"Mean: {avg_bmi}")
        ax.legend()
        ax.set_title("BMI Distribution")
        st.pyplot(fig)

    # 5. Regions with Most Policyholders
    elif question == "5. Regions with most policyholders":
        t.subheader("5️⃣ Policyholders by Region")

        region_counts = {
            "Northwest": df['region_northwest'].sum(),
            "Southeast": df['region_southeast'].sum(),
            "Southwest": df['region_southwest'].sum(),
            "Northeast": len(df) - (df['region_northwest'] + df['region_southeast'] + df['region_southwest']).sum()
        }

        region_df = pd.DataFrame(list(region_counts.items()), columns=["Region", "Count"])

        fig, ax = plt.subplots()
        sns.barplot(data=region_df, x="Region", y="Count", palette="pastel", ax=ax)
        ax.set_title("Number of Policyholders by Region")
        st.pyplot(fig)
    # 6. Charges vs Age
    elif question == "6. How do charges vary with age?":
        st.subheader("6️⃣ Charges vs Age")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.7, ax=ax)
        ax.set_title("Insurance Charges by Age (colored by Smoking Status)")
        st.pyplot(fig)

    # 7. Average charges: Smokers vs Non-smokers
    elif question == "7. Is there a difference in average charges between smokers and non-smokers?":
        st.subheader("7️⃣ Average Charges: Smokers vs Non-Smokers")
        avg_charges = df.groupby("smoker")["charges"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=avg_charges, x='smoker', y='charges', palette='Set2', ax=ax)
        ax.set_title("Average Charges by Smoking Status")
        st.pyplot(fig)

    # 8. BMI vs Charges
    elif question == "8. Does BMI impact insurance charges?":
        st.subheader("8️⃣ BMI vs Charges")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.6, ax=ax)
        ax.set_title("Charges vs BMI (colored by Smoking Status)")
        st.pyplot(fig)

    # 9. Charges by Gender
    elif question == "9. Do men or women pay more on average?":
        st.subheader("9️⃣ Average Charges by Gender")
        avg_gender = df.groupby("sex")["charges"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=avg_gender, x='sex', y='charges', palette='pastel', ax=ax)
        ax.set_title("Average Insurance Charges by Gender")
        st.pyplot(fig)

    # 10. Children vs Charges
    elif question == "10. Is there a correlation between the number of children and the insurance charges?":
        st.subheader("🔟 Charges vs Number of Children")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='children', y='charges', palette='coolwarm', ax=ax)
        ax.set_title("Charges by Number of Children")
        st.pyplot(fig)

    # 11. Age + Smoking status vs Charges
    elif question == "11. How does smoking status combined with age affect medical charges?":
        st.subheader("1️⃣1️⃣ Age + Smoking Status vs Charges")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.7, ax=ax)
        ax.set_title("Charges by Age and Smoking Status")
        st.pyplot(fig)

    # 12. Age, BMI, and Smoking Status on Charges
    elif question == "12. How do age, BMI, and smoking status together affect insurance cost?":
        st.subheader("1️⃣3️⃣ Age, BMI & Smoking Status vs Charges")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', size='bmi', sizes=(20, 200), alpha=0.6, ax=ax)
        ax.set_title("Charges by Age, BMI, and Smoking Status")
        st.pyplot(fig)
    #  13. Are there outliers in the charges column? Who are the individuals paying the highest costs?
    elif question == "13. Are there outliers in the charges column? Who are the individuals paying the highest costs?":
        st.subheader("1️⃣5️⃣ Outliers in Medical Charges")

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(data=df, y='charges', color='lightblue', ax=ax)
        ax.set_title("Boxplot of Medical Charges")
        st.pyplot(fig)

        # Top 5 highest charges
        st.markdown("### 💰 Top 5 Highest Paying Individuals")
        top_charges = df.sort_values(by='charges', ascending=False).head(5)
        st.dataframe(top_charges[['age', 'sex', 'bmi', 'children', 'smoker', 'charges']])

    #  14. Are there extreme BMI values that could skew predictions?
    elif question == "14. Are there extreme BMI values that could skew predictions?":
        st.subheader("1️⃣6️⃣ Outliers in BMI")

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(data=df, y='bmi', color='salmon', ax=ax)
        ax.set_title("Boxplot of BMI Values")
        st.pyplot(fig)

        # Threshold for extreme BMI
        high_bmi = df[df['bmi'] > 40]
        if not high_bmi.empty:
            st.markdown("### ⚠️ Individuals with BMI > 40 (Extreme Obesity)")
            st.dataframe(high_bmi[['age', 'sex', 'bmi', 'smoker', 'charges']])
        else:
            st.success("✅ No extreme BMI values (BMI > 40) found in the dataset.")

    # 15. What is the correlation between numeric features like age, BMI, number of children, and charges?
    elif question == "15. What is the correlation between numeric features like age, BMI, number of children, and charges?":
        st.subheader("1️⃣7️⃣ Correlation Between Numeric Features")

        # Select only numeric features
        numeric_cols = ['age', 'bmi', 'children', 'charges', 'log_charges']
        corr = df[numeric_cols].corr()

        # Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap of Numeric Features")
        st.pyplot(fig)

    #  16. Which features have the strongest correlation with the target variable (charges)?
    elif question == "16. Which features have the strongest correlation with the target variable (charges)?":
        st.subheader("1️⃣8️⃣ Correlation with Insurance Charges")

        # Only include desired numeric features (excluding 'log_charges' and 'charges' itself)
        numeric_cols = ['age', 'bmi', 'children', 'smoker', 'sex']
        numeric_corr = df[numeric_cols + ['charges']].corr()['charges'].drop('charges').sort_values(ascending=False)

        # Bar chart
        fig, ax = plt.subplots()
        sns.barplot(x=numeric_corr.values, y=numeric_corr.index, palette='viridis', ax=ax)
        ax.set_title("Features Most Correlated with Charges (Excluding log_charges)")
        ax.set_xlabel("Correlation Coefficient")
        st.pyplot(fig)

        st.markdown(f"📌 **Top Positively Correlated Feature**: `{numeric_corr.idxmax()}`")
        st.markdown(f"📌 **Top Negatively Correlated Feature**: `{numeric_corr.idxmin()}`")

# 3️⃣ Prediction Section
elif menu == "Prediction":
    st.title("🏥 Medical Insurance Cost Predictor")

    # User input form
    with st.form("input_form"):
        st.header("Patient Details")
    
        col1, col2 = st.columns(2)
    
        with col1:
            age = st.slider("Age", 18, 70, 30)
            bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
            children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
    
        with col2:
            sex = st.radio("Sex", ["Male", "Female"])
            smoker = st.radio("Smoker", ["No", "Yes"])
            region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    
        submitted = st.form_submit_button("Predict Cost")

    # When form is submitted
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == "Female" else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == "Yes" else 0],
            'region_northwest': [1 if region == "Northwest" else 0],
            'region_southeast': [1 if region == "Southeast" else 0],
            'region_southwest': [1 if region == "Southwest" else 0]
        })
    
        # Make prediction
        prediction = model.predict(input_data)[0]
    
        # Display results
        st.success(f"### Predicted Insurance Cost: ${prediction:,.2f}")
    
        # Show input summary
        with st.expander("See input details"):
            st.json(input_data.to_dict(orient='records')[0])
