import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Loan Prediction", page_icon="💰", layout="centered")

st.title("💰 Loan Approval Prediction App")
st.markdown("### Enter Applicant Details")

# -------------------------------
# Load and Prepare Data
# -------------------------------
df = pd.read_csv("LoanApprovalPrediction.csv")

# Handle missing values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fix special values
df['Dependents'] = df['Dependents'].replace('3+', '3')

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & target
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# -------------------------------
# UI INPUT (Beautiful Layout)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Married", ["Yes", "No"])
    dependents = st.selectbox("👨‍👩‍👧 Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("💰 Applicant Income", min_value=0)
    coapplicant_income = st.number_input("💰 Coapplicant Income", min_value=0)
    loan_amount = st.number_input("🏦 Loan Amount", min_value=0)
    loan_term = st.number_input("⏳ Loan Term", min_value=0)
    credit_history = st.selectbox("📊 Credit History", ["Good", "Bad"])
    property_area = st.selectbox("🏠 Property Area", ["Rural", "Semiurban", "Urban"])

# -------------------------------
# Convert Inputs to Numbers
# -------------------------------
def convert_input():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 1 if education == "Graduate" else 0
    self_emp_val = 1 if self_employed == "Yes" else 0
    credit_val = 1 if credit_history == "Good" else 0

    property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_val = property_map[property_area]

    return [[
        gender_val, married_val, dependents_val, education_val, self_emp_val,
        applicant_income, coapplicant_income, loan_amount,
        loan_term, credit_val, property_val
    ]]

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("---")

if st.button("🚀 Predict Loan Status"):
    input_data = convert_input()
    input_data = scaler.transform(input_data)

    result = model.predict(input_data)

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")