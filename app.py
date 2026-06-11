import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Apex Bank | Loan AI", layout="centered")
st.title("🏦 Apex Bank Loan Approval AI")
st.write("Input the applicant's metrics below to evaluate loan eligibility.")
st.divider()

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_tree_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"The actual error is: {e}")
    st.stop()
# Layout
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income ($)", 15000, 200000, 60000, step=1000)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    employment_years = st.slider("Years of Employment", 0, 40, 5)
with col2:
    existing_debt = st.number_input("Existing Annual Debt ($)", 0, 50000, 12000, step=500)
    loan_amount = st.number_input("Requested Loan ($)", 20000, 800000, 200000, step=1000)
    property_value = st.number_input("Property Value ($)", 50000, 1000000, 250000, step=1000)

col3, col4 = st.columns(2)
with col3:
    education = st.selectbox("Highest Education", ["High School", "Bachelors", "Masters/PhD"])
with col4:
    prior_default = st.radio("Prior Loan Default?", ["No", "Yes"])

st.divider()

# Background Math & Feature Engineering
dti_ratio = existing_debt / income
ltv_ratio = loan_amount / property_value
stability_score = (employment_years + 1) * (income / 10000)
default_flag = 1 if prior_default == "Yes" else 0

is_hs = 1 if education == "High School" else 0
is_ba = 1 if education == "Bachelors" else 0
is_ma = 1 if education == "Masters/PhD" else 0

# Match X_train columns exactly
input_data = pd.DataFrame([[
    income, credit_score, employment_years, existing_debt, loan_amount, property_value,
    default_flag, dti_ratio, ltv_ratio, stability_score, is_ba, is_hs, is_ma
]], columns=[
    'Applicant_Income', 'Credit_Score', 'Employment_Years', 'Existing_Debt_Annual', 
    'Loan_Amount_Requested', 'Property_Value', 'Prior_Default', 'Debt_To_Income_Ratio', 
    'Loan_To_Value_Ratio', 'Income_Stability_Score', 'Education_Level_Bachelors', 
    'Education_Level_High School', 'Education_Level_Masters/PhD'
])

st.subheader("📊 Algorithmic Decision")
m1, m2, m3 = st.columns(3)
m1.metric("Debt-to-Income", f"{dti_ratio:.1%}")
m2.metric("Loan-to-Value", f"{ltv_ratio:.1%}")
m3.metric("Stability Score", f"{stability_score:.2f}")

if st.button("Run Credit Decision", type="primary"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ **LOAN APPROVED**")
        st.balloons()
    else:
        st.error("❌ **LOAN DENIED**")
