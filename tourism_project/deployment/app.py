
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Tourism Product Prediction", page_icon="✈️", layout="centered")

# Download and load the trained model
# Replace repo_id and filename with your actual Hugging Face model repo details
model_path = hf_hub_download(
    repo_id="ArakdeeD71/tourism_prodtaken_model",
    filename="best_tourism_prodtaken_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("Tourism Product Prediction")
st.write(
    """
    This application predicts whether a customer will purchase the tourism product
    (`ProdTaken`) based on customer profile, pitch details, travel history, and income.
    Enter the customer details below and click the prediction button.
    """
)

# Categorical inputs
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
citytier = st.number_input("City Tier", min_value=1, max_value=3, value=1, step=1)
durationofpitch = st.number_input("Duration of Pitch", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
numberoffollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3, step=1)
preferredpropertystar = st.number_input("Preferred Property Star", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
numberoftrips = st.number_input("Number of Trips", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
passport = st.selectbox("Passport", ["No", "Yes"])
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
owncar = st.selectbox("Own Car", ["No", "Yes"])
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=20000.0, step=500.0)

passport_value = 1 if passport == "Yes" else 0
owncar_value = 1 if owncar == "Yes" else 0

# Build input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "NumberOfFollowups": numberoffollowups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport_value,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "OwnCar": owncar_value,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome
}])

if st.button("Predict ProdTaken"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Customer is likely to take the product")
    else:
        st.error("Prediction: Customer is not likely to take the product")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        st.info(f"Probability of ProdTaken = 1: {probability:.2%}")
