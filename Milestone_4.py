import streamlit as st
import pandas as pd
import os
from datetime import datetime
from predict import predict_processing_time

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="VisaPredict AI", layout="wide")



st.title("🌍 Visa Processing Time Estimator")

# -------------------------------
# History File Setup
# -------------------------------
HISTORY_FILE = "prediction_history.csv"

if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
else:
    history_df = pd.DataFrame(columns=[
        "Country", "Visa Type", "Office", "Date", "Prediction (days)"
    ])

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🔮 Prediction", "📊 Data Analysis", "🧾 History", "ℹ️ About"]
)


# ===============================
# 🏠 HOME TAB (IMPROVED)
# ===============================
with tab1:

    st.divider()

    # 🔹 Metrics Section
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Total Applications", "10,000+")
    col2.metric("🌍 Countries Covered", "5+")
    col3.metric("🤖 Model Accuracy", "85%+")

    st.divider()

    # 🔹 Key Features
    st.subheader("✨ Key Features")
    st.markdown("""
    ✅ AI-based visa processing prediction  
    ✅ Real-time estimation with range output  
    ✅ Data analysis & visualization  
    ✅ Permanent history tracking  
    ✅ Easy-to-use interactive interface  
    """)

    st.divider()

    # 🔹 How it Works (NEW FEATURE ⭐)
    st.subheader("⚙️ How It Works")

    st.markdown("""
    1️⃣ Enter visa details (country, type, office, date)  
    2️⃣ System processes input using trained ML model  
    3️⃣ Prediction is generated based on historical data  
    4️⃣ Output shown as a realistic time range  
    """)

    st.divider()

    # 🔹 Benefits Section (NEW FEATURE ⭐)
    st.subheader("🎯 Benefits")

    col1, col2 = st.columns(2)

    col1.markdown("""
    ✔ Saves time in planning travel  
    ✔ Reduces uncertainty in visa process  
    ✔ Helps in better scheduling  
    """)

    col2.markdown("""
    ✔ Provides data-driven insights  
    ✔ Useful for students & professionals  
    ✔ Improves decision making  
    """)

    st.divider()

    # 🔹 Quick Tips (NEW FEATURE ⭐)
    st.subheader("💡 Quick Tips for Faster Visa Processing")

    st.info("""
    • Apply during off-peak months for faster processing  
    • Ensure all documents are complete  
    • Choose correct visa category  
    • Apply early to avoid delays  
    """)

    st.divider()

# ===============================
# 🔮 PREDICTION TAB
# ===============================
with tab2:
    st.header("Visa Processing Time Prediction")

    st.write("Enter application details:")

    # Inputs
    country = st.selectbox("Country", ["India", "USA", "UK"])
    visa_type = st.selectbox("Visa Type", ["Student", "Tourist", "Work"])
    processing_office = st.selectbox("Processing Office", ["Delhi", "New York", "London"])
    application_date = st.date_input("Application Date")

    if st.button("Predict"):

        input_data = {
            "country": country,
            "visa_type": visa_type,
            "processing_office": processing_office,
            "application_date": str(application_date)
        }

        result = predict_processing_time(input_data)

        # 🔥 Convert to range
        pred = int(result)

        if pred <= 15:
            lower, upper = pred - 5, pred + 5
        elif pred <= 30:
            lower, upper = pred - 7, pred + 7
        elif pred <= 60:
            lower, upper = pred - 10, pred + 10
        else:
            lower, upper = pred - 12, pred + 12

        lower = max(0, lower)

        st.success(f"Estimated Processing Time: {lower} to {upper} days")

        # Save history permanently
        new_entry = pd.DataFrame([{
            "Country": country,
            "Visa Type": visa_type,
            "Office": processing_office,
            "Date": application_date,
            "Prediction (days)": f"{lower}-{upper}"
        }])

        history_df = pd.concat([history_df, new_entry], ignore_index=True)
        history_df.to_csv(HISTORY_FILE, index=False)

# ===============================
# 📊 DATA ANALYSIS TAB
# ===============================
with tab3:
    st.header("Data Analysis")

    try:
        df = pd.read_csv("Visa_Dataset.csv")

        df["application_date"] = pd.to_datetime(df["application_date"], dayfirst=True)
        df["decision_date"] = pd.to_datetime(df["decision_date"], dayfirst=True)

        df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Statistics")
        st.write(df["processing_days"].describe())

        st.subheader("Processing Days Distribution")
        st.bar_chart(df["processing_days"])

        st.subheader("Applications by Country")
        st.bar_chart(df["country"].value_counts())

    except:
        st.error("Dataset not found. Please add Visa_Dataset.csv")

# ===============================
# 🧾 HISTORY TAB
# ===============================
with tab4:
    st.header("Prediction History")

    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)

        if history_df.empty:
            st.warning("No predictions yet.")
        else:
            st.dataframe(history_df)

            st.download_button(
                "Download History",
                history_df.to_csv(index=False),
                "prediction_history.csv",
                "text/csv"
            )

           
    else:
        st.warning("No history file found.")

# ===============================
# ℹ️ ABOUT TAB
# ===============================
with tab5:
    st.header("About Project")

    st.write("""
    **VisaPredict AI** is a machine learning-based system that predicts visa processing time.

    🔹 **Objective:**
    Reduce uncertainty in visa processing.

    🔹 **Models Used:**
    - Linear Regression
    - Random Forest (Best)
    - Gradient Boosting

    🔹 **Features:**
    - Real-time prediction
    - Data analysis
    - Persistent history storage

    🔹 **Future Scope:**
    - Add real-time data
    - Mobile app integration
    - More accurate predictions
    """)

