import pandas as pd
import joblib

# ==========================================
# Load model and training columns
# ==========================================

model = joblib.load("visa_prediction_model.pkl")
model_columns = joblib.load("model_columns.pkl")   # VERY IMPORTANT

# ==========================================
# Preprocessing function
# ==========================================

def preprocess_input(data):
    """
    Convert user input into model-ready format
    """

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Convert date
    df["application_date"] = pd.to_datetime(df["application_date"])

    # Feature Engineering
    df["application_month"] = df["application_date"].dt.month

    df["season"] = df["application_month"].apply(
        lambda x: "Peak" if x in [12,1,2] else "Off-Peak"
    )

    # Drop original date
    df = df.drop(columns=["application_date"])

    # One-hot encoding
    df = pd.get_dummies(df)

    # ==========================================
    # Fix missing columns (IMPORTANT)
    # ==========================================

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure same column order
    df = df[model_columns]

    return df


# ==========================================
# Prediction function
# ==========================================

def predict_processing_time(input_data):
    """
    Takes user input → returns predicted processing time
    """

    processed_data = preprocess_input(input_data)

    prediction = model.predict(processed_data)

    return round(prediction[0], 2)