# AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator

This repository contains a machine learning project that predicts **visa processing time** using historical data. The system helps users estimate how long a visa application may take based on various features like country, visa type, and processing office.

## 📌 Overview

Many applicants struggle to know how long their visa processing will take. This project uses machine learning to:

✔ Predict visa processing time (in days)  
✔ Handle real Visa data with missing values  
✔ Preprocess and clean data  
✔ Train and evaluate regression models  
✔ Provide data insights and visualizations  
✔ Prepare for a deployment interface (web app)

## 🧠 Project Goals

1. **Data Collection & Cleaning:**  
   - Handle missing values  
   - Normalize and convert date formats   
   - Create target processing time variable

2. **Exploratory Data Analysis (EDA):**  
   - Visualize distribution of processing times  
   - Compare patterns across visa types and countries

3. **Machine Learning Model:**  
   - Train regression models (e.g., RandomForest)  
   - Evaluate with metrics like MAE and R²  

4. **Prediction System:**  
   - Input user visa details  
   - Output estimated processing time

5. **Deployment-Ready:**  
   - Dataset cleaned and model saved  
   - Can be extended to a web app (Flask, Streamlit)


## 📊 Dataset Description

The dataset contains:

| Column             | Description |
|-------------------|-------------|
| application_date  | Date visa application was submitted |
| decision_date     | Date decision was made |
| country           | Applicant’s nationality |
| visa_type         | Type of visa (Student, Tourist, Work, etc.) |
| processing_office | Office processing the application |

Processing time is **calculated** as:

processing_days = decision_date − application_date

## 🛠 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/Matrixfill/AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator.git
cd AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator
````

### 2. Install dependencies

Make sure you have:

Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn


Install with:
pip install pandas numpy scikit-learn matplotlib seaborn

### 3. Run the main script
python "AI Enabled Visa Status Prediction.py"

This script:

* Loads the dataset
* Cleans and preprocesses data
* Calculates processing time
* Trains a machine learning model
* Outputs results

## 📈 Model & Evaluation

The project uses machine learning regression models (like RandomForestRegressor) to predict processing time based on features.
Model evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score


## 🧑‍💻 Example Code

Here’s a snippet for calculating processing time:

# python

df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

df["processing_days"] = (
    df["decision_date"] - df["application_date"]
).dt.days

## 🚀 Future Extensions

✔ Deploy as a web app (Flask or Streamlit)
✔ Add real-time API for predictions
✔ Add confidence intervals to predictions
✔ Add user interface and dashboard


## ⚡ 🚀 How to Run the Project

🔹 Step 1: Install dependencies
pip install -r requirements.txt

🔹 Step 2: Run the app
python -m streamlit run Milestone_4.py

## 📄 License
This project is licensed under the **MIT License**.

🌐 ☁️ Deployment
This project is deployed using Streamlit Cloud.
👉 Make sure requirements.txt includes all dependencies.

Deployment Link
https://milestone4py-agfxzsiuicvuzqph5gezk7.streamlit.app/

