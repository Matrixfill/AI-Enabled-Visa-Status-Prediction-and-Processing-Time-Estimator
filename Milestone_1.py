import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("Visa_Dataset.csv")


print("Missing values BEFORE handling:")
print(df.isnull().sum())


df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")



df["country"].fillna(df["country"].mode()[0], inplace=True)
df["visa_type"].fillna(df["visa_type"].mode()[0], inplace=True)
df["processing_office"].fillna(df["processing_office"].mode()[0], inplace=True)

df["application_date"].fillna(method="ffill", inplace=True)
df["decision_date"].fillna(method="ffill", inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())



df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days

print("\nProcessing days created successfully")
print(df[["application_date","decision_date","processing_days"]].head())

df = df[df["processing_days"] >= 0]


print("\nStatistics:")
print("Mean:", df["processing_days"].mean())
print("Median:", df["processing_days"].median())
print("Max:", df["processing_days"].max())
print("Min:", df["processing_days"].min())
print("Std Dev:", df["processing_days"].std())


df["risk"] = df["processing_days"].apply(
    lambda x: "High Delay" if x > 60 else "Low Delay"
)



df["normalized_processing"] = (
    (df["processing_days"] - df["processing_days"].mean())
    / df["processing_days"].std()
)



df["application_month"] = df["application_date"].dt.month
df["application_year"] = df["application_date"].dt.year

def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    else:
        return "Autumn"

df["season"] = df["application_month"].apply(get_season)


df_encoded = pd.get_dummies(
    df,
    columns=["country","visa_type","processing_office","season","risk"],
    drop_first=True
)

print("\nEncoded dataset shape:", df_encoded.shape)


df_encoded.to_csv("visa_cleaned_dataset.csv", index=False)
