import matplotlib.pyplot as plt
import seaborn as sns

print("\nStatistical Summary of Processing Days:")
print(df["processing_days"].describe())

plt.figure(figsize=(8,5))
sns.histplot(df["processing_days"], kde=True)
plt.title("Distribution of Visa Processing Days")
plt.xlabel("Processing Days")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df["processing_days"])
plt.title("Boxplot of Processing Days")
plt.show()

df["application_month"] = df["application_date"].dt.month

print("\nApplication Month Feature Added:")
print(df[["application_date", "application_month"]].head())

corr_matrix = df[["processing_days", "application_month"]].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(5,4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x="application_month",
    y="processing_days",
    data=df
)
plt.title("Processing Days vs Application Month")
plt.show()

df["season"] = df["application_month"].apply(
    lambda x: "Peak" if x in [1,2,12] else "Off-Peak"
)

print("\nSeason Feature Added:")
print(df[["application_month", "season"]].head())

country_avg = df.groupby("country")["processing_days"].mean()
df["country_avg"] = df["country"].map(country_avg)

print("\nCountry Average Processing Time Added:")
print(df[["country", "country_avg"]].head())

visa_avg = df.groupby("visa_type")["processing_days"].mean()
df["visa_avg"] = df["visa_type"].map(visa_avg)

print("\nVisa Type Average Processing Time Added:")
print(df[["visa_type", "visa_avg"]].head())