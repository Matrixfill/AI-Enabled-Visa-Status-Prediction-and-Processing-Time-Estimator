

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


df = pd.read_csv("Visa_Dataset.csv")  

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

df["application_date"] = pd.to_datetime(df["application_date"], dayfirst=True)
df["decision_date"] = pd.to_datetime(df["decision_date"], dayfirst=True)
df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days


df = df[df["processing_days"] >= 0]
df["application_month"] = df["application_date"].dt.month
df["season"] = df["application_month"].apply(
    lambda x: "Peak" if x in [12,1,2] else "Off-Peak"
)

print("\nFeature Engineering Completed")

df_encoded = pd.get_dummies(
    df,
    columns=["country","visa_type","processing_office","season"],
    drop_first=True
)

print("Encoding Completed")

X = df_encoded.drop(
    ["processing_days","application_date","decision_date"],
    axis=1
)

y = df_encoded["processing_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,   
    random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLinear Regression Results")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

rf_params = {
    "n_estimators":[100,200],
    "max_depth":[5,10,20],
    "min_samples_split":[2,5]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=5,
    scoring="neg_mean_squared_error"
)

rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_

rf_pred = rf_best.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Best Parameters:")
print(rf_grid.best_params_)

print("\nRandom Forest Results")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

gb_params = {
    "n_estimators":[100,200],
    "learning_rate":[0.05,0.1],
    "max_depth":[3,5]
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    cv=5,
    scoring="neg_mean_squared_error"
)

gb_grid.fit(X_train, y_train)

gb_best = gb_grid.best_estimator_

gb_pred = gb_best.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("\nGradient Boosting Best Parameters:")
print(gb_grid.best_params_)

print("\nGradient Boosting Results")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R2 Score:", gb_r2)

results = pd.DataFrame({
    "Model":["Linear Regression","Random Forest","Gradient Boosting"],
    "MAE":[lr_mae,rf_mae,gb_mae],
    "RMSE":[lr_rmse,rf_rmse,gb_rmse],
    "R2 Score":[lr_r2,rf_r2,gb_r2]
})

print("\nModel Comparison")
print(results)

best_model = rf_best

joblib.dump(best_model,"visa_prediction_model.pkl")

print("\nBest Model Saved Successfully")
