
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(
    r"stock_market_clean_dataset_with_Feature_Eng\nse_prices.csv"
)

print("Dataset Loaded")


df = df.dropna()

df["trade_date"] = pd.to_datetime(df["trade_date"])

df = df.sort_values(["company", "trade_date"])


encoder = LabelEncoder()

df["company_encoded"] = encoder.fit_transform(df["company"])


df["prev_close"] = df.groupby("company")["close"].shift(1)

df["ma_5"] = df.groupby("company")["close"].rolling(5).mean().reset_index(0, drop=True)

df["ma_10"] = df.groupby("company")["close"].rolling(10).mean().reset_index(0, drop=True)

df["volatility"] = df["high"] - df["low"]

df["target"] = df.groupby("company")["close"].shift(-1)

df = df.dropna()

print("Feature Engineering Done")


features = [
    "company_encoded",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "ma_5",
    "ma_10",
    "volatility"
]

X = df[features]
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)



model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Trained")


predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nRESULTS:")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))



joblib.dump(
    model,
    r"FULL_STACK_FROJECT\ML Model\stock_model.pkl"
)

joblib.dump(
    encoder,
    r"FULL_STACK_FROJECT\ML Model\company_encoder.pkl"
)


print("\nModel Saved Successfully")


sample = X_test.iloc[-1:]

pred = model.predict(sample)

print("\nPrediction:", round(pred[0], 2))
print("Actual:", round(y_test.iloc[-1], 2))
