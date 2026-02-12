# ===============================
# STOCK PRICE PREDICTION SCRIPT (FINAL VERSION)
# ===============================

import pandas as pd
import joblib
import os

# ===============================
# PATH SETUP
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "stock_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "company_encoder.pkl")

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "stock_market_clean_dataset_with_Feature_Eng",
    "nse_prices.csv"
)

# ===============================
# LOAD MODEL AND ENCODER
# ===============================

print("Loading Model and Encoder...")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Model and Encoder Loaded Successfully")


# ===============================
# LOAD DATASET (SAFE LOADING)
# ===============================

print("Loading Dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Remove duplicate header row if exists
df = df[df["company"] != "company"]

# Clean company column
df["company"] = df["company"].astype(str).str.strip().str.upper()

# Convert numeric columns safely
numeric_cols = ["open", "high", "low", "close"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert date safely (DD-MM-YYYY FIX)
df["trade_date"] = pd.to_datetime(
    df["trade_date"],
    dayfirst=True,
    errors="coerce"
)

# Remove invalid rows
df.dropna(inplace=True)

# Sort properly
df.sort_values(["company", "trade_date"], inplace=True)

df.reset_index(drop=True, inplace=True)

print("Dataset Loaded Successfully")


# ===============================
# FEATURE ENGINEERING
# ===============================

print("Performing Feature Engineering...")

# Safe encoding
def safe_encode(company):
    if company in encoder.classes_:
        return encoder.transform([company])[0]
    return None

df["company_encoded"] = df["company"].apply(safe_encode)

df.dropna(subset=["company_encoded"], inplace=True)

# Previous close
df["prev_close"] = df.groupby("company")["close"].shift(1)

# Moving averages
df["ma_5"] = (
    df.groupby("company")["close"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

df["ma_10"] = (
    df.groupby("company")["close"]
    .rolling(10)
    .mean()
    .reset_index(level=0, drop=True)
)

# Volatility
df["volatility"] = df["high"] - df["low"]

df.dropna(inplace=True)

print("Feature Engineering Completed")


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_company(company_name):

    company_name = company_name.strip().upper()

    # Check encoder
    if company_name not in encoder.classes_:
        print(f"❌ Company '{company_name}' not found in trained model")
        return None

    company_data = df[df["company"] == company_name]

    if company_data.empty:
        print(f"❌ No dataset data found for {company_name}")
        return None

    latest = company_data.iloc[-1]

    print("\n===============================")
    print("Company:", company_name)
    print("Current Price:", latest["close"])

    # Prepare input
    input_df = pd.DataFrame([{
        "company_encoded": latest["company_encoded"],
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "close": latest["close"],
        "prev_close": latest["prev_close"],
        "ma_5": latest["ma_5"],
        "ma_10": latest["ma_10"],
        "volatility": latest["volatility"]
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    predicted_price = round(float(prediction), 2)

    print("Predicted Next Day Price:", predicted_price)

    if predicted_price > latest["close"]:
        print("Trend: UP 📈")
    else:
        print("Trend: DOWN 📉")

    print("===============================")

    return predicted_price


# ===============================
# TERMINAL EXECUTION
# ===============================

if __name__ == "__main__":

    print("\nAvailable Companies:\n")

    companies = sorted(df["company"].unique())

    for comp in companies[:50]:
        print(comp)

    company = input("\nEnter Company Name: ").strip().upper()

    result = predict_company(company)

    if result is not None:
        print("\nPrediction Completed Successfully")
