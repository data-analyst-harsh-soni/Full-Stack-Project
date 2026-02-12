# ===============================
# FASTAPI STOCK PREDICTION API
# ===============================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
# LOAD MODEL & ENCODER
# ===============================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Model & Encoder Loaded")

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv(DATA_PATH)

df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.sort_values(["company", "trade_date"])

# ===============================
# FEATURE ENGINEERING
# ===============================

df["company_encoded"] = encoder.transform(df["company"])

df["prev_close"] = df.groupby("company")["close"].shift(1)

df["ma_5"] = (
    df.groupby("company")["close"]
    .rolling(5)
    .mean()
    .reset_index(0, drop=True)
)

df["ma_10"] = (
    df.groupby("company")["close"]
    .rolling(10)
    .mean()
    .reset_index(0, drop=True)
)

df["volatility"] = df["high"] - df["low"]

df.dropna(inplace=True)

print("Feature Engineering Done")

# ===============================
# CREATE APP
# ===============================

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Stock Market API is running"
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# GET COMPANIES
# ===============================

@app.get("/companies")
def get_companies():

    companies = sorted(df["company"].unique().tolist())

    return companies


# ===============================
# GET LATEST PRICE
# ===============================

@app.get("/latest/{company}")
def get_latest(company: str):

    company_data = df[df["company"] == company]

    if company_data.empty:
        return {"error": "Company not found"}

    latest = company_data.iloc[-1]

    return {
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "close": float(latest["close"])
    }


# ===============================
# PREDICT
# ===============================

@app.post("/predict")
def predict(data: dict):

    company = data["company"]

    company_data = df[df["company"] == company]

    if company_data.empty:
        return {"error": "Company not found"}

    latest = company_data.iloc[-1]

    input_df = pd.DataFrame([{

        "company_encoded": latest["company_encoded"],

        "open": data["open"],

        "high": data["high"],

        "low": data["low"],

        "close": data["close"],

        "prev_close": latest["prev_close"],

        "ma_5": latest["ma_5"],

        "ma_10": latest["ma_10"],

        "volatility": data["high"] - data["low"]

    }])

    prediction = model.predict(input_df)[0]

    trend = "UP" if prediction > data["close"] else "DOWN"

    return {
        "prediction": float(prediction),
        "trend": trend
    }
