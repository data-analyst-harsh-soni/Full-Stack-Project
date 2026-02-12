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
# CREATE FASTAPI APP
# ===============================

app = FastAPI(title="Stock Market Prediction API")

# ===============================
# ENABLE CORS (frontend connect)
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# LOAD MODEL & ENCODER
# ===============================

model = None
encoder = None

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and Encoder Loaded Successfully")

except Exception as e:
    print("❌ Model load error:", str(e))


# ===============================
# LOAD DATASET
# ===============================

df = None

try:
    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"
    )

    # FIX DATE FORMAT (IMPORTANT)
    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        dayfirst=True,
        errors="coerce"
    )

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

    print("✅ Dataset Loaded and Feature Engineering Completed")

except Exception as e:
    print("❌ Dataset load error:", str(e))


# ===============================
# HOME ROUTE
# ===============================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Stock Market Prediction API is working"
    }


# ===============================
# GET COMPANIES LIST
# ===============================

@app.get("/companies")
def get_companies():

    if df is None:
        return {"error": "Dataset not loaded"}

    return sorted(df["company"].unique().tolist())


# ===============================
# GET LATEST PRICE
# ===============================

@app.get("/latest/{company}")
def get_latest(company: str):

    if df is None:
        return {"error": "Dataset not loaded"}

    company_data = df[df["company"] == company]

    if company_data.empty:
        return {"error": "Company not found"}

    latest = company_data.iloc[-1]

    return {
        "company": company,
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "close": float(latest["close"])
    }


# ===============================
# PREDICT STOCK PRICE
# ===============================

@app.post("/predict")
def predict(data: dict):

    if df is None or model is None:
        return {"error": "Model or dataset not loaded"}

    company = data.get("company")

    company_data = df[df["company"] == company]

    if company_data.empty:
        return {"error": "Company not found"}

    latest = company_data.iloc[-1]

    input_df = pd.DataFrame([{

        "company_encoded": latest["company_encoded"],

        "open": float(data.get("open")),

        "high": float(data.get("high")),

        "low": float(data.get("low")),

        "close": float(data.get("close")),

        "prev_close": latest["prev_close"],

        "ma_5": latest["ma_5"],

        "ma_10": latest["ma_10"],

        "volatility": float(data.get("high")) - float(data.get("low"))

    }])

    prediction = model.predict(input_df)[0]

    trend = "UP" if prediction > float(data.get("close")) else "DOWN"

    return {
        "company": company,
        "prediction": float(prediction),
        "trend": trend
    }
