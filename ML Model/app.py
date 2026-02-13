# ===============================
# FASTAPI STOCK PREDICTION API
# ===============================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# ===============================
# PATH SETUP (IMPORTANT FIX)
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
# ENABLE CORS
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

    print("✅ Model and Encoder Loaded")

except Exception as e:
    print("❌ Model load error:", str(e))


# ===============================
# LOAD DATASET
# ===============================

df = None

try:

    df = pd.read_csv(DATA_PATH)

    # Clean company column
    df["company"] = (
        df["company"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Convert numeric safely
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix date
    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    df = df.sort_values(
        ["company", "trade_date"]
    )

    # ===============================
    # FEATURE ENGINEERING
    # ===============================

    df["company_encoded"] = df["company"].apply(
        lambda x:
        encoder.transform([x])[0]
        if x in encoder.classes_
        else None
    )

    df.dropna(subset=["company_encoded"], inplace=True)

    df["prev_close"] = (
        df.groupby("company")["close"]
        .shift(1)
    )

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

    print("✅ Dataset Loaded Successfully")

except Exception as e:

    print("❌ Dataset load error:", str(e))


# ===============================
# HOME ROUTE
# ===============================

@app.get("/")
def home():

    return {
        "status": "running",
        "model_loaded": model is not None,
        "dataset_loaded": df is not None
    }


# ===============================
# GET COMPANIES
# ===============================

@app.get("/companies")
def get_companies():

    if df is None:
        return {"error": "Dataset not loaded"}

    return sorted(
        df["company"].unique().tolist()
    )


# ===============================
# GET LATEST PRICE
# ===============================

@app.get("/latest/{company}")
def latest_price(company: str):

    if df is None:
        return {"error": "Dataset not loaded"}

    company = company.strip().upper()

    company_data = df[
        df["company"] == company
    ]

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
# PREDICT
# ===============================

@app.post("/predict")
def predict(data: dict):

    if model is None or df is None:

        return {
            "error":
            "Model or dataset not loaded"
        }

    company = (
        data.get("company", "")
        .strip()
        .upper()
    )

    if company not in encoder.classes_:

        return {
            "error":
            "Company not supported"
        }

    company_data = df[
        df["company"] == company
    ]

    latest = company_data.iloc[-1]

    open_price = float(data.get("open"))
    high_price = float(data.get("high"))
    low_price = float(data.get("low"))
    close_price = float(data.get("close"))

    input_df = pd.DataFrame([{

        "company_encoded":
        latest["company_encoded"],

        "open":
        open_price,

        "high":
        high_price,

        "low":
        low_price,

        "close":
        close_price,

        "prev_close":
        latest["prev_close"],

        "ma_5":
        latest["ma_5"],

        "ma_10":
        latest["ma_10"],

        "volatility":
        high_price - low_price

    }])

    prediction = float(
        model.predict(input_df)[0]
    )

    trend = (
        "UP"
        if prediction > close_price
        else "DOWN"
    )

    return {

        "company": company,

        "prediction":
        round(prediction, 2),

        "trend":
        trend
    }
