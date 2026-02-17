<div align="center">

# 📈 Stock Market Intelligence & Prediction System

### *End-to-End Data Analyst Project using Python, SQL, Power BI, Machine Learning, and FastAPI*

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-CC2927?style=for-the-badge&logo=microsoft-sql-server&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

<br>

![Project Banner](https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=120&section=header&text=Stock%20Market%20Intelligence&fontSize=32&fontColor=e94560&animation=fadeIn&fontAlignY=65)

</div>

---

## 🌟 Project Overview

This is a **fully end-to-end, production-grade Data Analytics & Machine Learning project** built on the Indian Stock Market (NSE). Unlike typical projects that rely on pre-collected external datasets, this project **generates its own realistic stock market dataset using Python**, then takes it through a complete analytical pipeline — from raw data creation to interactive web-based predictions.

| Stage | Technology | Description |
|-------|-----------|-------------|
| 🏗️ Dataset Creation | Python | Synthetically generated realistic NSE stock market data |
| 🧹 Data Cleaning | Python / Jupyter | 8 specialized cleaning notebooks for each data domain |
| ⚙️ Feature Engineering | Python | Derived financial metrics and predictive features |
| 🗄️ SQL Analysis | SQL Server | Data import, Data Mart creation, 20+ analytical queries |
| 📊 Visualization | Power BI | 8 interactive dashboards with cross-filter intelligence |
| 🤖 ML Model | Scikit-Learn | Trained prediction model serialized as `.pkl` |
| ⚡ API | FastAPI | RESTful prediction endpoint for real-time inference |
| 🌐 Frontend | HTML/CSS/JS | Interactive web interface to display stock predictions |

---

## 🔄 Complete Project Workflow

```mermaid
flowchart TD
    A([🐍 Python Dataset Creation]) --> B

    B([🗂️ Raw Dataset Storage])

    B --> C([🧹 Data Cleaning ])

    C --> D([⚙️ Feature Engineering])

    D --> E([📁 Clean Dataset Storage])

    E --> F([🗄️ SQL Database Import])

    F --> G([🧱 Data Mart Creation])

    G --> H([🔍 SQL Analysis — 20+ Queries])

    H --> I([🗺️ ER Diagram Creation])

    E --> J([🤖 ML Model Training])

    J --> K([💾 Model Serialization])

    K --> L([⚡ FastAPI Prediction ])

    L --> M([🌐 Frontend Application])

    G --> N([📊 Power BI Integration])

    N --> O([💡 Business Insights])

    style A fill:#1e3a5f,stroke:#4fc3f7,color:#fff
    style E fill:#1b4332,stroke:#52b788,color:#fff
    style J fill:#4a1942,stroke:#da77f2,color:#fff
    style N fill:#7c2d12,stroke:#fb923c,color:#fff
    style M fill:#1e3a5f,stroke:#4fc3f7,color:#fff
    style O fill:#7c2d12,stroke:#fb923c,color:#fff
```

---

## 🗺️ ER Diagram

The data model is built on **5 core entity tables** connected through a clean relational schema, forming the backbone of both the SQL analysis layer and Power BI dashboards.

<div align="center">

![ER Diagram](E-R%20Diagram.png)

</div>

### 🔗 Entity Relationships

| Entity | Primary Key | Connects To | Relationship |
|--------|------------|-------------|--------------|
| **NSE Prices** (base_price) | `symbol + date` | Company Fundamentals | Many-to-One |
| **Company Fundamentals** | `symbol` | NSE Prices, Volumes | One-to-Many |
| **Daily Sentiment** | `symbol + date` | NSE Prices | Many-to-One |
| **Global Indices** | `date` | NSE Prices | One-to-Many |
| **Volumes** | `symbol + date` | NSE Prices | One-to-One |
| **Macro Data** | `date` | NSE Prices | One-to-Many |

> 📌 **Design Logic:** The `NSE Prices` table acts as the central **fact table**, with company, sentiment, volume, macro, and global data as **dimension tables** — a classic star schema optimized for analytical queries.

---

## 📊 Dashboard Showcase

> **8 Power BI dashboards** built from SQL-connected live data, enabling cross-filtered intelligence across market segments.

<div align="center">

<h2>🏠 Market Overview Dashboard</h2>

<img src="Dashboard/Market Overview.png" width="900">

<br><br>

<h2>🏢 Company Fundamentals Dashboard</h2>

<img src="Dashboard/Company Fundamentals.png" width="900">

<br><br>

<h2>📈 Stock Price Trend Dashboard</h2>

<img src="Dashboard/Stock Price Trend.png" width="900">

<br><br>

<h2>📦 Volume Analysis Dashboard</h2>

<img src="Dashboard/Volume Analysis Dashboard.png" width="900">

<br><br>

<h2>🧠 Market Sentiment Intelligence Dashboard</h2>

<img src="Dashboard/Market Sentiment Intelligence Dashboard.png" width="900">

<br><br>

<h2>🌍 Global Market Intelligence Dashboard</h2>

<img src="Dashboard/Global Market Intelligence Dashboard.png" width="900">

<br><br>

<h2>🇮🇳 Indian Stock Market Intelligence & Analytics</h2>

<img src="Dashboard/Indian Stock Market Intelligence & Analytics.png" width="900">

</div>


## 🚀 Step-by-Step Project Workflow

### Step 1 — 🏗️ Dataset Creation Using Python
**📁 Folder:** `CREATION_OF_DATASET_USING_PYTHON/`

The dataset is **not collected from any external source**. It was entirely **synthesized using Python** to simulate real NSE stock market behavior — including OHLCV prices, company fundamentals, macro indicators, trading volumes, and news sentiment signals.

```
CREATION_OF_DATASET_USING_PYTHON/
└── Stock_Market_Prediction.py       ← Master dataset generation script
```

---

### Step 2 — 🗂️ Raw Dataset Storage
**📁 Folder:** `stock_market_unclean_dataset/`

The generated raw data is organized into **6 domain-specific subdirectories**, mimicking real-world data lake structure.

```
stock_market_unclean_dataset/
├── raw_data/          ← Core OHLCV price data
├── company_data/      ← Fundamentals (P/E, EPS, Market Cap, etc.)
├── macro_data/        ← Inflation, interest rates, GDP data
├── trading_data/      ← Intraday trading volumes
├── news_sentiment/    ← Sentiment scores per stock per day
└── metadata/          ← Ticker symbols, sector mappings
```

---

### Step 3 — 🧹 Data Cleaning Using Python
**📁 Folder:** `Cleaning_Code/`

**8 dedicated Jupyter notebooks** handle cleaning for each data domain separately — ensuring precision without cross-contamination.

```
Cleaning_Code/
├── basa_price_cleaning.ipynb         ← Base price OHLCV cleaning
├── nse_price_cleaning.ipynb          ← NSE-specific price normalization
├── company_data_cleaning.ipynb       ← Fundamentals cleaning & outlier handling
├── macro_data_cleaning.ipynb         ← Macro indicator smoothing
├── meta_data_cleaning.ipynb          ← Symbol & metadata standardization
├── news_sentiment.ipynb              ← Sentiment score normalization
├── global_indices_cleaning.ipynb     ← International index alignment
└── volume_cleaning.ipynb             ← Volume anomaly detection & fixing
```

> Each notebook handles: null treatment, type casting, outlier handling, date normalization, and domain-specific business rules.

---

### Step 4 — ⚙️ Feature Engineering
Performed **within the cleaning notebooks**, adding derived columns like:
- 📐 Moving averages (5-day, 20-day, 50-day)
- 📉 Daily returns & volatility scores
- 📊 RSI, MACD signals
- 💬 Sentiment rolling averages
- 🌏 Global index correlation features
- 🏭 Sector-based normalization

---

### Step 5 — 📁 Clean Dataset Storage
**📁 Folder:** `stock_market_clean_dataset_with_Feature_Eng/`

Post-cleaning, all data consolidates into **7 structured CSV files**, ready for SQL ingestion and ML training.

```
stock_market_clean_dataset_with_Feature_Eng/
├── base_price.csv               ← Clean OHLCV base prices
├── nse_prices.csv               ← NSE-specific cleaned prices
├── company_fundamentals.csv     ← P/E, EPS, Market Cap, Book Value
├── daily_sentiment.csv          ← Daily sentiment scores per symbol
├── global_indices.csv           ← Global market index data
├── inflation_interest.csv       ← Macro economic indicators
└── volumes.csv                  ← Daily trading volumes
```

---

### Step 6 — 🗄️ SQL Analysis
**📁 Folder:** `SQL/`

```
SQL/
├── DATA_IMPORTING_CODE.sql       ← Bulk imports all 7 CSVs into SQL tables
├── SQL Data Mart.sql             ← Creates star schema Data Mart
└── QUESTION_WITH_SOLUTION.sql    ← 20+ business analytical queries
```

**Highlights:**
- ✅ All 7 clean CSVs imported into SQL relational database
- ✅ Data Mart designed with fact + dimension table architecture
- ✅ 20+ real-world analytical queries covering sector performance, volatility ranking, sentiment impact, volume anomalies, macro correlation, and more

---

### Step 7 — 🤖 Machine Learning Model
**📁 Folder:** `ML Model/`

```
ML Model/
├── train_model.py         ← Model training pipeline (Scikit-Learn)
├── predict.py             ← Standalone prediction logic
├── app.py                 ← FastAPI application server
├── stock_model.pkl        ← Trained ML model (serialized)
└── company_encoder.pkl    ← Label encoder for company symbols
```

---

### Step 8 — 🌐 Frontend Application
**📁 Folder:** `frontend/`

```
frontend/
├── index.html     ← Main prediction UI
├── script.js      ← API call logic & result rendering
└── style.css      ← Styling & responsive layout
```

---

### Step 9 — 📊 Power BI Dashboards
**📁 Folder:** `Dashboard/`

```
Dashboard/
├── Market Overview.png
├── Company Fundamentals.png
├── Stock Price Trend.png
├── Volume Analysis Dashboard.png
├── Market Sentiment Intelligence Dashboard.png
├── Global Market Intelligence Dashboard.png
└── Indian Stock Market Intelligence & Analytics.png
```

**Power BI File:** `Stock Market Prediction.pbix`

---

## 🗄️ SQL Analysis Deep Dive

The SQL layer is the analytical engine of this project, enabling structured business intelligence over the cleaned dataset.

### 📥 Data Import Strategy
All 7 cleaned CSV files are loaded into SQL Server using `DATA_IMPORTING_CODE.sql`, creating properly typed relational tables with primary and foreign key constraints.

### 🏗️ Data Mart Architecture
`SQL Data Mart.sql` builds a **star schema** with:
- **Fact Table:** `fact_stock_prices` — daily price + volume records
- **Dimension Tables:** `dim_company`, `dim_date`, `dim_sentiment`, `dim_macro`, `dim_global`

### 🔍 20+ Analytical SQL Queries — Sample Topics

| # | Query Topic |
|---|-------------|
| 1 | Top 10 stocks by average daily return |
| 2 | Sector-wise performance comparison |
| 3 | Stocks with highest volatility (std dev of returns) |
| 4 | Sentiment vs. price movement correlation |
| 5 | Volume anomaly detection (>2x average) |
| 6 | 52-week high/low breaches |
| 7 | Macro interest rate vs. market index movement |
| 8 | Most consistent performers (low drawdown) |
| 9 | Global index impact on NSE movement |
| 10 | Month-over-month growth ranking |
| 11–20+ | Moving average crossovers, EPS vs stock return, sentiment heatmaps, etc. |

---

## 🤖 Machine Learning Model

### 🎯 Model Objective
Predict the **next-day closing price** (or directional movement) of NSE-listed stocks based on historical patterns, macro signals, and sentiment features.

### 🧬 Feature Set Used
- Historical OHLCV data (lagged features)
- Rolling moving averages (5/20/50-day)
- RSI & MACD technical indicators
- Daily sentiment scores (rolling 3-day)
- Macro: inflation rate, interest rate
- Global index returns (S&P 500, DAX, Nikkei proxies)

### 🛠️ Pipeline
```
Clean CSV Data
     ↓
Feature Matrix Construction (train_model.py)
     ↓
Train/Test Split (80/20)
     ↓
Scikit-Learn Model Training
     ↓
Evaluation (MAE, RMSE, R²)
     ↓
Serialization → stock_model.pkl + company_encoder.pkl
     ↓
FastAPI Integration (app.py)
     ↓
Frontend API Calls (script.js)
```

### ⚡ FastAPI Prediction Endpoint
```python
POST /predict
{
  "symbol": "RELIANCE",
  "open": 2450.0,
  "high": 2490.0,
  "low": 2430.0,
  "volume": 1200000,
  "sentiment": 0.65,
  "rsi": 58.2
}

→ Response: { "predicted_close": 2478.5, "confidence": 0.87 }
```

---

## 📊 Power BI Integration

### 🔌 SQL → Power BI Connection
- Power BI Desktop connected directly to **SQL Server** using native connector
- **DirectQuery / Import mode** for live or cached data
- Relationships established mirroring the SQL Data Mart star schema

### 📋 Dashboards Built

| Dashboard | Key Metrics |
|-----------|-------------|
| 🏠 Market Overview | Index movement, daily gainers/losers, market breadth |
| 🏢 Company Fundamentals | P/E ratio, EPS, Market Cap, Book Value trends |
| 📈 Stock Price Trend | OHLCV candlesticks, moving averages, YTD performance |
| 📦 Volume Analysis | Volume spikes, liquidity heatmap, delivery % |
| 🧠 Market Sentiment | Sentiment scores, news impact, bullish/bearish gauge |
| 🌍 Global Market Intelligence | S&P 500 / DAX / Nikkei correlation with NSE |
| 🇮🇳 Indian Market Analytics | Sector rotation, index composition, FII/DII flows |

> **Business Value:** The dashboards enable data-driven decisions by revealing hidden patterns in market sentiment, macro-economic impact, and sector rotation behavior.

---

## 🗂️ Repository Structure

```
Stock-Market-Prediction/
│
├── 📂 CREATION_OF_DATASET_USING_PYTHON/
│   └── Stock_Market_Prediction.py        ← Dataset generation script
│
├── 📂 stock_market_unclean_dataset/
│   ├── raw_data/                         ← Core OHLCV raw data
│   ├── company_data/                     ← Company fundamentals
│   ├── macro_data/                       ← Macro economic data
│   ├── trading_data/                     ← Trading volume data
│   ├── news_sentiment/                   ← News sentiment data
│   └── metadata/                         ← Ticker metadata
│
├── 📂 Cleaning_Code/
│   ├── basa_price_cleaning.ipynb
│   ├── company_data_cleaning.ipynb
│   ├── global_indices_cleaning.ipynb
│   ├── macro_data_cleaning.ipynb
│   ├── meta_data_cleaning.ipynb
│   ├── news_sentiment.ipynb
│   ├── nse_price_cleaning.ipynb
│   └── volume_cleaning.ipynb
│
├── 📂 stock_market_clean_dataset_with_Feature_Eng/
│   ├── base_price.csv
│   ├── company_fundamentals.csv
│   ├── daily_sentiment.csv
│   ├── global_indices.csv
│   ├── inflation_interest.csv
│   ├── nse_prices.csv
│   └── volumes.csv
│
├── 📂 SQL/
│   ├── DATA_IMPORTING_CODE.sql           ← CSV import scripts
│   ├── SQL Data Mart.sql                 ← Star schema creation
│   └── QUESTION_WITH_SOLUTION.sql        ← 20+ analytical queries
│
├── 📂 ML Model/
│   ├── train_model.py                    ← ML training pipeline
│   ├── predict.py                        ← Prediction logic
│   ├── app.py                            ← FastAPI server
│   ├── stock_model.pkl                   ← Serialized ML model
│   └── company_encoder.pkl               ← Label encoder
│
├── 📂 frontend/
│   ├── index.html                        ← Web interface
│   ├── script.js                         ← API integration
│   └── style.css                         ← Styling
│
├── 📂 Dashboard/
│   ├── Market Overview.png
│   ├── Company Fundamentals.png
│   ├── Stock Price Trend.png
│   ├── Volume Analysis Dashboard.png
│   ├── Market Sentiment Intelligence Dashboard.png
│   ├── Global Market Intelligence Dashboard.png
│   └── Indian Stock Market Intelligence & Analytics.png
│
├── 🗺️ E-R Diagram.png                   ← Entity-Relationship Diagram
├── 📊 Stock Market Prediction.pbix       ← Power BI report file
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Dataset creation, cleaning, ML |
| **Data Manipulation** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Data transformation & feature engineering |
| **Machine Learning** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) | Model training & prediction |
| **API Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | Prediction REST API |
| **Database** | ![SQL Server](https://img.shields.io/badge/SQL_Server-CC2927?style=flat&logo=microsoftsqlserver&logoColor=white) | Data storage & analytical queries |
| **BI & Visualization** | ![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black) | Interactive dashboards |
| **Notebooks** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Data cleaning & EDA |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) | Web prediction interface |
| **Model Serialization** | Pickle (`.pkl`) | Save & load trained models |

---

## 👤 Author

<div align="center">

<img src="https://avatars.githubusercontent.com/u/placeholder?v=4" width="100" style="border-radius:50%" alt="Harsh Soni"/>

### Harsh Soni
**Data Analyst**

*Passionate about transforming raw data into actionable intelligence through end-to-end analytical pipelines.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harsh-soni-data-analyst)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)

</div>

---

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:1a1a2e&height=100&section=footer)

**⭐ If you found this project helpful, please consider giving it a star!**

*Built with ❤️ by Harsh Soni — Data never lies, but it needs the right analyst to speak.*

</div>
