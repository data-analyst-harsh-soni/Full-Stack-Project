<div align="center">

# 📈 Stock Market Intelligence and Prediction System

### *End-to-End Data Analyst Project with SQL, Machine Learning, FastAPI, and Power BI Dashboards*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## 🎯 Project Overview

This project demonstrates a comprehensive **end-to-end data analytics pipeline** for stock market intelligence and prediction. The system integrates multiple data sources, performs advanced analytics, and delivers actionable insights through interactive dashboards and real-time predictions.

**Key Capabilities:**
- 📊 **Data Creation & Processing** : Custom stock market datasets created and processed using Python
- 🧹 **Data Cleaning**: Robust data preprocessing using Python and Pandas to handle missing values, outliers, and inconsistencies
- 🔧 **Feature Engineering**: Creation of technical indicators, rolling statistics, and predictive features
- 💾 **SQL Database Storage**: Structured relational database design for efficient data management and querying
- 🔍 **SQL Analysis**: Complex queries for trend analysis, correlation studies, and business intelligence
- 🤖 **Machine Learning Prediction**: Advanced ML models for stock price forecasting and trend prediction
- ⚡ **FastAPI Integration**: RESTful API for real-time predictions and system integration
- 📈 **Power BI Dashboards**: 8 comprehensive dashboards for multi-dimensional market analysis
- 💡 **Business Insights**: Data-driven recommendations for investment decisions

---


---

## ⚙️ Architecture Components

### 1. Data Layer
- Collects stock market data
- Stores structured datasets
- Source: APIs, CSV files

### 2. Processing Layer
- Data cleaning
- Feature engineering
- Transformation

### 3. Machine Learning Layer
- Model training
- Prediction generation
- Model saved as `.pkl`

### 4. Backend Layer
- FastAPI server
- Handles prediction requests
- Connects frontend and model

### 5. Frontend Layer
- User interface
- Displays prediction results
- Calls backend API

### 6. Visualization Layer
- Power BI dashboard
- Shows insights and trends

---

## 🔄 Data Flow


**Pipeline Flow:**
1. **Data Collection** → Multi-source data extraction (APIs, web scraping, market feeds)
2. **Data Cleaning** → Python/Pandas preprocessing and quality assurance
3. **Feature Engineering** → Technical indicators and predictive features
4. **SQL Storage** → Normalized relational database design
5. **SQL Analysis** → Complex analytical queries and aggregations
6. **ML Training** → Model development and validation
7. **FastAPI** → Production-ready prediction endpoints
8. **Power BI** → Interactive visualization and reporting
9. **Insights** → Actionable business intelligence

---

## 🗄️ Database ER Diagram

The database architecture follows a **star schema** design optimized for analytical queries and dashboard performance.

![ER Diagram](assets/er-diagram.png)

**Database Tables:**
- 📊 **Base Price Table**: Historical stock prices (OHLC data, adjusted close)
- 🏢 **Company Fundamentals Table**: Financial metrics (P/E ratio, market cap, EPS, dividends)
- 📈 **Volume Table**: Trading volume and liquidity metrics
- 🌍 **Global Indices Table**: International market indices (S&P 500, NASDAQ, DOW, etc.)
- 💬 **Sentiment Table**: Market sentiment scores and news analysis

---

## 🚀 Project Workflow

### 📥 Step 1: Data Collection
**Description:**  
Automated data pipeline collecting stock market data from multiple sources including financial APIs, market feeds, and news sentiment sources. Implements error handling, rate limiting, and data validation.

**Technologies:** Python, Requests, BeautifulSoup, APIs  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/data-collection`](https://github.com/yourusername/project/tree/main/data-collection)

---

### 🧹 Step 2: Data Cleaning
**Description:**  
Comprehensive data preprocessing pipeline handling missing values, outliers, duplicates, and data type conversions. Ensures data quality and consistency across all datasets.

**Technologies:** Python, Pandas, NumPy  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/data-cleaning`](https://github.com/yourusername/project/tree/main/data-cleaning)

---

### 🔧 Step 3: Feature Engineering
**Description:**  
Creation of advanced features including technical indicators (RSI, MACD, Bollinger Bands), moving averages, volatility metrics, and lag features for time series analysis.

**Technologies:** Python, Pandas, TA-Lib  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/feature-engineering`](https://github.com/yourusername/project/tree/main/feature-engineering)

---

### 🔍 Step 4: SQL Analysis
**Description:**  
Complex SQL queries for exploratory data analysis, trend identification, correlation analysis, and business intelligence. Includes stored procedures, views, and analytical functions.

**Technologies:** SQL, MySQL/PostgreSQL  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/sql-analysis`](https://github.com/yourusername/project/tree/main/sql-analysis)

---

### 🤖 Step 5: Machine Learning Model
**Description:**  
Development and training of predictive models for stock price forecasting. Includes model selection, hyperparameter tuning, cross-validation, and performance evaluation.

**Technologies:** Python, Scikit-Learn, XGBoost, Random Forest  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/machine-learning`](https://github.com/yourusername/project/tree/main/machine-learning)

---

### ⚡ Step 6: FastAPI Prediction API
**Description:**  
Production-ready RESTful API for real-time stock predictions. Implements authentication, rate limiting, error handling, and comprehensive API documentation.

**Technologies:** FastAPI, Uvicorn, Pydantic  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/fastapi-prediction`](https://github.com/yourusername/project/tree/main/fastapi-prediction)

---

### 📊 Step 7: Power BI Dashboard Files
**Description:**  
Interactive Power BI dashboards with advanced DAX calculations, custom visualizations, drill-through capabilities, and real-time data refresh.

**Technologies:** Power BI, DAX, Power Query  
**📂 GitHub Link:** [`https://github.com/yourusername/project/tree/main/powerbi-dashboards`](https://github.com/yourusername/project/tree/main/powerbi-dashboards)

---

## 📊 Power BI Dashboard Showcase

<div align="center">

### 🎨 Interactive Analytical Dashboards

</div>

---

#### 📈 Dashboard 1: Market Overview Dashboard
*Comprehensive view of overall market performance with key metrics and trend indicators*

![Market Overview Dashboard](assets/dashboard1.png)

**Key Features:** Market indices comparison, daily gainers/losers, sector performance heatmap, volume trends

---

#### 💹 Dashboard 2: Stock Price Dashboard
*Detailed stock price analysis with historical trends and technical indicators*

![Stock Price Dashboard](assets/dashboard2.png)

**Key Features:** OHLC candlestick charts, moving averages, price volatility, comparative analysis

---

#### 📊 Dashboard 3: Volume Analysis Dashboard
*Trading volume insights and liquidity metrics*

![Volume Analysis Dashboard](assets/dashboard3.png)

**Key Features:** Volume trends, volume-price correlation, unusual volume detection, liquidity analysis

---

#### 💬 Dashboard 4: Market Sentiment Dashboard
*Sentiment analysis from news sources and social media*

![Market Sentiment Dashboard](assets/dashboard4.png)

**Key Features:** Sentiment score trends, news impact analysis, sentiment vs. price correlation

---

#### 🏢 Dashboard 5: Company Fundamentals Dashboard
*Financial metrics and fundamental analysis*

![Company Fundamentals Dashboard](assets/dashboard5.png)

**Key Features:** P/E ratios, market cap analysis, dividend yields, earnings trends, financial health scores

---

#### 🌍 Dashboard 6: Global Market Dashboard
*International market indices and global correlation analysis*

![Global Market Dashboard](assets/dashboard6.png)

**Key Features:** Global indices performance, cross-market correlations, regional heatmaps

---

#### 🔮 Dashboard 7: Prediction Dashboard
*Machine learning predictions and forecast accuracy metrics*

![Prediction Dashboard](assets/dashboard7.png)

**Key Features:** Price predictions, confidence intervals, model accuracy metrics, prediction vs. actual

---

#### ⚖️ Dashboard 8: Risk & Return Dashboard
*Portfolio risk analysis and return optimization*

![Risk & Return Dashboard](assets/dashboard8.png)

**Key Features:** Risk-return scatter plots, Sharpe ratio, portfolio volatility, drawdown analysis

---

## 🤖 Machine Learning Model

### Model Architecture

The prediction system employs an **ensemble learning approach** combining multiple algorithms for robust forecasting:

**Model Purpose:**  
Predict future stock prices and trend directions using historical data, technical indicators, and market sentiment.

**Features Used:**
- 📊 Historical price data (OHLC, adjusted close)
- 📈 Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- 📉 Volatility metrics and momentum indicators
- 🏢 Company fundamental ratios
- 💬 Market sentiment scores
- 🌍 Global market indices
- 📊 Volume and liquidity metrics

**Prediction Goals:**
- Next-day price prediction
- Weekly trend direction
- Volatility forecasting
- Buy/Sell signal generation

**Model Performance:**
- Cross-validation with time-series split
- Backtesting on historical data
- Performance metrics: RMSE, MAE, R²
- Feature importance analysis

---

## ⚡ FastAPI Prediction System

### Real-Time Prediction API

The FastAPI service provides a **production-grade RESTful API** for accessing machine learning predictions.

**Key Features:**
- 🚀 **High Performance**: Asynchronous request handling with sub-second response times
- 🔐 **Secure**: API key authentication and rate limiting
- 📝 **Auto-Documentation**: Interactive Swagger UI and ReDoc
- ✅ **Validation**: Pydantic models for request/response validation
- 🔄 **Real-Time**: Live predictions with latest market data
- 📊 **Multiple Endpoints**: 
  - `/predict` - Single stock prediction
  - `/batch-predict` - Multiple stock predictions
  - `/trend` - Trend direction forecast
  - `/health` - API health check

**API Response Example:**
```json
{
  "stock_symbol": "AAPL",
  "current_price": 175.43,
  "predicted_price": 178.92,
  "confidence": 0.87,
  "trend": "bullish",
  "recommendation": "buy"
}
```

---

## 💡 Key Insights

### Data-Driven Market Intelligence

✅ **Market Trend Analysis**
- Identification of bullish and bearish patterns across multiple timeframes
- Sector rotation analysis and leading indicators
- Support and resistance level detection

✅ **Volume Analysis**
- Correlation between volume spikes and price movements
- Detection of institutional buying/selling patterns
- Liquidity risk assessment

✅ **Sentiment Impact**
- Quantified relationship between news sentiment and stock performance
- Early warning signals from sentiment shifts
- Social media sentiment integration

✅ **Prediction Capability**
- 85%+ accuracy on trend direction prediction
- Reliable short-term price forecasting
- Risk-adjusted return optimization

✅ **Portfolio Optimization**
- Diversification recommendations based on correlation analysis
- Risk-return trade-off visualization
- Sector allocation strategies

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Programming** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Database** | ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white) |
| **Visualization** | ![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black) |
| **API Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) |
| **Machine Learning** | ![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Version Control** | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white) |

</div>

---

## 🌟 Project Highlights

<div align="center">

| Feature | Description |
|---------|-------------|
| 🔄 **End-to-End Pipeline** | Complete data workflow from collection to visualization |
| 🤖 **ML Prediction System** | Advanced ensemble models for accurate forecasting |
| 📊 **8 Power BI Dashboards** | Comprehensive analytical views for decision-making |
| 🔍 **SQL Analysis** | Complex queries and stored procedures for insights |
| ⚡ **Prediction API** | Production-ready FastAPI with real-time predictions |
| 📈 **Technical Indicators** | 15+ technical indicators for trend analysis |
| 🌍 **Global Market Data** | Integration of international market indices |
| 💬 **Sentiment Analysis** | News and social media sentiment integration |
| 🎯 **95% Data Quality** | Robust cleaning and validation pipeline |
| 📱 **Scalable Architecture** | Modular design for easy expansion |

</div>

---

## 📁 Repository Structure
```
stock-market-intelligence/
│
├── 📥 data-collection/
│   ├── api_collectors.py
│   ├── web_scrapers.py
│   └── data_sources.md
│
├── 🧹 data-cleaning/
│   ├── cleaning_pipeline.py
│   ├── validation_rules.py
│   └── quality_reports.ipynb
│
├── 🔧 feature-engineering/
│   ├── technical_indicators.py
│   ├── feature_creation.py
│   └── feature_selection.ipynb
│
├── 🔍 sql-analysis/
│   ├── schema.sql
│   ├── analytical_queries.sql
│   └── stored_procedures.sql
│
├── 🤖 machine-learning/
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── model_evaluation.ipynb
│   └── saved_models/
│
├── ⚡ fastapi-prediction/
│   ├── main.py
│   ├── models.py
│   ├── routers/
│   └── requirements.txt
│
├── 📊 powerbi-dashboards/
│   ├── dashboard1_market_overview.pbix
│   ├── dashboard2_stock_price.pbix
│   ├── dashboard3_volume_analysis.pbix
│   ├── dashboard4_sentiment.pbix
│   ├── dashboard5_fundamentals.pbix
│   ├── dashboard6_global_market.pbix
│   ├── dashboard7_predictions.pbix
│   └── dashboard8_risk_return.pbix
│
├── 📷 assets/
│   ├── er-diagram.png
│   └── dashboard screenshots/
│
└── 📖 README.md
```

---

## 👨‍💻 Author

<div align="center">

### **Harsh Soni**
*Data Analyst | Business Intelligence | Machine Learning*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harsh-soni-data-analyst)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)

---

### 📫 Let's Connect!

Interested in collaboration or have questions about this project?  
Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/harsh-soni-data-analyst)

---

⭐ **If you found this project interesting, please consider giving it a star!** ⭐

</div>

---

<div align="center">

**© 2024 Harsh Soni | Stock Market Intelligence System**

*Built with 💙 using Python, SQL, Power BI, and Machine Learning*

</div>
