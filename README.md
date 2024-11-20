# Stock Market Sentiment Analysis and Predictive Modeling

This repository contains a pipeline to analyze the relationship between stock market sentiment and price movements, combining news sentiment analysis with stock market data to predict trading signals and evaluate strategies.

## Project Overview

This project leverages APIs, machine learning models, and data processing techniques to create a comprehensive framework for financial market analysis. The key components of the pipeline are:

- **Data Collection**: Gathering stock price and news data using financial APIs.  
- **Sentiment Analysis**: Evaluating the sentiment of news articles using the FinBERT model.  
- **Feature Engineering**: Creating predictive features based on historical price data and sentiment.  
- **Modeling and Backtesting**: Building machine learning models to predict buy/sell/hold signals and backtesting trading strategies.

## Run Order:

### 1. `ticker_data_generation.py`
- **Purpose**: Generate stock price data and label it for predictive modeling.
- **Description**:  
  - Fetch historical stock price data using the Yahoo Finance API.  
  - Generate labels using the triple barrier method.  
  - Compute moving window metrics as predictive features.  
  - Save the labeled data to a CSV file for subsequent use.  

---

### 2. `news_data_generation.py`
- **Purpose**: Collect news articles related to the stock of interest.
- **Description**:  
  - Fetch news articles via the Seeking Alpha API for a given stock and date range.  
  - Combine and structure API responses into a single DataFrame.  
  - Export the resulting DataFrame to a CSV file.  

---

### 3. `get_news_sentiment.py`
- **Purpose**: Perform sentiment analysis on the news dataset.
- **Description**:  
  - Use the FinBERT model to evaluate sentiment probabilities (positive, neutral, negative) for each article.  
  - Annotate each article with the sentiment label having the highest probability.  
  - Save the sentiment-enriched DataFrame to a new CSV file.  

---

### 4. `aggregate_ticker_and_news_data.py`
- **Purpose**: Merge and prepare ticker and sentiment data for modeling.
- **Description**:  
  - Load stock price and news sentiment data from CSV files.  
  - Clip data to the overlapping date range.  
  - Aggregate daily sentiment data and compute a sentiment score (-100 to 100).  
  - Merge sentiment scores with stock ticker data using the date as an index.  
  - Save the combined dataset to a new CSV file.  

---

### 5. `model_and_backtest.py`
- **Purpose**: Train predictive models and evaluate trading strategies.
- **Description**:  
  - Prepare the merged dataset with technical indicators and sentiment features.  
  - Train a Random Forest model to predict buy/sell/hold signals.  
  - Adjust predictions based on sentiment thresholds.  
  - Backtest the model's performance, calculating cumulative returns, Sharpe ratio, and maximum drawdown.  
  - Visualize strategy performance against a buy-and-hold baseline.  

---

## Requirements

- Python (version 3.12 used)
- Pandas
- Scikit-learn
- Yahoo Finance API library (`yfinance`)
- Seeking Alpha API access (aquired via RapidAPI)
- FinBERT library for sentiment analysis (`transformers`)
- Torch
- Scipy
- Matplotlib/Plotly for visualization
- Numpy
- Requests, OS, Time
- Builtins, Collections

---

## Setup and Installation
- Install required dependencies
- Run file `1`
- Obtain API key for Seeking Alpha from RapidAPI, update enviroment varaibles
- Run file `2`
- Run file `3`
- Run file `4`
- Play around with thresholds, running file `5`

---

## Dataset Information

- **Stock Ticker Data**: Historical price data, features, and labels.  
- **News Data**: Articles and sentiment analysis results.  
- **Aggregated Dataset**: Merged stock and news data, with sentiment scores as features.  

---

## Results and Analysis
- The backtesting takes place over a almost 2 year span of Intel from 2021-11-03 to 2024-10-31. During this time, a buy and hold approach (buying at the start date of the backtest) would provide a -57.3% return.
- In the below results, you will see that our base model obtains -11.3% (2.0% with no trading costs) returns. We see a growth in returns obtained when thresholding increases from 5 -> 10.
- At -5,5 we obtain -3.6% (37.5% with no trading fee) returns, a notable increase from the base model.
-   Because of our frequent trades, our fees are limiting our room for the gains we see.
- When thresholding is at -10, 10, we see the best performance of our model, getting 4.65% returns (47.5% without trading fees). As we continue to increase our threshold, our trading fees continue to decrease, however, our gains become worse.
- When thresholding exceeds 95, there is no predicted signals swapped from the sentiment, so our adjusted model performs the same as our base model.

- **Base Model Backtest Performance**:
  ![BaseModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_base.png)
  ![BaseModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_base.png)
   
- **-5,5 Thresholding Backtest Performance**:
  ![5ModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_5.png)
  ![5ModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_5.png)

- **-10,10 Thresholding Backtest Performance**:
  ![5ModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_10.png)
  ![5ModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_10.png)

- **-20,20 Thresholding Backtest Performance**:
  ![5ModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_20.png)
  ![5ModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_20.png)

- **-40,40 Thresholding Backtest Performance**:
  ![5ModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_40.png)
  ![5ModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_40.png)

- **-80,80 Thresholding Backtest Performance**:
  ![5ModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_80.png)
  ![5ModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_80.png)

- **-90,90 Thresholding Backtest Performance**:
  ![5ModelBacktest](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/backtest_90.png)
  ![5ModelTrading](https://github.com/k4404c/setiment_based_stock_pred/blob/main/results/tradingsignals_90.png)
---

## Future Work

A significant limitation of this model is that it calculates returns based on the price movement from today's close to tomorrow's close. This framework is designed to prevent misuse of data that would be unavailable given current knowledge.

In simpler terms, if we predict a signal of 1 today, the investment would only be made at today’s close and, if necessary, liquidated at tomorrow’s close. This approach may lead to delayed investments, particularly in scenarios involving news articles, where price fluctuations often occur within short time spans intraday rather than across daily closes.

To address this issue, we could retain hour/minute timestamps when collecting news data instead of discarding them, as was previously done. Additionally, acquiring intraday data for the relevant ticker could enhance the model's performance by capturing finer granularity.

While this method could improve the model’s accuracy, it also introduces additional complexity, which is a critical consideration.

---

## Contributors

Project done in collaboration with Alice Zhu for CSE472 under Professor Huan Liu at Arizona State University

---
