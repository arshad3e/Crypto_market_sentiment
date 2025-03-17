# Market Sentiment Analyzer

## Overview
This Python script analyzes the current market sentiment for cryptocurrency trading pairs on Coinbase Advanced Trade. It uses technical indicators (RSI, MACD, ADX, and price change) to assess whether the market is **Bullish**, **Bearish**, or **Neutral** as of the current timestamp (e.g., March 16, 2025). The script aggregates sentiment across multiple volatile coins to provide an overall market outlook.

## Features
- **Real-Time Analysis**: Fetches 1-minute OHLCV data for up-to-date sentiment.
- **Multi-Coin Aggregation**: Analyzes a predefined list of volatile coins (e.g., DOGE/USD, BTC/USD).
- **Technical Indicators**:
  - Price change over a 10-minute lookback.
  - Relative Strength Index (RSI).
  - Moving Average Convergence Divergence (MACD).
  - Average Directional Index (ADX) for trend strength.
- **Sentiment Scoring**: Combines indicators into a weighted score per coin, then determines overall market sentiment via majority vote with a price change tiebreaker.
- **Logging**: Detailed logs for debugging and tracking.

## Prerequisites
- **Python**: 3.7+
- **Libraries**:
  - `ccxt`: For interacting with Coinbase Advanced Trade API.
  - `pandas`: For data manipulation.
  - `ta`: For technical analysis indicators.
  - `tenacity`: For retry logic on API calls.
  - Install via pip:
    ```bash
    pip install ccxt pandas ta tenacity
