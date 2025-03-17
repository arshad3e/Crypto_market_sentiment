Market Sentiment Analyzer
Overview
This Python script analyzes the current market sentiment for cryptocurrency trading pairs on Coinbase Advanced Trade. It uses technical indicators (RSI, MACD, ADX, and price change) to assess whether the market is Bullish, Bearish, or Neutral as of the current timestamp (e.g., March 16, 2025). The script aggregates sentiment across multiple volatile coins to provide an overall market outlook.

Features
Real-Time Analysis: Fetches 1-minute OHLCV data for up-to-date sentiment.
Multi-Coin Aggregation: Analyzes a predefined list of volatile coins (e.g., DOGE/USD, BTC/USD).
Technical Indicators:
Price change over a 10-minute lookback.
Relative Strength Index (RSI).
Moving Average Convergence Divergence (MACD).
Average Directional Index (ADX) for trend strength.
Sentiment Scoring: Combines indicators into a weighted score per coin, then determines overall market sentiment via majority vote with a price change tiebreaker.
Logging: Detailed logs for debugging and tracking.
Prerequisites
Python: 3.7+
Libraries:
ccxt: For interacting with Coinbase Advanced Trade API.
pandas: For data manipulation.
ta: For technical analysis indicators.
tenacity: For retry logic on API calls.
Install via pip:
bash

Collapse

Wrap

Copy
pip install ccxt pandas ta tenacity
Coinbase Advanced Trade API Keys:
Create a JSON file (e.g., coinbase.json) with:
json

Collapse

Wrap

Copy
{
  "name": "your_api_key_name",
  "privateKey": "your_private_key",
  "passphrase": "your_passphrase"
}
Update the json_file_path in the code to point to this file.
Usage
Clone the Repository:
bash

Collapse

Wrap

Copy
git clone <repository_url>
cd <repository_directory>
Set Up API Keys:
Place your coinbase.json file in the specified directory (default: /Users/arshadshaik/Desktop/python_practice/momentum_trading/).
Run the Script:
bash

Collapse

Wrap

Copy
python market_sentiment_analyzer.py
Output:
Console: Prints the overall market sentiment (e.g., "The market sentiment is Bullish as of 2025-03-16 10:00:02...").
Log File: Detailed per-symbol analysis and summary in log.txt.
Code Structure
fetch_data(symbol, timeframe, limit): Fetches OHLCV data from Coinbase with retry logic for rate limits or errors.
analyze_symbol_sentiment(df, symbol): Computes sentiment for a single coin using RSI, MACD, ADX, and price change.
read_market_sentiment(): Aggregates sentiment across volatile coins and determines the overall market outlook.
Main Execution: Runs the sentiment analysis and prints the result.
Configuration
Volatile Coins: Edit volatile_coins list to include/exclude trading pairs (default: DOGE/USD, SHIB/USD, XRP/USD, SOL/USD, BTC/USD, ETH/USD).
Timeframe: Default is 1m (1-minute candles); adjust in timeframe variable.
Lookback: Price change calculated over 10 periods (lookback = 10); modify in analyze_symbol_sentiment.
Thresholds:
RSI: > 70 (Bullish), < 30 (Bearish).
Price Change: > 0.5% (Bullish), < -0.5% (Bearish).
ADX: > 25 amplifies sentiment score.
Adjust these in analyze_symbol_sentiment for sensitivity.
Example Output
text

Collapse

Wrap

Copy
2025-03-16 10:00:00,123 - INFO - DOGE/USD: Sentiment: Bullish | Price Change: 1.20% | RSI: 72.50 | MACD Diff: 0.0005 | ADX: 28.30
2025-03-16 10:00:00,456 - INFO - SHIB/USD: Sentiment: Neutral | Price Change: 0.10% | RSI: 45.20 | MACD Diff: -0.0001 | ADX: 15.60
[...]
2025-03-16 10:00:02,000 - INFO - The market sentiment is Bullish as of 2025-03-16 10:00:02. Breakdown: Bullish: 3, Bearish: 1, Neutral: 2. Avg Price Change: 0.23%
Console:

text

Collapse

Wrap

Copy
The market sentiment is Bullish as of 2025-03-16 10:00:02. Breakdown: Bullish: 3, Bearish: 1, Neutral: 2. Avg Price Change: 0.23%
Limitations
Data Dependency: Requires sufficient historical data (at least 50 periods) per symbol.
API Rate Limits: Handles via retries, but excessive calls may delay results.
Market Scope: Limited to specified volatile coins; broader sentiment may differ.
Customization
Add Coins: Expand volatile_coins for a wider market view.
Adjust Sensitivity: Modify RSI, price change, or ADX thresholds in analyze_symbol_sentiment.
Integrate: Combine with a trading bot by calling read_market_sentiment() in your main() function.
Troubleshooting
API Errors: Check log.txt for rate limit or authentication issues; ensure coinbase.json is correct.
No Output: Verify internet connection and API key validity.
Inconsistent Sentiment: Adjust lookback or thresholds if results seem off.

