import ccxt
import pandas as pd
import ta
import time
import json
import logging
import os
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError

# Set up logging (minimal, for errors and sentiment reasoning)
log_file = os.path.join(os.path.dirname(__file__), 'log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

# Trading parameters
timeframe = '1m'
limit = 100
volatile_coins = ['DOGE/USD', 'SHIB/USD', 'XRP/USD', 'SOL/USD', 'BTC/USD', 'ETH/USD']
REQUESTS_PER_MINUTE = 200
SLEEP_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# Load API keys
def load_keys(file_path):
    with open(file_path, 'r') as f:
        keys = json.load(f)
    return keys['name'], keys['privateKey'], keys.get('passphrase', '')

json_file_path = '/Users/arshadshaik/Desktop/python_practice/momentum_trading/coinbase.json'
api_key, secret, passphrase = load_keys(json_file_path)

# Initialize exchange
exchange = ccxt.coinbaseadvanced({
    'apiKey': api_key,
    'secret': secret,
    'passphrase': passphrase,
    'enableRateLimit': True,
})

# Fetch OHLCV data with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(HTTPError)
)
def fetch_data(symbol, timeframe='1m', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.RateLimitExceeded:
        logger.error(f"Rate limit exceeded for {symbol}")
        raise HTTPError("429 Too Many Requests")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

# Analyze sentiment for a single symbol
def analyze_symbol_sentiment(df, symbol):
    if df is None or len(df) < 50:
        return None
    
    # Technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'], window_slow=13, window_fast=6, window_sign=5)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    lookback = 10
    
    # Price change over last 10 periods
    price_change_pct = ((latest['close'] - df.iloc[-lookback]['close']) / df.iloc[-lookback]['close']) * 100 if len(df) >= lookback else 0
    
    # Sentiment scoring
    sentiment_score = 0
    
    # RSI
    if latest['rsi'] > 70:
        sentiment_score += 1
    elif latest['rsi'] < 30:
        sentiment_score -= 1
    
    # MACD crossover
    if latest['macd_diff'] > 0 and prev['macd_diff'] <= 0:
        sentiment_score += 1
    elif latest['macd_diff'] < 0 and prev['macd_diff'] >= 0:
        sentiment_score -= 1
    
    # Price change
    if price_change_pct > 0.5:
        sentiment_score += 1
    elif price_change_pct < -0.5:
        sentiment_score -= 1
    
    # ADX trend strength
    if latest['adx'] > 25:
        sentiment_score = min(2, max(-2, sentiment_score * 2))
    
    # Determine sentiment
    if sentiment_score > 0:
        return "Bullish"
    elif sentiment_score < 0:
        return "Bearish"
    else:
        return "Neutral"

# Read overall market sentiment
def read_market_sentiment():
    sentiment_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    sentiment_coins = {"Bullish": [], "Bearish": [], "Neutral": []}  # Track coins per sentiment
    avg_price_change = 0
    valid_symbols = 0
    
    for symbol in volatile_coins:
        df = fetch_data(symbol, timeframe, limit)
        if df is None:
            continue
        
        sentiment = analyze_symbol_sentiment(df, symbol)
        if sentiment is None:
            continue
        
        sentiment_counts[sentiment] += 1
        sentiment_coins[sentiment].append(symbol)  # Add coin to its sentiment list
        # Price change for tiebreaker
        latest = df.iloc[-1]
        lookback = df.iloc[-10] if len(df) >= 10 else df.iloc[0]
        price_change_pct = ((latest['close'] - lookback['close']) / lookback['close']) * 100
        avg_price_change += price_change_pct
        valid_symbols += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    if valid_symbols == 0:
        return "Unable to determine market sentiment as of {} - No data available.".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    avg_price_change /= valid_symbols
    total_votes = sum(sentiment_counts.values())
    
    # Majority vote
    max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if sentiment_counts[max_sentiment] > total_votes / 2:
        overall_sentiment = max_sentiment
        reasoning = f"Majority of coins showed {overall_sentiment.lower()} signals"
        contributing_coins = sentiment_coins[overall_sentiment]
    else:
        overall_sentiment = "Bullish" if avg_price_change > 0 else "Bearish" if avg_price_change < 0 else "Neutral"
        reasoning = f"No majority; decided by average price change ({avg_price_change:.2f}%)"
        contributing_coins = sentiment_coins[overall_sentiment]
    
    coins_str = ", ".join(contributing_coins) if contributing_coins else "None"
    result = f"{overall_sentiment} as of {current_time} - {reasoning} (Coins: {coins_str})"
    logger.info(result)  # Log the full result
    return result

# Execute sentiment reading
if __name__ == "__main__":
    print(read_market_sentiment())

//add logic to switch between strategies
