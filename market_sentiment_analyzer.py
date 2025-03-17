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

# Set up logging
log_file = os.path.join(os.path.dirname(__file__), 'log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
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

json_file_path = 'coinbase.json'
api_key, secret, passphrase = load_keys(json_file_path)

# Initialize exchange
exchange = ccxt.coinbaseadvanced({
    'apiKey': api_key,
    'secret': secret,
    'passphrase': passphrase,
    'enableRateLimit': True,
})
logger.info("Using coinbaseadvanced exchange")

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
        logger.warning(f"Not enough data to analyze {symbol}")
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
    lookback = 10  # 10 minutes lookback
    
    # Price change over last 10 periods
    price_change_pct = ((latest['close'] - df.iloc[-lookback]['close']) / df.iloc[-lookback]['close']) * 100 if len(df) >= lookback else 0
    
    # Sentiment scoring
    sentiment_score = 0  # -1 (Bearish), 0 (Neutral), 1 (Bullish)
    
    # RSI
    if latest['rsi'] > 70:
        sentiment_score += 1  # Bullish
    elif latest['rsi'] < 30:
        sentiment_score -= 1  # Bearish
    
    # MACD crossover
    if latest['macd_diff'] > 0 and prev['macd_diff'] <= 0:
        sentiment_score += 1  # Bullish crossover
    elif latest['macd_diff'] < 0 and prev['macd_diff'] >= 0:
        sentiment_score -= 1  # Bearish crossover
    
    # Price change
    if price_change_pct > 0.5:
        sentiment_score += 1  # Bullish
    elif price_change_pct < -0.5:
        sentiment_score -= 1  # Bearish
    
    # ADX trend strength (amplifier)
    trend_strength = "Strong" if latest['adx'] > 25 else "Weak"
    if trend_strength == "Strong":
        sentiment_score = min(2, max(-2, sentiment_score * 2))  # Double impact for strong trends
    
    # Determine sentiment
    if sentiment_score > 0:
        sentiment = "Bullish"
    elif sentiment_score < 0:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    
    return {
        'symbol': symbol,
        'sentiment': sentiment,
        'price_change_pct': price_change_pct,
        'rsi': latest['rsi'],
        'macd_diff': latest['macd_diff'],
        'adx': latest['adx'],
        'timestamp': latest['timestamp']
    }

# Read overall market sentiment
def read_market_sentiment():
    sentiment_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    avg_price_change = 0
    valid_symbols = 0
    
    for symbol in volatile_coins:
        df = fetch_data(symbol, timeframe, limit)
        if df is None:
            continue
        
        result = analyze_symbol_sentiment(df, symbol)
        if result is None:
            continue
        
        sentiment_counts[result['sentiment']] += 1
        avg_price_change += result['price_change_pct']
        valid_symbols += 1
        logger.info(f"{symbol}: Sentiment: {result['sentiment']} | Price Change: {result['price_change_pct']:.2f}% | RSI: {result['rsi']:.2f} | MACD Diff: {result['macd_diff']:.4f} | ADX: {result['adx']:.2f}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    if valid_symbols == 0:
        return "Unable to determine market sentiment due to insufficient data."
    
    avg_price_change /= valid_symbols
    total_votes = sum(sentiment_counts.values())
    
    # Majority vote
    max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    if sentiment_counts[max_sentiment] > total_votes / 2:
        overall_sentiment = max_sentiment
    else:
        # Tiebreaker: Use average price change
        overall_sentiment = "Bullish" if avg_price_change > 0 else "Bearish" if avg_price_change < 0 else "Neutral"
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sentiment_summary = f"The market sentiment is {overall_sentiment} as of {current_time}. " \
                        f"Breakdown: Bullish: {sentiment_counts['Bullish']}, Bearish: {sentiment_counts['Bearish']}, Neutral: {sentiment_counts['Neutral']}. " \
                        f"Avg Price Change: {avg_price_change:.2f}%"
    logger.info(sentiment_summary)
    return sentiment_summary

# Execute sentiment reading
if __name__ == "__main__":
    print(read_market_sentiment())
