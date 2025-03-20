import ccxt
import pandas as pd
import time
from datetime import datetime

exchange = ccxt.binance({'apiKey': 'your_key', 'secret': 'your_secret'})
symbol = 'BTC/USD'

def get_lob_data():
    order_book = exchange.fetch_order_book(symbol)
    bids, asks = order_book['bids'], order_book['asks']
    return sum(b[1] for b in bids[:5]), sum(a[1] for a in asks[:5])  # Top 5 levels

def get_ema(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().iloc[-1]

def trade():
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=20)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price = df['close'].iloc[-1]
        vol_avg = df['volume'].mean()

        ema5 = get_ema(df['close'], 5)
        ema15 = get_ema(df['close'], 15)
        bid_vol, ask_vol = get_lob_data()

        if ema5 > ema15 and df['volume'].iloc[-1] > vol_avg * 1.5 and bid_vol > ask_vol * 2:
            print(f"Buy at {price} - {datetime.now()}")
            # Place buy order logic here
            time.sleep(60)  # Check every minute
        time.sleep(1)

if __name__ == "__main__":
    trade()
