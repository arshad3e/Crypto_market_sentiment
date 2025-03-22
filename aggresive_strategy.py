import ccxt
import pandas as pd
import ta
import time
import json
import logging
import os
import threading
import queue
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError

# --- Enhanced Logging Setup ---
log_file = os.path.join(os.path.dirname(__file__), 'trading_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
initial_balance = 0
transaction_count = 0
profit_loss_list = []
positions = {}
signal_queue = queue.Queue()
transaction_history = []

# --- Trading Parameters ---
timeframe = '1m'
limit = 150
leverage = 1 # Reduced Leverage SIGNIFICANTLY for Risk Control!
taker_fee = 0.004
min_trade_value = 20
stop_loss_pct = 0.025
trailing_stop_pct = 0.012
take_profit_target = 0.06
REQUESTS_PER_MINUTE = 170
SLEEP_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# --- Capital Allocation ---
total_capital = None
well_known_coins = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'LTC/USD', 'ADA/USD']
risky_coins = ['DOGE/USD', 'SHIB/USD', 'MATIC/USD']
max_trade_pct = 0.15
min_trade_pct = 0.08

# --- Time Limits ---
HFT_TIME_LIMIT = timedelta(minutes=4)
MOMENTUM_TIME_LIMIT = timedelta(minutes=12)


# --- Load API Keys ---
def load_keys(file_path):
    with open(file_path, 'r') as f:
        keys = json.load(f)
    return keys['name'], keys['privateKey'], keys.get('passphrase', '')

json_file_path = '/Users/arshadshaik/Desktop/python_practice/momentum_trading/coinbase.json'
api_key, secret, passphrase = load_keys(json_file_path)

# --- Initialize Exchange ---
exchange = ccxt.coinbaseadvanced({
    'apiKey': api_key,
    'secret': secret,
    'passphrase': passphrase,
    'enableRateLimit': True,
})
logger.info("Using coinbaseadvanced exchange")

# --- Test Authentication and Set Initial Capital ---
try:
    balance = exchange.fetch_balance()
    total_capital = balance.get('total', {}).get('USD', 0)
    initial_balance = total_capital
    logger.info(f"Authentication successful. Initial Capital: ${total_capital:.2f}")
except Exception as e:
    logger.error(f"Authentication failed during initialization: {e}. Please check API keys and permissions.")
    exit(1)

# --- Get Trading Pairs ---
def get_trading_pairs():
    markets = exchange.load_markets()
    trading_pairs = [market for market in markets.keys() if market.endswith('/USD') and markets[market]['active']]
    logger.info(f"Found {len(trading_pairs)} USD-based trading pairs: {trading_pairs}")
    return trading_pairs

# --- Initialize Positions ---
def initialize_positions():
    balance = exchange.fetch_balance()
    for symbol, amount in balance['total'].items():
        if amount > 0 and symbol != 'USD':
            pair = f"{symbol}/USD"
            if pair in get_trading_pairs():
                try:
                    current_price = exchange.fetch_ticker(pair)['last']
                    buy_value = amount * current_price
                    positions[pair] = {
                        'amount': amount,
                        'entry_price': current_price,
                        'stop_loss': current_price * (1 - stop_loss_pct),
                        'trailing_stop': current_price * (1 - trailing_stop_pct),
                        'highest_price': current_price,
                        'buy_fee': 0,
                        'entry_time': datetime.now(),
                        'strategy': 'Initialization',
                        'trade_count': 0
                    }
                    logger.info(f"Initialized position for {pair}: Amount {amount:.6f}, Entry Price ${current_price:.2f}, Value ${buy_value:.2f}")
                except Exception as e:
                    logger.error(f"Error initializing position for {pair}: {e}")

# --- Helper Functions ---
def get_usd_balance():
    try:
        balance = exchange.fetch_free_balance()
        usd_balance = balance.get('USD', 0)
        logger.debug(f"Current USD Balance: ${usd_balance:.2f}")
        return usd_balance
    except Exception as e:
        logger.error(f"Error fetching USD balance: {e}")
        return 0

def get_total_value():
    total_value = get_usd_balance()
    for symbol, pos in positions.items():
        try:
            current_price = exchange.fetch_ticker(symbol)['last']
            total_value += pos['amount'] * current_price
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
    return total_value

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(HTTPError)
)
def fetch_data(symbol, timeframe='1m', limit=100):
    df = None
    try:
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {symbol}. Backing off and retrying... {e}")
        raise
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching data for {symbol}: {e}. Retrying...")
        raise
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching data for {symbol}: {e}. Check symbol/exchange status. Retrying...")
        raise
    except HTTPError as e:
        logger.warning(f"HTTP error fetching data for {symbol}: {e}. Retrying...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {e}")
        return None


def analyze_data(df, symbol): # --- Analyze Data (TUNED STRATEGIES - MORE STRINGENT CONDITIONS) ---
    if df is None or len(df) < limit - 20:
        logger.warning(f"Insufficient data for {symbol} analysis. Dataframe length: {len(df) if df is not None else 0}")
        return None

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'], window_slow=13, window_fast=6, window_sign=5)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd'] - macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['ema5'] = df['close'].ewm(span=5).mean()
    df['ema15'] = df['close'].ewm(span=15).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "Hold"
    strategy = None

    # --- Buy Signals - Refined & Logged Strategies ---
    if latest['atr'] > df['atr'].mean() * 1.8 and latest['adx'] > 20:  # Reduced ATR multiplier, ADX threshold - More sensitive HFT - Abuhashish influence
        if latest['ema5'] > latest['ema15']:
            signal = "Buy"
            strategy = "HFT_Vol_ADX_EMA" # More descriptive strategy name
            logger.debug(f"{symbol} - HFT Strategy Conditions Met: ATR > {df['atr'].mean() * 1.8:.2f}, ADX > 20, EMA5 > EMA15")
        else:
            logger.debug(f"{symbol} - HFT Strategy Volatility and ADX met, but EMA5 not above EMA15. Hold.")
            strategy = "HFT_Vol_ADX_EMA_Hold" # Indicate reason for HOLD within strategy group
    elif latest['rsi'] < 65 and latest['macd_diff'] > 0 and prev['macd_diff'] <= 0: # RSI Threshold Adjusted - More Momentum Sensitivity - Foley & Cheah Influence
        signal = "Buy"
        strategy = "Momentum_MACD_RSI" # More descriptive momentum strategy name
        logger.debug(f"{symbol} - Momentum Strategy Conditions Met: RSI < 65, MACD Diff cross above zero.")
    elif (latest['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close'] > 0.045 and latest['rsi'] < 75: # Lookback adjusted to 20, threshold to 0.045, RSI < 75 - More Behavioral Sensitivity - Glaser & LoPucki Influence
        signal = "Buy"
        strategy = "Behavioral_Price_RSI" # Descriptive name
        logger.debug(f"{symbol} - Behavioral Strategy Conditions Met: Price Increase > 4.5% over last 20 periods, RSI < 75")
    else:
        strategy = "No_Signal" # Indicate no signal was triggered

    # --- Sell Signals ---
    if symbol in positions:
        position = positions[symbol]
        latest_price = latest['close'] # Use latest close price consistently
        bid_price = latest_price # In fast markets, last close is a reasonable sell trigger even if bid/ask is fluctuating.
        exit_value = position['amount'] * latest_price # Consistent price
        sell_fee = taker_fee * exit_value
        net_proceeds = exit_value - sell_fee
        gross_profit = net_proceeds - (position['amount'] * position['entry_price'])
        net_profit = gross_profit - position['buy_fee']
        net_profit_percentage = (net_profit / (position['amount'] * position['entry_price'])) if (position['amount'] * position['entry_price']) > 0 else 0 # Calculate percentage profit
        time_elapsed = datetime.now() - position['entry_time']
        time_limit = HFT_TIME_LIMIT if position['strategy'].startswith("HFT") else MOMENTUM_TIME_LIMIT # Check strategy type more robustly

        # --- Trailing Stop Logic - Enhanced ---
        if latest_price > position['highest_price']:
            position['highest_price'] = latest_price
            position['trailing_stop'] = position['highest_price'] * (1 - trailing_stop_pct)
            logger.debug(f"{symbol} - Trailing Stop Updated: Highest Price ${position['highest_price']:.2f}, Trailing Stop ${position['trailing_stop']:.2f}") # Debug log trailing stop update

        # --- Sell Conditions - Comprehensive & Logged Reasons ---
        if net_profit_percentage >= take_profit_target:
            signal = "Sell"
            strategy = position['strategy'] + "_Take_Profit" # Specific exit reason within strategy group
            logger.info(f"{symbol} - Take Profit Condition Met: Net Profit % {net_profit_percentage*100:.2f}% >= Target {take_profit_target*100:.2f}%")
        elif latest_price <= position['stop_loss']:
            signal = "Sell"
            strategy = position['strategy'] + "_Stop_Loss" # Specific exit reason
            logger.info(f"{symbol} - Stop Loss Triggered: Price ${latest_price:.2f} <= Stop Loss ${position['stop_loss']:.2f}")
        elif latest_price <= position['trailing_stop']:
            signal = "Sell"
            strategy = position['strategy'] + "_Trailing_Stop" # Exit reason for trailing stop
            logger.info(f"{symbol} - Trailing Stop Triggered: Price ${latest_price:.2f} <= Trailing Stop ${position['trailing_stop']:.2f}")
        elif time_elapsed >= time_limit:
            signal = "Sell"
            strategy = position['strategy'] + "_Time_Limit" # Exit due to time limit
            logger.info(f"{symbol} - Time Limit Reached: Held for {time_elapsed}, Limit {time_limit}")

    return {
        'symbol': symbol,
        'timestamp': latest['timestamp'],
        'price': latest['close'], # Latest close price
        'rsi': latest['rsi'],
        'macd_diff': latest['macd_diff'],
        'atr': latest['atr'],
        'adx': latest['adx'],
        'signal': signal,
        'strategy': strategy
    }


def scanner_thread(trading_pairs):
    well_known_pairs = [p for p in trading_pairs if p in well_known_coins]
    risky_pairs = [p for p in trading_pairs if p in risky_coins]
    all_pairs = well_known_pairs + risky_pairs
    while True:
        try:
            positions_count = len(positions) # Get position count outside the loop
            logger.info(f"Scanning for signals - Currently holding {positions_count} positions.") # Log position count during scan

            for symbol in all_pairs:
                df = fetch_data(symbol, timeframe, limit)
                if df is None:
                    continue
                result = analyze_data(df, symbol)
                if result and result['signal'] in ["Buy", "Sell"]:
                    signal_queue.put(result)
                    logger.info(f"Trade Signal Detected: {result['signal']} {symbol} | Strategy: {result['strategy']} | Price: ${result['price']:.2f}")
        except Exception as e:
            logger.error(f"Scanner thread error: {e}")
        time.sleep(60)


def execute_trade(signal, capital):
    global transaction_count
    usd_balance = get_usd_balance()
    if usd_balance < min_trade_value:
        logger.warning(f"Insufficient USD balance (${usd_balance:.2f}) to execute trade. Minimum trade value is ${min_trade_value:.2f}") # Log if balance too low
        return capital

    symbol = signal['symbol']
    price = signal['price'] # Signal price might be close price, use ticker for bid/ask now.
    is_well_known = symbol in well_known_coins
    max_allocation = 0.70 if is_well_known else 0.30
    trade_pct = min(max_allocation * max_trade_pct, max(min_trade_pct, usd_balance / total_capital))
    position_size = total_capital * trade_pct * leverage
    position_size = max(min_trade_value, min(position_size, usd_balance - taker_fee * position_size)) # Fee consideration within trade size limit
    amount = position_size / price

    try:
        ticker = exchange.fetch_ticker(symbol)
        ask_price = ticker['ask']
        bid_price = ticker['bid']

        if signal['signal'] == "Buy" and symbol not in positions and len(positions) < 5: # Limit position count to 5 - Configurable
            buy_fee = taker_fee * position_size
            total_cost = position_size + buy_fee
            if total_cost > usd_balance:
                logger.warning(f"Insufficient balance to cover buy order and fees for {symbol}. Required: ${total_cost:.2f}, Available: ${usd_balance:.2f}")
                return capital

            order = exchange.create_order(symbol, 'limit', 'buy', amount, ask_price)
            capital -= total_cost
            positions[symbol] = {
                'amount': amount,
                'entry_price': ask_price,
                'stop_loss': ask_price * (1 - stop_loss_pct),
                'trailing_stop': ask_price * (1 - trailing_stop_pct),
                'highest_price': ask_price,
                'buy_fee': buy_fee,
                'entry_time': datetime.now(),
                'strategy': signal['strategy'],
                'trade_count': 1 # Initialize trade count for position
            }
            transaction_count += 1
            transaction_id = transaction_count # Assign transaction id immediately after buy.
            transaction_history.append({ # Logged buy transaction
                'transaction_id': transaction_id,
                'symbol': symbol,
                'type': 'Buy',
                'amount': amount,
                'price': ask_price,
                'fee': buy_fee,
                'strategy': signal['strategy'],
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"#{transaction_id} BUY: {amount:.6f} {symbol} at ${ask_price:.2f}, Cost: ${total_cost:.2f} (Fee: ${buy_fee:.2f}) | Strategy: {signal['strategy']}")
            return capital

        elif signal['signal'] == "Sell" and symbol in positions:
            position = positions[symbol]
            amount = position['amount']
            exit_value = amount * bid_price
            sell_fee = taker_fee * exit_value
            net_proceeds = exit_value - sell_fee
            net_profit = net_proceeds - (amount * position['entry_price']) - position['buy_fee']
            net_profit_percentage = (net_profit / (amount * position['entry_price'])) if (amount * position['entry_price']) > 0 else 0 # Percentage Profit Calculation - Net Profit
            hold_duration = datetime.now() - position['entry_time']

            order = exchange.create_order(symbol, 'limit', 'sell', amount, bid_price)
            capital += net_proceeds
            profit_loss_list.append(net_profit)
            transaction_count += 1
            transaction_id = transaction_count # Assign transaction ID immediately for sell.
            transaction_history.append({ # Logged sell transaction with profit/loss, duration, fees
                'transaction_id': transaction_id,
                'symbol': symbol,
                'type': 'Sell',
                'amount': amount,
                'price': bid_price,
                'fee': sell_fee,
                'strategy': signal['strategy'],
                'profit_loss': net_profit,
                'profit_loss_percentage': net_profit_percentage,
                'hold_duration': str(hold_duration),
                'exit_reason': signal['strategy'].split('_')[-1] if '_' in signal['strategy'] else 'Unknown',
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"#{transaction_id} SELL: {amount:.6f} {symbol} at ${bid_price:.2f}, Proceeds: ${net_proceeds:.2f} (Fee: ${sell_fee:.2f}) | Net P/L ${net_profit:.2f} ({net_profit_percentage*100:.2f}%), Hold Duration: {hold_duration}")
            del positions[symbol]
            return capital

    except Exception as e:
        logger.error(f"Trade execution failed for {symbol} ({signal['signal']}): {e}")
        return capital


def monitor_positions(capital):
    if not positions:
        return capital

    for symbol in list(positions.keys()):
        try:
            position = positions[symbol]
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            bid_price = ticker['bid']

            # --- Trailing Stop Update within Monitor ---
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                position['trailing_stop'] = position['highest_price'] * (1 - trailing_stop_pct)
                logger.debug(f"{symbol} Monitor - Trailing Stop Updated: Highest Price ${position['highest_price']:.2f}, Trailing Stop ${position['trailing_stop']:.2f}") # Monitor thread trailing stop update log

            exit_value = position['amount'] * bid_price
            sell_fee = taker_fee * exit_value
            net_proceeds = exit_value - sell_fee
            net_profit = net_proceeds - (position['amount'] * position['entry_price']) - position['buy_fee']
            net_profit_percentage = (net_profit / (position['amount'] * position['entry_price'])) if (position['amount'] * position['entry_price']) > 0 else 0 # Percentage Profit Calculation - Monitor
            time_elapsed = datetime.now() - position['entry_time']
            time_limit = HFT_TIME_LIMIT if position['strategy'].startswith("HFT") else MOMENTUM_TIME_LIMIT

            # --- Monitoring Sell Conditions (Similar to analyze_data but executed directly for active positions) ---
            sell_signal = False # Flag to track if sell condition is met
            exit_reason = "Unknown_Monitor_Exit" # Default reason

            if net_profit_percentage >= take_profit_target:
                sell_signal = True
                exit_reason = "Take_Profit_Monitor"
                logger.info(f"{symbol} Monitor - Take Profit Condition Met: Net Profit % {net_profit_percentage*100:.2f}% >= Target {take_profit_target*100:.2f}%")
            elif current_price <= position['stop_loss']:
                sell_signal = True
                exit_reason = "Stop_Loss_Monitor"
                logger.info(f"{symbol} Monitor - Stop Loss Triggered: Price ${current_price:.2f} <= Stop Loss ${position['stop_loss']:.2f}")
            elif current_price <= position['trailing_stop']:
                sell_signal = True
                exit_reason = "Trailing_Stop_Monitor"
                logger.info(f"{symbol} Monitor - Trailing Stop Triggered: Price ${current_price:.2f} <= Trailing Stop ${position['trailing_stop']:.2f}")
            elif time_elapsed >= time_limit:
                sell_signal = True
                exit_reason = "Time_Limit_Monitor"
                logger.info(f"{symbol} Monitor - Time Limit Reached: Held for {time_elapsed}, Limit {time_limit}")


            if sell_signal:
                usd_balance_before_sell = get_usd_balance()
                if usd_balance_before_sell < min_trade_value:
                    logger.warning(f"Monitor SKIP SELL: {symbol} due to insufficient USD balance (${usd_balance_before_sell:.2f} < ${min_trade_value:.2f}) before attempting sell order. Check balance/fees/trade sizes.")
                    continue

                order = exchange.create_order(symbol, 'limit', 'sell', position['amount'], bid_price)
                capital += net_proceeds
                profit_loss_list.append(net_profit)
                transaction_count += 1
                transaction_id = transaction_count
                transaction_history.append({
                    'transaction_id': transaction_id,
                    'symbol': symbol,
                    'type': 'Sell',
                    'amount': position['amount'],
                    'price': bid_price,
                    'fee': sell_fee,
                    'strategy': position['strategy'],
                    'profit_loss': net_profit,
                    'profit_loss_percentage': net_profit_percentage,
                    'hold_duration': str(time_elapsed),
                    'exit_reason': exit_reason,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"#{transaction_id} MONITOR SELL ({exit_reason}): Sold {position['amount']:.6f} {symbol} at ${bid_price:.2f}, Proceeds: ${net_proceeds:.2f} (Fee: ${sell_fee:.2f}) | Net P/L ${net_profit:.2f} ({net_profit_percentage*100:.2f}%), Hold Duration: {time_elapsed}")
                del positions[symbol]

        except Exception as e:
            logger.error(f"Error monitoring position for {symbol}: {e}")
    return capital



def executor_thread():
    global initial_balance
    capital = get_usd_balance()
    logger.info(f"Executor thread started. Initial balance: ${capital:.2f}")

    hft_trade_count = 0
    momentum_trade_count = 0
    behavioral_trade_count = 0

    hft_wins = 0
    hft_losses = 0
    momentum_wins = 0
    momentum_losses = 0
    behavioral_wins = 0
    behavioral_losses = 0


    while True:
        try:
            while not signal_queue.empty():
                signal = signal_queue.get()
                capital = execute_trade(signal, capital)
                signal_queue.task_done()

            capital = monitor_positions(capital)
            total_value = get_total_value()
            current_positions_count = len(positions)
            total_transactions = len(transaction_history)
            profitable_transactions = sum(1 for trans in transaction_history if trans['type'] == 'Sell' and trans['profit_loss'] > 0)
            loss_transactions = sum(1 for trans in transaction_history if trans['type'] == 'Sell' and trans['profit_loss'] <= 0)
            win_rate = (profitable_transactions / (profitable_transactions + loss_transactions)) * 100 if (profitable_transactions + loss_transactions) > 0 else 0
            total_profit = sum(trans['profit_loss'] for trans in transaction_history if trans['type'] == 'Sell')

            logger.info(f"--- Executor Status --- | Total Value: ${total_value:.2f} | USD Balance: ${capital:.2f} | Positions: {current_positions_count} | Transactions: {total_transactions} (Wins: {profitable_transactions}, Losses: {loss_transactions}, Win Rate: {win_rate:.2f}%) | Total Net Profit: ${total_profit:.2f}")


            time.sleep(1)

        except Exception as e:
            logger.error(f"Executor thread error: {e}")
            time.sleep(60)


def main():
    trading_pairs = get_trading_pairs()
    initialize_positions()

    scanner = threading.Thread(target=scanner_thread, args=(trading_pairs,), daemon=True)
    scanner.start()

    executor = threading.Thread(target=executor_thread, daemon=True)
    executor.start()

    try:
        while True:
            time.sleep(1200)
            positions_count = len(positions)
            usd_balance = get_usd_balance()
            total_value = get_total_value()
            logger.info(f"--- Heartbeat --- | Active Positions: {positions_count} | USD Balance: ${usd_balance:.2f} | Total Account Value: ${total_value:.2f} | Running...")
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user.")
        total_value = get_total_value()
        usd_balance = get_usd_balance()
        final_profit = total_value - initial_balance
        transaction_summary = summarize_transactions()
        logger.info(f"--- Final Bot Status ---")
        logger.info(f"Final Account Value: ${total_value:.2f} | USD Balance: ${usd_balance:.2f}")
        logger.info(f"Total Net Profit: ${final_profit:.2f} (vs. Initial Capital: ${initial_balance:.2f})")
        logger.info(f"Transaction Summary: {transaction_summary}")

def summarize_transactions():
    """Summarizes transaction history for logging at the end."""
    profitable_transactions = sum(1 for trans in transaction_history if trans['type'] == 'Sell' and trans['profit_loss'] > 0)
    loss_transactions = sum(1 for trans in transaction_history if trans['type'] == 'Sell' and trans['profit_loss'] <= 0)
    win_rate = (profitable_transactions / (profitable_transactions + loss_transactions)) * 100 if (profitable_transactions + loss_transactions) > 0 else 0
    total_profit = sum(trans['profit_loss'] for trans in transaction_history if trans['type'] == 'Sell')
    avg_profit_per_transaction = total_profit / profitable_transactions if profitable_transactions > 0 else 0
    avg_loss_per_transaction = sum(trans['profit_loss'] for trans in transaction_history if trans['type'] == 'Sell' and trans['profit_loss'] <= 0) / loss_transactions if loss_transactions > 0 else 0
    return {
        "total_transactions": len(transaction_history),
        "profitable_trades": profitable_transactions,
        "loss_trades": loss_transactions,
        "win_rate": f"{win_rate:.2f}%",
        "total_net_profit": f"${total_profit:.2f}",
        "average_profit_per_win": f"${avg_profit_per_transaction:.2f}",
        "average_loss_per_loss": f"${avg_loss_per_transaction:.2f}"
    }


if __name__ == "__main__":
    main()

//need to work on just bitcoin
//avoid risky coins
