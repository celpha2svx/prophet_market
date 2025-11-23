import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MT5Connector:
    """
    MetaTrader 5 API integration for trade execution and monitoring.
    Handles connections, orders, positions, and account info.
    """

    def __init__(self):
        self.connected = False
        self.account_info = None

    def connect(self, login=None, password=None, server=None):
        """
        Connect to MT5 terminal.

        Args:
            login: MT5 account login (optional if already logged in)
            password: MT5 account password (optional)
            server: Broker server name (optional)

        Returns:
            bool: True if connected successfully
        """
        # Initialize MT5 connection
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # If credentials provided, login
        if login and password and server:
            authorized = mt5.login(login=login, password=password, server=server)
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            logger.info(f"✓ Logged in to MT5: Account {login}")
        else:
            logger.info("✓ Connected to MT5 (using existing login)")

        self.connected = True
        self.update_account_info()

        # Display account info
        logger.info(f"  Account: {self.account_info['login']}")
        logger.info(f"  Balance: ${self.account_info['balance']:.2f}")
        logger.info(f"  Equity: ${self.account_info['equity']:.2f}")
        logger.info(f"  Broker: {self.account_info['server']}")

        return True

    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("✓ Disconnected from MT5")

    def update_account_info(self):
        """Update account information."""
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None

        account = mt5.account_info()
        if account is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None

        self.account_info = {
            'login': account.login,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'profit': account.profit,
            'server': account.server,
            'currency': account.currency,
            'leverage': account.leverage
        }

        return self.account_info

    def get_current_price(self, symbol):
        """
        Get current bid/ask price for a symbol.

        Args:
            symbol: Forex pair (e.g., 'EURUSD')

        Returns:
            dict: {'bid': float, 'ask': float, 'spread': float}
        """
        # Remove '=X' suffix if present
        symbol_clean = symbol.replace('=X', '')

        tick = mt5.symbol_info_tick(symbol_clean)

        if tick is None:
            logger.error(f"Failed to get price for {symbol_clean}: {mt5.last_error()}")
            return None

        return {
            'symbol': symbol_clean,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }

    def place_order(self, trade_setup):
        """
        Place a market order based on trade setup.

        Args:
            trade_setup: Dict from RiskManager with trade details

        Returns:
            dict: Order result with ticket number
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        # Clean symbol name
        symbol = trade_setup.get('broker_symbol',trade_setup['pair'].replace('=X', ''))

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None

        # Get symbol info for proper formatting
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None

        # Determine order type
        if trade_setup['direction'] == 'LONG':
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:  # SHORT
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid

        # Format price and levels to symbol's digits
        point = symbol_info.point
        digits = symbol_info.digits

        price = round(price, digits)
        sl = round(trade_setup['stop_loss'], digits)
        tp = round(trade_setup['take_profit'], digits)

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": trade_setup['position_size_lots'],
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Max price slippage in points
            "magic": 234000,  # Magic number to identify bot trades
            "comment": f"{trade_setup['session']} {trade_setup['confidence']:.0%}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        # Check result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        logger.info(f"✓ Order placed successfully!")
        logger.info(f"  Ticket: {result.order}")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Type: {trade_setup['direction']}")
        logger.info(f"  Volume: {trade_setup['position_size_lots']} lots")
        logger.info(f"  Entry: {result.price:.5f}")
        logger.info(f"  SL: {sl:.5f}")
        logger.info(f"  TP: {tp:.5f}")

        return {
            'ticket': result.order,
            'symbol': symbol,
            'direction': trade_setup['direction'],
            'volume': trade_setup['position_size_lots'],
            'entry_price': result.price,
            'sl': sl,
            'tp': tp,
            'time': datetime.now(),
            'trade_setup': trade_setup
        }

    def modify_position(self, ticket, new_sl=None, new_tp=None):
        """
        Modify stop loss or take profit of an open position.

        Args:
            ticket: Position ticket number
            new_sl: New stop loss price (optional)
            new_tp: New take profit price (optional)

        Returns:
            bool: True if modification successful
        """
        position = mt5.positions_get(ticket=ticket)

        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False

        position = position[0]
        symbol = position.symbol

        # Get symbol info for formatting
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits

        # Use existing values if not provided
        sl = round(new_sl, digits) if new_sl is not None else position.sl
        tp = round(new_tp, digits) if new_tp is not None else position.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify position {ticket}: {mt5.last_error()}")
            return False

        logger.info(f"✓ Position {ticket} modified: SL={sl:.5f}, TP={tp:.5f}")
        return True

    def close_position(self, ticket):
        """
        Close an open position.

        Args:
            ticket: Position ticket number

        Returns:
            bool: True if closed successfully
        """
        position = mt5.positions_get(ticket=ticket)

        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False

        position = position[0]
        symbol = position.symbol
        volume = position.volume
        position_type = position.type

        # Opposite order type to close
        if position_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Bot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}: {mt5.last_error()}")
            return False

        logger.info(f"✓ Position {ticket} closed at {price:.5f}")
        return True

    def get_open_positions(self):
        """
        Get all open positions.

        Returns:
            List of position dicts
        """
        positions = mt5.positions_get()

        if positions is None:
            return []

        position_list = []
        for pos in positions:
            position_list.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'LONG' if pos.type == mt5.ORDER_TYPE_BUY else 'SHORT',
                'volume': pos.volume,
                'entry_price': pos.price_open,
                'current_price': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'time': datetime.fromtimestamp(pos.time)
            })

        return position_list

    def get_historical_data(self, symbol, timeframe, num_bars=100):
        """
        Get historical OHLC data from MT5.

        Args:
            symbol: Forex pair
            timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_H1)
            num_bars: Number of bars to retrieve

        Returns:
            DataFrame with OHLC data
        """
        symbol_clean = symbol.replace('=X', '')

        rates = mt5.copy_rates_from_pos(symbol_clean, timeframe, 0, num_bars)

        if rates is None:
            logger.error(f"Failed to get historical data: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Initialize connector
    mt5_conn = MT5Connector()

    # Connect (will use existing MT5 login if terminal is open)
    if mt5_conn.connect():

        # Get current price
        price = mt5_conn.get_current_price('EURUSD')
        if price:
            print(f"\nEURUSD: Bid={price['bid']:.5f}, Ask={price['ask']:.5f}")

        # Get open positions
        positions = mt5_conn.get_open_positions()
        print(f"\nOpen positions: {len(positions)}")

        for pos in positions:
            print(f"  {pos['ticket']}: {pos['symbol']} {pos['type']} "
                  f"{pos['volume']} lots, P&L: ${pos['profit']:.2f}")

        # Example trade setup (don't actually place this)
        example_trade = {
            'pair': 'EURUSD=X',
            'direction': 'LONG',
            'position_size_lots': 0.01,
            'stop_loss': 1.0800,
            'take_profit': 1.0900,
            'session': 'London',
            'confidence': 0.87
        }

        # Uncomment to place order
        # result = mt5_conn.place_order(example_trade)

        # Disconnect
        mt5_conn.disconnect()
    else:
        print("Failed to connect to MT5")