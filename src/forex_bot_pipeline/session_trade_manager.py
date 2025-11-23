import time
from datetime import datetime, time as dt_time
import logging
import pandas as pd
from src.forex_bot_pipeline.session_mt5 import  MT5Connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeManager:
    """
    Monitors active trades and manages trailing stops in real-time.
    Implements breakeven and progressive trailing stop logic.
    """

    def __init__(self, mt5_connector):
        """
        Initialize trade manager.

        Args:
            mt5_connector: MT5Connector instance
        """
        self.mt5 = mt5_connector
        self.active_trades = {}  # ticket: trade_info
        self.trailing_state = {}  # ticket: current_trailing_level

    def add_trade(self, order_result):
        """
        Add a new trade to monitoring.

        Args:
            order_result: Dict returned from mt5_connector.place_order()
        """
        ticket = order_result['ticket']

        self.active_trades[ticket] = {
            'ticket': ticket,
            'symbol': order_result['symbol'],
            'direction': order_result['direction'],
            'entry_price': order_result['entry_price'],
            'original_sl': order_result['sl'],
            'original_tp': order_result['tp'],
            'current_sl': order_result['sl'],
            'current_tp': order_result['tp'],
            'volume': order_result['volume'],
            'trade_setup': order_result['trade_setup'],
            'open_time': order_result['time'],
            'session': order_result['trade_setup']['session']
        }

        # Initialize trailing state
        self.trailing_state[ticket] = 'initial'  # States: initial, breakeven, trail_1, trail_2

        logger.info(f"✓ Trade {ticket} added to monitoring")

    def check_trailing_stop(self, ticket):
        """
        Check if trailing stop should be activated/updated.

        Args:
            ticket: Position ticket number

        Returns:
            bool: True if SL was modified
        """
        if ticket not in self.active_trades:
            return False

        trade = self.active_trades[ticket]
        current_state = self.trailing_state[ticket]

        # Get current position info
        positions = self.mt5.get_open_positions()
        position = next((p for p in positions if p['ticket'] == ticket), None)

        if position is None:
            logger.warning(f"Position {ticket} not found (may be closed)")
            return False

        current_price = position['current_price']
        direction = trade['direction']
        entry = trade['entry_price']

        # Get trailing levels from trade setup
        trailing_levels = trade['trade_setup']['trailing_levels']

        # Determine which trailing level to activate
        new_sl = None
        new_state = current_state

        if direction == 'LONG':
            # Check trailing levels (from lowest to highest)
            if current_state == 'initial':
                # Check for breakeven
                if current_price >= trailing_levels['breakeven_trigger']:
                    new_sl = trailing_levels['breakeven_sl']
                    new_state = 'breakeven'
                    logger.info(f"🔄 Trade {ticket}: Moving to BREAKEVEN (price: {current_price:.5f})")

            elif current_state == 'breakeven':
                # Check for trail level 1
                if current_price >= trailing_levels['trail_1_trigger']:
                    new_sl = trailing_levels['trail_1_sl']
                    new_state = 'trail_1'
                    logger.info(f"🔄 Trade {ticket}: TRAIL LEVEL 1 activated (locking +0.5R)")

            elif current_state == 'trail_1':
                # Check for trail level 2
                if current_price >= trailing_levels['trail_2_trigger']:
                    new_sl = trailing_levels['trail_2_sl']
                    new_state = 'trail_2'
                    logger.info(f"🔄 Trade {ticket}: TRAIL LEVEL 2 activated (locking +1.0R)")

        else:  # SHORT
            # Check trailing levels (from highest to lowest)
            if current_state == 'initial':
                # Check for breakeven
                if current_price <= trailing_levels['breakeven_trigger']:
                    new_sl = trailing_levels['breakeven_sl']
                    new_state = 'breakeven'
                    logger.info(f"🔄 Trade {ticket}: Moving to BREAKEVEN (price: {current_price:.5f})")

            elif current_state == 'breakeven':
                # Check for trail level 1
                if current_price <= trailing_levels['trail_1_trigger']:
                    new_sl = trailing_levels['trail_1_sl']
                    new_state = 'trail_1'
                    logger.info(f"🔄 Trade {ticket}: TRAIL LEVEL 1 activated (locking +0.5R)")

            elif current_state == 'trail_1':
                # Check for trail level 2
                if current_price <= trailing_levels['trail_2_trigger']:
                    new_sl = trailing_levels['trail_2_sl']
                    new_state = 'trail_2'
                    logger.info(f"🔄 Trade {ticket}: TRAIL LEVEL 2 activated (locking +1.0R)")

        # If new SL should be set, modify position
        if new_sl is not None:
            success = self.mt5.modify_position(ticket, new_sl=new_sl)

            if success:
                # Update trade info
                trade['current_sl'] = new_sl
                self.trailing_state[ticket] = new_state

                logger.info(f"✓ Trade {ticket}: SL updated to {new_sl:.5f}")
                return True

        return False

    def should_close_at_session_end(self, ticket):
        """
        Check if trade should be closed at session end.

        Args:
            ticket: Position ticket number

        Returns:
            bool: True if session has ended
        """
        if ticket not in self.active_trades:
            return False

        trade = self.active_trades[ticket]
        session = trade['session']

        # Get current time (GMT)
        current_time = datetime.utcnow().time()

        # Session end times
        london_end = dt_time(12, 0)  # 12:00 PM GMT
        ny_end = dt_time(17, 0)  # 5:00 PM GMT

        if session == 'London' and current_time >= london_end:
            return True
        elif session == 'NY' and current_time >= ny_end:
            return True

        return False

    def monitor_active_trades(self):
        """
        Main monitoring loop - checks all active trades for trailing stops
        and session end closures.
        """
        if not self.active_trades:
            return

        logger.debug(f"Monitoring {len(self.active_trades)} active trades...")

        # Get current positions from MT5
        current_positions = self.mt5.get_open_positions()
        current_tickets = {p['ticket'] for p in current_positions}

        # Check each active trade
        tickets_to_remove = []

        for ticket in list(self.active_trades.keys()):
            # Check if position still exists
            if ticket not in current_tickets:
                logger.info(f"✓ Trade {ticket} has been closed")
                tickets_to_remove.append(ticket)
                continue

            # Check trailing stop
            self.check_trailing_stop(ticket)

            # Check if session ended
            if self.should_close_at_session_end(ticket):
                logger.info(f"⏰ Session ended - Closing trade {ticket}")
                success = self.mt5.close_position(ticket)

                if success:
                    tickets_to_remove.append(ticket)

        # Remove closed trades from monitoring
        for ticket in tickets_to_remove:
            self.remove_trade(ticket)

    def remove_trade(self, ticket):
        """
        Remove trade from monitoring (after closure).

        Args:
            ticket: Position ticket number
        """
        if ticket in self.active_trades:
            trade = self.active_trades[ticket]
            del self.active_trades[ticket]
            del self.trailing_state[ticket]
            logger.info(f"Trade {ticket} removed from monitoring")

    def get_trade_summary(self):
        """
        Get summary of all active trades.

        Returns:
            DataFrame with trade details
        """
        if not self.active_trades:
            return pd.DataFrame()

        # Get current positions
        positions = self.mt5.get_open_positions()

        summary_data = []

        for ticket, trade in self.active_trades.items():
            position = next((p for p in positions if p['ticket'] == ticket), None)

            if position:
                summary_data.append({
                    'ticket': ticket,
                    'symbol': trade['symbol'],
                    'direction': trade['direction'],
                    'entry': trade['entry_price'],
                    'current': position['current_price'],
                    'sl': position['sl'],
                    'tp': position['tp'],
                    'profit': position['profit'],
                    'trailing_state': self.trailing_state[ticket],
                    'session': trade['session']
                })

        return pd.DataFrame(summary_data)

    def display_active_trades(self):
        """Display formatted table of active trades."""
        summary = self.get_trade_summary()

        if summary.empty:
            logger.info("No active trades")
            return

        logger.info("\n" + "=" * 80)
        logger.info("ACTIVE TRADES")
        logger.info("=" * 80)

        for _, row in summary.iterrows():
            pnl_sign = "+" if row['profit'] >= 0 else ""
            logger.info(f"\nTicket: {row['ticket']}")
            logger.info(f"  {row['symbol']} {row['direction']} | "
                        f"Entry: {row['entry']:.5f} | Current: {row['current']:.5f}")
            logger.info(f"  SL: {row['sl']:.5f} | TP: {row['tp']:.5f}")
            logger.info(f"  P&L: {pnl_sign}${row['profit']:.2f} | "
                        f"Trailing: {row['trailing_state'].upper()}")

    def close_all_trades(self, reason="Manual close"):
        """
        Close all active trades.

        Args:
            reason: Reason for closure (for logging)
        """
        if not self.active_trades:
            logger.info("No active trades to close")
            return

        logger.info(f"\n{'=' * 60}")
        logger.info(f"CLOSING ALL TRADES: {reason}")
        logger.info(f"{'=' * 60}")

        tickets = list(self.active_trades.keys())

        for ticket in tickets:
            success = self.mt5.close_position(ticket)

            if success:
                self.remove_trade(ticket)

        logger.info(f"✓ All trades closed")


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":

    # Initialize MT5
    mt5_conn = MT5Connector()

    if not mt5_conn.connect():
        print("Failed to connect to MT5")
        exit()

    # Initialize trade manager
    trade_mgr = TradeManager(mt5_conn)

    # Example: Monitor existing positions
    positions = mt5_conn.get_open_positions()

    if positions:
        print(f"\nFound {len(positions)} open positions")

        # In real scenario, you'd add these from order placement
        # For now, just display them
        for pos in positions:
            print(f"  {pos['ticket']}: {pos['symbol']} {pos['type']} "
                  f"P&L: ${pos['profit']:.2f}")

        # Monitoring loop example (run for 1 minute)
        print("\nStarting monitoring loop (60 seconds)...")

        end_time = time.time() + 60
        while time.time() < end_time:
            trade_mgr.monitor_active_trades()
            time.sleep(10)  # Check every 10 seconds

        print("\nMonitoring complete")
    else:
        print("No open positions to monitor")

    # Disconnect
    mt5_conn.disconnect()