import time
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from src.forex_bot_pipeline.session_predictor import SessionPredictor
from src.forex_bot_pipeline.session_risk_manager import RiskManager
from src.forex_bot_pipeline.session_mt5 import MT5Connector
from src.forex_bot_pipeline.session_trade_manager import TradeManager
from src.forex_bot_pipeline.symbol_mapping import SymbolMapper
from src.forex_bot_pipeline.data_updater import DataUpdater
from src.data_pipeline.forex_cl_data import DataProcessor
from src.utils.paths import DB_PATH,CONFIG_JSON

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SessionTradingBot:
    """
    Complete automated trading bot for session-based forex trading.
    Predicts session direction, manages risk, executes trades, and monitors positions.
    """

    def __init__(self, config_file=CONFIG_JSON):
        """
        Initialize the trading bot.

        Args:
            config_file: Path to configuration JSON file
        """
        logger.info("=" * 80)
        logger.info("SESSION TRADING BOT - INITIALIZING")
        logger.info("=" * 80)

        # Load configuration
        self.config = self.load_config(config_file)

        # Initialize components
        self.data_processor = DataProcessor(db_path=str(DB_PATH))
        self.predictor = SessionPredictor()
        self.symbol_mapper = SymbolMapper(broker=self.config.get('broker','exness'))
        self.symbol_mapper.display_mapping()
        self.risk_manager = RiskManager(profile=self.config['risk_profile'])
        self.mt5 = MT5Connector()
        self.trade_manager = TradeManager(self.mt5)

        # State tracking
        self.last_session_traded = None
        self.daily_stats = {
            'trades_taken': 0,
            'trades_won': 0,
            'trades_lost': 0,
            'total_profit': 0.0,
            'start_balance': 0.0
        }

        logger.info("✓ Bot initialization complete")

    def load_config(self, config_file):
        """Load configuration from JSON file."""
        config_path = Path(config_file)

        if not config_path.exists():
            logger.warning(f"Config file not found, creating default: {config_file}")
            default_config = {
                "risk_profile": "CONSERVATIVE",
                "min_confidence": 0.70,
                "pairs": [
                    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                    "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
                    "EURJPY=X", "GBPJPY=X", "GC=F"
                ],
                "mt5_login": None,
                "mt5_password": None,
                "mt5_server": None,
                "trading_enabled": True,
                "notification_enabled": False
            }

            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)

            return default_config

        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"✓ Configuration loaded from {config_file}")
        logger.info(f"  Risk Profile: {config['risk_profile']}")
        logger.info(f"  Min Confidence: {config['min_confidence']:.0%}")
        logger.info(f"  Pairs: {len(config['pairs'])}")

        return config

    def connect_to_broker(self):
        """Connect to MT5 broker."""
        logger.info("\nConnecting to MT5...")

        success = self.mt5.connect(
            login=self.config.get('mt5_login'),
            password=self.config.get('mt5_password'),
            server=self.config.get('mt5_server')
        )

        if not success:
            logger.error("Failed to connect to MT5!")
            return False

        # Update account info
        self.mt5.update_account_info()
        self.daily_stats['start_balance'] = self.mt5.account_info['balance']

        return True

    def get_next_session_time(self):
        """
        Determine when the next session starts.

        Returns:
            tuple: (session_name, session_start_datetime, minutes_until_start)
        """
        now = datetime.utcnow()
        current_hour = now.hour

        # London: 8:00 GMT, NY: 13:00 GMT
        if current_hour < 8:
            # Next is London today
            next_session = 'London'
            next_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
        elif 8 <= current_hour < 13:
            # Next is NY today
            next_session = 'NY'
            next_time = now.replace(hour=13, minute=0, second=0, microsecond=0)
        else:
            # Next is London tomorrow
            next_session = 'London'
            next_time = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

        minutes_until = (next_time - now).total_seconds() / 60

        return next_session, next_time, minutes_until

    def should_trade_now(self):
        """
        Check if bot should make predictions and trade now.
        Trades 15 minutes before session start.

        Returns:
            tuple: (should_trade, session_name)
        """
        session_name, session_time, minutes_until = self.get_next_session_time()

        # Trade if we're within 15 minutes of session start
        if 0 <= minutes_until <= 15:
            # Avoid trading same session twice
            session_id = f"{session_time.date()}_{session_name}"

            if session_id != self.last_session_traded:
                return True, session_name

        return False, None

    def make_predictions(self):
        """
        Get predictions for all pairs for upcoming session.

        Returns:
            List of prediction dicts sorted by confidence
        """
        updater = DataUpdater(str(DB_PATH))
        updater.update_all_symbols(self.config['pairs'])

        logger.info("\n" + "=" * 80)
        logger.info("MAKING PREDICTIONS")
        logger.info("=" * 80)

        predictions = self.predictor.predict_all_pairs(
            pairs=self.config['pairs'],
            data_processor=self.data_processor,
            min_confidence=self.config['min_confidence']
        )

        return predictions

    def execute_trades(self, predictions):
        """
        Execute trades based on predictions.

        Args:
            predictions: List of prediction dicts

        Returns:
            List of executed trade results
        """
        if not predictions:
            logger.info("No signals meet criteria - skipping this session")
            return []

        # Filter trades by risk profile
        selected_predictions = self.risk_manager.filter_trades_by_profile(predictions)

        if not selected_predictions:
            logger.info("No trades selected after filtering")
            return []

        # Get current account balance
        self.mt5.update_account_info()
        account_balance = self.mt5.account_info['balance']

        logger.info(f"\nAccount Balance: ${account_balance:.2f}")
        logger.info(f"\n{'=' * 80}")
        logger.info("EXECUTING TRADES")
        logger.info(f"{'=' * 80}")

        executed_trades = []

        for pred in selected_predictions:
            try:
                # Get current price
                broker_symbol = self.symbol_mapper.to_broker(pred['pair'])
                price_info = self.mt5.get_current_price(broker_symbol)

                if price_info is None:
                    logger.warning(f"Could not get price for {pred['pair']}, skipping")
                    continue

                current_price = price_info['ask'] if pred['direction'] == 'BULLISH' else price_info['bid']

                # Get ATR for dynamic SL sizing (optional - can use historical data)
                # For now, we'll let risk_manager use fixed values

                # Calculate trade setup
                trade_setup = self.risk_manager.calculate_trade_levels(
                    prediction=pred,
                    account_balance=account_balance,
                    current_price=current_price,
                    atr=None  # Will use fixed SL based on profile
                )

                trade_setup['broker_symbol'] = broker_symbol

                # Display trade plan
                logger.info(f"\n--- Trade {len(executed_trades) + 1} ---")
                logger.info(f"Pair: {trade_setup['pair']}")
                logger.info(f"Direction: {trade_setup['direction']}")
                logger.info(f"Confidence: {trade_setup['confidence']:.1%}")
                logger.info(f"Entry: {trade_setup['entry_price']:.5f}")
                logger.info(f"SL: {trade_setup['stop_loss']:.5f} ({trade_setup['sl_pips']:.1f} pips)")
                logger.info(f"TP: {trade_setup['take_profit']:.5f} ({trade_setup['tp_pips']:.1f} pips)")
                logger.info(f"Position: {trade_setup['position_size_lots']} lots")
                logger.info(f"Risk: ${trade_setup['risk_amount']:.2f} ({trade_setup['risk_percentage']:.1f}%)")
                logger.info(f"Potential Profit: ${trade_setup['potential_profit']:.2f}")

                # Execute if trading is enabled
                if self.config['trading_enabled']:
                    order_result = self.mt5.place_order(trade_setup)

                    if order_result:
                        # Add to trade manager for monitoring
                        self.trade_manager.add_trade(order_result)
                        executed_trades.append(order_result)
                        self.daily_stats['trades_taken'] += 1
                    else:
                        logger.error(f"Failed to execute trade for {pred['pair']}")
                else:
                    logger.info("⚠  Trading disabled - trade NOT executed (dry run)")

            except Exception as e:
                logger.error(f"Error executing trade for {pred['pair']}: {e}")

        logger.info(f"\n✓ Executed {len(executed_trades)} trades")

        return executed_trades

    def monitor_session(self, session_name):
        """
        Monitor active trades throughout the session.

        Args:
            session_name: 'London' or 'NY'
        """
        # Determine session end time
        if session_name == 'London':
            end_hour = 12
        else:  # NY
            end_hour = 17

        logger.info(f"\n{'=' * 80}")
        logger.info(f"MONITORING {session_name.upper()} SESSION")
        logger.info(f"{'=' * 80}")

        while True:
            current_time = datetime.utcnow()
            # Check if session ended
            if current_time.hour >= end_hour:
                logger.info(f"\n⏰ {session_name} session ended")
                break

            # Monitor trades (trailing stops, etc.)
            self.trade_manager.monitor_active_trades()

            # Display active trades every 5 minutes
            if current_time.minute % 5 == 0:
                self.trade_manager.display_active_trades()

            # Sleep for 30 seconds before next check
            time.sleep(30)

        # Close any remaining trades at session end
        self.trade_manager.close_all_trades(reason=f"{session_name} session ended")

    def calculate_session_results(self):
        """Calculate and log session results."""
        self.mt5.update_account_info()

        current_balance = self.mt5.account_info['balance']
        session_profit = current_balance - self.daily_stats['start_balance']

        logger.info(f"\n{'=' * 80}")
        logger.info("SESSION RESULTS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Starting Balance: ${self.daily_stats['start_balance']:.2f}")
        logger.info(f"Ending Balance: ${current_balance:.2f}")
        logger.info(
            f"Session P&L: ${session_profit:+.2f} ({(session_profit / self.daily_stats['start_balance'] * 100):+.2f}%)")
        logger.info(f"Trades Taken: {self.daily_stats['trades_taken']}")

        # Update for next session
        self.daily_stats['start_balance'] = current_balance
        self.daily_stats['total_profit'] += session_profit

    def run_session_cycle(self):
        """
        Execute one complete trading cycle:
        1. Wait for session time
        2. Make predictions
        3. Execute trades
        4. Monitor session
        5. Calculate results
        """
        should_trade, session_name = self.should_trade_now()

        if should_trade:
            logger.info(f"\n🚀 STARTING {session_name.upper()} SESSION CYCLE")

            # Mark this session as traded
            now = datetime.utcnow()
            self.last_session_traded = f"{now.date()}_{session_name}"

            # 1. Make predictions
            predictions = self.make_predictions()

            # 2. Execute trades
            executed_trades = self.execute_trades(predictions)

            if executed_trades:
                # 3. Monitor session
                self.monitor_session(session_name)

                # 4. Calculate results
                self.calculate_session_results()
            else:
                logger.info("No trades executed this session")

    def run(self):
        """
        Main bot loop - runs continuously.
        """
        logger.info("\n" + "=" * 80)
        logger.info("SESSION TRADING BOT - STARTING")
        logger.info("=" * 80)

        # Connect to broker
        if not self.connect_to_broker():
            logger.error("Cannot start bot without MT5 connection")
            return

        logger.info("\n✓ Bot is LIVE and running!")
        logger.info("Press Ctrl+C to stop\n")

        try:
            while True:
                # Check if it's time to trade
                session_name, session_time, minutes_until = self.get_next_session_time()

                logger.info(f"Next session: {session_name} at {session_time.strftime('%H:%M GMT')} "
                            f"({minutes_until:.0f} minutes)")

                # Run session cycle if it's time
                self.run_session_cycle()

                # Sleep for 5 minutes before next check
                time.sleep(300)

        except KeyboardInterrupt:
            logger.info("\n\n⚠  Bot stopped by user")
            self.shutdown()
        except Exception as e:
            logger.error(f"\n\n❌ Bot error: {e}")
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the bot."""
        logger.info("\n" + "=" * 80)
        logger.info("SHUTTING DOWN BOT")
        logger.info("=" * 80)

        # Close all open positions
        logger.info("Closing all open positions...")
        self.trade_manager.close_all_trades(reason="Bot shutdown")

        # Display daily summary
        logger.info(f"\nDaily Summary:")
        logger.info(f"  Total Trades: {self.daily_stats['trades_taken']}")
        logger.info(f"  Total P&L: ${self.daily_stats['total_profit']:+.2f}")

        # Disconnect from MT5
        self.mt5.disconnect()

        logger.info("\n✓ Bot shutdown complete")


# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    config_file = CONFIG_JSON

    # Initialize and run bot
    bot = SessionTradingBot(config_file=config_file)

    print("\n" + "=" * 80)
    print("SESSION TRADING BOT")
    print("=" * 80)
    print(f"Risk Profile: {bot.config['risk_profile']}")
    print(f"Min Confidence: {bot.config['min_confidence']:.0%}")
    print(f"Trading Pairs: {len(bot.config['pairs'])}")
    print(f"Trading Enabled: {'YES' if bot.config['trading_enabled'] else 'NO (DRY RUN)'}")
    print("=" * 80)

    # Start the bot
    bot.run()