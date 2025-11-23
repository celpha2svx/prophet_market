import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages position sizing, stop loss, take profit, and trailing stops.
    Implements adaptive R:R ratios based on risk profile.
    """

    # Risk profiles configuration
    RISK_PROFILES = {
        'CONSERVATIVE': {
            'risk_per_trade': 0.03,  # 3%
            'max_trades_per_session': 3,
            'rr_ratio': 2.0,  # 1:2
            'description': 'Balanced approach, 3 trades per session'
        },
        'MODERATE': {
            'risk_per_trade': 0.05,  # 5%
            'max_trades_per_session': 2,
            'rr_ratio': 1.5,  # 1:1.5
            'description': 'Focus on top 2 signals'
        },
        'AGGRESSIVE': {
            'risk_per_trade': 0.10,  # 10%
            'max_trades_per_session': 1,
            'rr_ratio': 1.0,  # 1:1
            'description': 'All-in on best signal'
        }
    }

    def __init__(self, profile='CONSERVATIVE'):
        """
        Initialize risk manager with chosen profile.

        Args:
            profile: 'CONSERVATIVE', 'MODERATE', or 'AGGRESSIVE'
        """
        if profile not in self.RISK_PROFILES:
            raise ValueError(f"Invalid profile. Choose from: {list(self.RISK_PROFILES.keys())}")

        self.profile = profile
        self.config = self.RISK_PROFILES[profile]

        logger.info(f"Risk Manager initialized: {profile} profile")
        logger.info(f"  Risk per trade: {self.config['risk_per_trade']:.1%}")
        logger.info(f"  Max trades: {self.config['max_trades_per_session']}")
        logger.info(f"  R:R ratio: 1:{self.config['rr_ratio']}")

    def calculate_atr(self, pair_data, period=14):
        """
        Calculate Average True Range for dynamic SL/TP sizing.

        Args:
            pair_data: DataFrame with OHLC data
            period: ATR period

        Returns:
            float: ATR value
        """
        if len(pair_data) < period:
            logger.warning(f"Not enough data for ATR calculation (need {period}, got {len(pair_data)})")
            return None

        high = pair_data['high']
        low = pair_data['low']
        close = pair_data['close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def calculate_stop_loss_pips(self, pair, current_price, atr=None, direction='LONG'):
        """
        Calculate stop loss distance in pips.
        Uses ATR-based dynamic sizing.

        Args:
            pair: Forex pair
            current_price: Current market price
            atr: ATR value (optional, will use fixed if not provided)
            direction: 'LONG' or 'SHORT'

        Returns:
            float: Stop loss distance in pips
        """
        # Base SL on risk profile
        if self.profile == 'CONSERVATIVE':
            base_pips = 35
            atr_multiplier = 1.5
        elif self.profile == 'MODERATE':
            base_pips = 50
            atr_multiplier = 2.0
        else:  # AGGRESSIVE
            base_pips = 100
            atr_multiplier = 2.5

        # If ATR is available, use it
        if atr is not None and atr > 0:
            # Convert ATR to pips
            atr_pips = atr * 10000  # Assuming 4-decimal pairs
            sl_pips = atr_pips * atr_multiplier
        else:
            sl_pips = base_pips

        # Ensure minimum SL (avoid too tight stops)
        sl_pips = max(sl_pips, 20)

        return sl_pips

    def calculate_position_size(self, account_balance, risk_amount, entry_price,
                                sl_price, pair, pip_value_per_lot = 10.0):
        """
        Calculate position size in lots based on risk amount and SL distance.

        Args:
            account_balance: Current account balance
            risk_amount: Amount willing to risk in account currency
            entry_price: Entry price
            sl_price: Stop loss price
            pair: Forex pair

        Returns:
            dict: {
                'lots': float,
                'units': int,
                'risk_amount': float,
                'pip_value': float
            }
        """
        # Calculate SL distance in pips
        sl_distance_price = abs(entry_price - sl_price)

        if 'XAU' in pair or 'GOLD' in pair or pair == 'GC=F':
            pip_size = 0.10
        elif 'JPY' in pair:
            pip_size = 0.01
        else:
            pip_size = 0.0001


        sl_distance_pips = sl_distance_price / pip_size# Convert to pips

        # Calculate required lot size
        # risk_amount = sl_distance_pips * pip_value_per_lot * lot_size
        # lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)

        required_lots = risk_amount / (sl_distance_pips * pip_value_per_lot)

        # Round to MT5 standard (0.01 lots minimum, 0.01 increments)
        lots = max(0.01, round(required_lots, 2))

        # Calculate actual risk with rounded lots
        actual_risk = lots * sl_distance_pips * pip_value_per_lot

        # Calculate units (1 lot = 100,000 units)
        units = int(lots * 100000)

        result = {
            'lots': lots,
            'units': units,
            'risk_amount': actual_risk,
            'pip_value': pip_value_per_lot * lots,
            'sl_distance_pips': sl_distance_pips
        }

        logger.debug(f"Position size: {lots} lots ({units} units), Risk: ${actual_risk:.2f}")

        return result

    def calculate_trade_levels(self, prediction, account_balance, current_price, atr=None):
        """
        Calculate complete trade setup: entry, SL, TP, position size.

        Args:
            prediction: Prediction dict from SessionPredictor
            account_balance: Current account balance
            current_price: Current market price
            atr: ATR value (optional)

        Returns:
            dict: Complete trade setup
        """
        direction = 'LONG' if prediction['direction'] == 'BULLISH' else 'SHORT'
        pair = prediction['pair']

        # Calculate risk amount
        risk_per_trade = self.config['risk_per_trade']
        risk_amount = account_balance * risk_per_trade

        if self.profile == 'CONSERVATIVE':
            base_sl_pips = 35
            atr_multiplier = 1.5
        elif self.profile == 'MODERATE':
            base_sl_pips = 50
            atr_multiplier = 2.0
        else:  # AGGRESSIVE
            base_sl_pips = 100
            atr_multiplier = 2.5

            # ===== KEY FIX: INSTRUMENT-SPECIFIC HANDLING =====

            # 1. GOLD (XAUUSD)
        if 'XAU' in pair or 'GOLD' in pair or pair == 'GC=F':
            pip_to_price = 0.10  # Gold: 1 pip = $0.10
            sl_pips = base_sl_pips  # Scale up for gold (300 pips for conservative)
            pip_value_per_lot = 1.0  # $1 per pip per lot for gold

            # 2. JPY PAIRS
        elif 'JPY' in pair:
            pip_to_price = 0.01  # JPY: 1 pip = 0.01
            sl_pips = base_sl_pips  # Normal pip count
            pip_value_per_lot = 10.0  # Standard

            # 3. NORMAL FOREX PAIRS
        else:
            pip_to_price = 0.0001  # Normal: 1 pip = 0.0001
            sl_pips = base_sl_pips  # Normal pip count
            pip_value_per_lot = 10.0  # Standard

        # Calculate TP distance using R:R ratio
        rr_ratio = self.config['rr_ratio']
        tp_pips = sl_pips * rr_ratio

        if direction == 'LONG':
            entry = current_price
            stop_loss = entry - (sl_pips * pip_to_price)
            take_profit = entry + (tp_pips * pip_to_price)
        else:  # SHORT
            entry = current_price
            stop_loss = entry + (sl_pips * pip_to_price)
            take_profit = entry - (tp_pips * pip_to_price)

        # Calculate position size
        position = self.calculate_position_size(
            account_balance=account_balance,
            risk_amount=risk_amount,
            entry_price=entry,
            sl_price=stop_loss,
            pair=pair,
            pip_value_per_lot=pip_value_per_lot
        )

        # Calculate trailing stop levels
        trailing_levels = self.calculate_trailing_levels(
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction
        )

        trade_setup = {
            'pair': pair,
            'direction': direction,
            'confidence': prediction['confidence'],
            'session': prediction['session'],
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size_lots': position['lots'],
            'position_size_units': position['units'],
            'risk_amount': position['risk_amount'],
            'risk_percentage': risk_per_trade * 100,
            'potential_profit': position['risk_amount'] * rr_ratio,
            'rr_ratio': rr_ratio,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'trailing_levels': trailing_levels,
            'profile': self.profile
        }

        return trade_setup

    def calculate_trailing_levels(self, entry, stop_loss, take_profit, direction):
        """
        Calculate trailing stop levels based on profit milestones.

        Trailing logic:
        - At 0.5R profit → Move SL to breakeven
        - At 1.0R profit → Move SL to +0.5R
        - At 1.5R profit → Move SL to +1.0R

        Args:
            entry: Entry price
            stop_loss: Initial stop loss
            take_profit: Take profit target
            direction: 'LONG' or 'SHORT'

        Returns:
            dict: Trailing stop levels
        """
        risk_distance = abs(entry - stop_loss)
        reward_distance = abs(entry - take_profit)

        if direction == 'LONG':
            levels = {
                'breakeven_trigger': entry + (risk_distance * 0.5),  # +0.5R
                'breakeven_sl': entry,

                'trail_1_trigger': entry + risk_distance,  # +1.0R
                'trail_1_sl': entry + (risk_distance * 0.5),  # Lock +0.5R

                'trail_2_trigger': entry + (risk_distance * 1.5),  # +1.5R
                'trail_2_sl': entry + risk_distance,  # Lock +1.0R
            }
        else:  # SHORT
            levels = {
                'breakeven_trigger': entry - (risk_distance * 0.5),  # +0.5R
                'breakeven_sl': entry,

                'trail_1_trigger': entry - risk_distance,  # +1.0R
                'trail_1_sl': entry - (risk_distance * 0.5),  # Lock +0.5R

                'trail_2_trigger': entry - (risk_distance * 1.5),  # +1.5R
                'trail_2_sl': entry - risk_distance,  # Lock +1.0R
            }

        return levels

    def filter_trades_by_profile(self, predictions):
        """
        Filter and select trades based on risk profile limits.

        Args:
            predictions: List of prediction dicts sorted by confidence

        Returns:
            List of selected predictions (top N based on profile)
        """
        max_trades = self.config['max_trades_per_session']
        selected = predictions[:max_trades]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TRADE SELECTION ({self.profile} profile)")
        logger.info(f"{'=' * 60}")
        logger.info(f"Available signals: {len(predictions)}")
        logger.info(f"Selected for trading: {len(selected)}")

        if selected:
            logger.info("\nSelected trades:")
            for i, pred in enumerate(selected, 1):
                logger.info(f"  {i}. {pred['pair']:.<12} {pred['direction']:.<8} {pred['confidence']:.1%}")

        return selected


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "_main_":
    # Initialize risk manager
    risk_mgr = RiskManager(profile='CONSERVATIVE')

    # Example prediction
    prediction = {
        'pair': 'EURUSD=X',
        'direction': 'BULLISH',
        'confidence': 0.87,
        'session': 'London'
    }

    # Account details
    account_balance = 100.0
    current_price = 1.0850

    # Calculate trade setup
    trade_setup = risk_mgr.calculate_trade_levels(
        prediction=prediction,
        account_balance=account_balance,
        current_price=current_price,
        atr=0.0025  # Example ATR
    )

    # Display trade setup
    print("\n" + "=" * 60)
    print("TRADE SETUP")
    print("=" * 60)
    print(f"Pair: {trade_setup['pair']}")
    print(f"Direction: {trade_setup['direction']}")
    print(f"Confidence: {trade_setup['confidence']:.1%}")
    print(f"\nEntry: {trade_setup['entry_price']:.5f}")
    print(f"Stop Loss: {trade_setup['stop_loss']:.5f} ({trade_setup['sl_pips']:.1f} pips)")
    print(f"Take Profit: {trade_setup['take_profit']:.5f} ({trade_setup['tp_pips']:.1f} pips)")
    print(f"\nPosition Size: {trade_setup['position_size_lots']} lots")
    print(f"Risk Amount: ${trade_setup['risk_amount']:.2f} ({trade_setup['risk_percentage']:.1f}%)")
    print(f"Potential Profit: ${trade_setup['potential_profit']:.2f}")
    print(f"R:R Ratio: 1:{trade_setup['rr_ratio']}")

    print(f"\nTrailing Stop Levels:")
    print(f"  Breakeven trigger: {trade_setup['trailing_levels']['breakeven_trigger']:.5f}")
    print(f"  Trail 1 trigger: {trade_setup['trailing_levels']['trail_1_trigger']:.5f}")
    print(f"  Trail 2 trigger: {trade_setup['trailing_levels']['trail_2_trigger']:.5f}")