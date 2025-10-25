import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from src.utils.paths import  ALL_CLEANED_CSV,CONFLUENCE_CSV,ALL_Stocluster_CSV,RECOMMEND_PORTFOLIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolio allocation based on signals, risk profiles, and constraints.
    Implements stop-loss calculation and position sizing.
    """

    def __init__(self, total_capital: float = 10000):
        self.total_capital = total_capital
        self.min_position_size = 0.05  # Minimum 5% per position
        self.max_position_size = 0.40  # Maximum 40% per position
        self.cash_buffer = 0.10  # Keep 10% in cash minimum

    def calculate_position_size(self,
                                signal_strength: float,
                                risk_label: str,
                                volatility: float) -> float:
        """
        Calculate position size based on signal strength and risk.

        Returns: Allocation percentage (0.0 to 1.0)
        """
        # Base allocation on signal confidence
        base_allocation = signal_strength

        # Adjust for risk profile
        risk_multipliers = {
            'Low Risk': 1.2,
            'Medium Risk': 1.0,
            'Medium High Risk': 0.7,
            'High Risk': 0.5
        }

        risk_multiplier = risk_multipliers.get(risk_label, 1.0)

        # Adjust for volatility (inverse relationship)
        # Higher volatility = smaller position
        volatility_adjustment = 1 / (1 + volatility)

        # Calculate final position size
        position_size = base_allocation * risk_multiplier * volatility_adjustment

        # Apply constraints
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))

        return position_size

    def calculate_stop_loss(self,
                            current_price: float,
                            method: str = 'atr',
                            atr_value: float = None,
                            max_drawdown: float = None,
                            atr_multiplier: float = 2.0) -> Dict:
        """
        Calculate stop-loss price and risk amount.

        Methods:
        - 'atr': Based on Average True Range (volatility-based)
        - 'drawdown': Based on historical maximum drawdown

        Returns dict with stop_loss_price, risk_per_share, risk_percentage
        """
        if method == 'atr' and atr_value is not None:
            # ATR-based stop loss
            stop_loss_price = current_price - (atr_value * atr_multiplier)
            risk_per_share = atr_value * atr_multiplier

        elif method == 'drawdown' and max_drawdown is not None:
            # Historical drawdown-based stop loss
            # max_drawdown is negative (e.g., -0.0737 for -7.37%)
            stop_loss_price = current_price * (1 + max_drawdown)  # Note: adding negative value
            risk_per_share = current_price - stop_loss_price

        else:
            # Default: 5% stop loss
            stop_loss_price = current_price * 0.95
            risk_per_share = current_price * 0.05

        risk_percentage = (risk_per_share / current_price) * 100

        return {
            'stop_loss_price': round(stop_loss_price, 2),
            'risk_per_share': round(risk_per_share, 2),
            'risk_percentage': round(risk_percentage, 2)
        }

    def calculate_position_shares(self,
                                  allocation_amount: float,
                                  current_price: float,
                                  stop_loss_price: float,
                                  max_risk_per_trade: float = 0.02) -> Dict:
        """
        Calculate number of shares to buy based on risk management.

        Uses fixed fractional position sizing:
        - Never risk more than max_risk_per_trade (default 2%) of total capital per trade
        """
        # Maximum dollar amount we're willing to lose on this trade
        max_loss_dollars = self.total_capital * max_risk_per_trade

        # Risk per share (distance to stop loss)
        risk_per_share = current_price - stop_loss_price

        if risk_per_share <= 0:
            logger.warning("Stop loss is above current price!")
            return None

        # Position size based on risk
        risk_based_shares = max_loss_dollars / risk_per_share

        # Position size based on allocation
        allocation_based_shares = allocation_amount / current_price

        # Take the MINIMUM (most conservative)
        shares_to_buy = min(risk_based_shares, allocation_based_shares)
        shares_to_buy = int(shares_to_buy)  # Round down to whole shares

        # Calculate actual amounts
        actual_cost = shares_to_buy * current_price
        actual_risk = shares_to_buy * risk_per_share

        return {
            'shares': shares_to_buy,
            'cost': round(actual_cost, 2),
            'actual_risk_dollars': round(actual_risk, 2),
            'actual_risk_percent': round((actual_risk / self.total_capital) * 100, 2)
        }

    def optimize_portfolio(self,
                           actionable_signals: pd.DataFrame,
                           risk_profiles: pd.DataFrame,
                           features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create optimized portfolio allocation from actionable signals.

        Returns DataFrame with position details, stop losses, and risk metrics.
        """
        # Merge signals with risk profiles
        portfolio_data = actionable_signals.merge(
            risk_profiles[['symbol', 'risk_label', 'avg_volatility', 'max_drawdown']],
            on='symbol',
            how='left'
        )

        # Get latest ATR for each symbol
        latest_features = features_df.sort_values('date').groupby('symbol').tail(1)
        portfolio_data = portfolio_data.merge(
            latest_features[['symbol', 'atr']],
            on='symbol',
            how='left'
        )

        positions = []

        for _, row in portfolio_data.iterrows():
            # Calculate position size
            position_pct = self.calculate_position_size(
                signal_strength=row['confidence_score'],
                risk_label=row['risk_label'],
                volatility=row['avg_volatility']
            )

            allocation_amount = self.total_capital * position_pct

            # Calculate stop loss (using ATR method)
            stop_loss_info = self.calculate_stop_loss(
                current_price=row['close_price'],
                method='atr',
                atr_value=row['atr'],
                atr_multiplier=2.0
            )

            # Calculate shares to buy
            position_info = self.calculate_position_shares(
                allocation_amount=allocation_amount,
                current_price=row['close_price'],
                stop_loss_price=stop_loss_info['stop_loss_price'],
                max_risk_per_trade=0.02  # 2% max risk per trade
            )

            if position_info is None:
                continue

            positions.append({
                'symbol': row['symbol'],
                'signal': row['overall_signal'],
                'confidence': row['confidence'],
                'confidence_score': row['confidence_score'],
                'risk_label': row['risk_label'],
                'current_price': row['close_price'],
                'allocation_pct': round(position_pct * 100, 2),
                'shares': position_info['shares'],
                'total_cost': position_info['cost'],
                'stop_loss_price': stop_loss_info['stop_loss_price'],
                'stop_loss_distance_pct': stop_loss_info['risk_percentage'],
                'risk_per_trade_dollars': position_info['actual_risk_dollars'],
                'risk_per_trade_pct': position_info['actual_risk_percent']
            })

        portfolio_df = pd.DataFrame(positions)

        if portfolio_df.empty:
            logger.warning("No valid positions to allocate")
            return portfolio_df

        # Normalize allocations to not exceed available capital
        total_cost = portfolio_df['total_cost'].sum()
        available_capital = self.total_capital * (1 - self.cash_buffer)

        if total_cost > available_capital:
            # Scale down proportionally
            scale_factor = available_capital / total_cost
            portfolio_df['shares'] = (portfolio_df['shares'] * scale_factor).astype(int)
            portfolio_df['total_cost'] = portfolio_df['shares'] * portfolio_df['current_price']
            portfolio_df['risk_per_trade_dollars'] = (
                    portfolio_df['shares'] *
                    (portfolio_df['current_price'] - portfolio_df['stop_loss_price'])
            )

        # Recalculate allocation percentages
        portfolio_df['allocation_pct'] = (portfolio_df['total_cost'] / self.total_capital * 100).round(2)

        # Sort by allocation
        portfolio_df = portfolio_df.sort_values('allocation_pct', ascending=False)

        return portfolio_df

    def generate_portfolio_summary(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the portfolio.
        """
        if portfolio_df.empty:
            return {
                'total_invested': 0,
                'cash_remaining': self.total_capital,
                'number_of_positions': 0,
                'total_portfolio_risk': 0,
                'largest_position': None,
                'average_position_size': 0
            }

        total_invested = portfolio_df['total_cost'].sum()
        cash_remaining = self.total_capital - total_invested
        total_risk = portfolio_df['risk_per_trade_dollars'].sum()

        return {
            'total_capital': self.total_capital,
            'total_invested': round(total_invested, 2),
            'cash_remaining': round(cash_remaining, 2),
            'cash_remaining_pct': round((cash_remaining / self.total_capital) * 100, 2),
            'number_of_positions': len(portfolio_df),
            'total_portfolio_risk_dollars': round(total_risk, 2),
            'total_portfolio_risk_pct': round((total_risk / self.total_capital) * 100, 2),
            'largest_position': portfolio_df.iloc[0]['symbol'],
            'largest_position_pct': portfolio_df.iloc[0]['allocation_pct'],
            'average_position_size': round(portfolio_df['allocation_pct'].mean(), 2)
        }


# ============ EXAMPLE USAGE ============

def main():
    # Load required data
    print("Loading data...")
    features_df = pd.read_csv(ALL_CLEANED_CSV)
    features_df['date'] = pd.to_datetime(features_df['date'])

    actionable_signals = pd.read_csv(CONFLUENCE_CSV)
    actionable_signals = actionable_signals[
        (actionable_signals['confidence'] != 'LOW') &
        (actionable_signals['overall_signal'] != 'neutral')
        ]

    # Load risk profiles from clustering
    risk_profiles = pd.read_csv(ALL_Stocluster_CSV)

    # Initialize optimizer
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    print(f"Total Capital: $10,000")

    optimizer = PortfolioOptimizer(total_capital=10000)

    # Generate optimal portfolio
    print("\nOptimizing portfolio allocation...")
    portfolio = optimizer.optimize_portfolio(actionable_signals, risk_profiles, features_df)

    # Display results
    print("\n" + "=" * 60)
    print("RECOMMENDED PORTFOLIO ALLOCATION")
    print("=" * 60)

    if portfolio.empty:
        print("No positions recommended at this time.")
    else:
        display_cols = ['symbol', 'signal', 'risk_label', 'shares', 'current_price',
                        'total_cost', 'allocation_pct', 'stop_loss_price',
                        'risk_per_trade_dollars', 'risk_per_trade_pct']
        print(portfolio[display_cols].to_string(index=False))

    # Portfolio summary
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    summary = optimizer.generate_portfolio_summary(portfolio)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Save portfolio
    portfolio.to_csv(RECOMMEND_PORTFOLIO, index=False)
    print("\n✅ Portfolio saved to 'recommended_portfolio.csv'")

    # Trading instructions
    print("\n" + "=" * 60)
    print("TRADING INSTRUCTIONS")
    print("=" * 60)
    for _, pos in portfolio.iterrows():
        print(f"\n{pos['symbol']} ({pos['signal'].upper()}):")
        print(f"  Action: {'BUY' if pos['signal'] == 'bullish' else 'SHORT/SELL'} {pos['shares']} shares")
        print(f"  Entry Price: ${pos['current_price']}")
        print(f"  Stop Loss: ${pos['stop_loss_price']} ({pos['stop_loss_distance_pct']}% risk)")
        print(f"  Position Size: ${pos['total_cost']} ({pos['allocation_pct']}% of portfolio)")
        print(f"  Max Loss: ${pos['risk_per_trade_dollars']} ({pos['risk_per_trade_pct']}% of total capital)")