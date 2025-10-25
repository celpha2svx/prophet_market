import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import json
from src.utils.paths import ALL_CLEANED_CSV,RECOMMEND_CSV
from src.utils.paths import RECOMMEND_JSON,CONFLUENCE_CSV,PROPHET_METRICS_JSON,STOCK_PREDICTABILY
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generates actionable trade recommendations by combining:
    - Prophet forecasts (predictive accuracy)
    - Signal confluence (technical confirmation)
    - Risk profiles (position sizing)
    - Portfolio constraints (diversification)
    """

    def __init__(self,
               total_capital: float = 10000,
               min_prophet_accuracy: float = 70.0,
               min_signal_confidence: str = 'MEDIUM'):
        """
        Initialize recommendation engine.

        Args:
            total_capital: Total capital to allocate
            min_prophet_accuracy: Minimum Prophet directional accuracy (%)
            min_signal_confidence: Minimum signal confidence level
        """
        self.total_capital = total_capital
        self.min_prophet_accuracy = min_prophet_accuracy
        self.min_signal_confidence = min_signal_confidence
        self.confidence_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}

    def load_prophet_predictions(self, prophet_metrics_file: str = PROPHET_METRICS_JSON) -> pd.DataFrame:
        """Load Prophet model predictions and metrics."""
        try:
            with open(prophet_metrics_file, 'r') as f:
                metrics = json.load(f)

            # Convert to DataFrame
            rows = []
            for symbol, data in metrics.items():
                row = {'symbol': symbol}
                row.update(data['metrics'])
                rows.append(row)

            df = pd.DataFrame(rows)
            logger.info(f"Loaded Prophet metrics for {len(df)} stocks")
            return df

        except FileNotFoundError:
            logger.error(f"Prophet metrics file not found: {prophet_metrics_file}")
            return pd.DataFrame()

    def load_signal_confluence(self, signals_file: str = CONFLUENCE_CSV) -> pd.DataFrame:
        """Load signal confluence results."""
        try:
            df = pd.read_csv(signals_file)
            # Get latest signal per stock
            df['date'] = pd.to_datetime(df['date'])
            latest_signals = df.sort_values('date').groupby('symbol').tail(1)
            logger.info(f"Loaded signals for {len(latest_signals)} stocks")
            return latest_signals

        except FileNotFoundError:
            logger.error(f"Signal confluence file not found: {signals_file}")
            return pd.DataFrame()

    def load_risk_profiles(self, leaderboard_file: str = STOCK_PREDICTABILY) -> pd.DataFrame:
        """Load stock risk profiles and predictability."""
        try:
            df = pd.read_csv(leaderboard_file)
            logger.info(f"Loaded risk profiles for {len(df)} stocks")
            return df

        except FileNotFoundError:
            logger.error(f"Leaderboard file not found: {leaderboard_file}")
            return pd.DataFrame()

    def calculate_combined_confidence(self,
                                      prophet_accuracy: float,
                                      signal_confidence: str,
                                      signal_agreement: str) -> Tuple[float, str]:
        """
        Calculate combined confidence score from Prophet + Signals.

        Returns:
            (confidence_score, confidence_label)
        """
        # Prophet contribution (0-1 scale)
        prophet_score = prophet_accuracy / 100.0

        # Signal contribution (0-1 scale)
        signal_level = self.confidence_levels.get(signal_confidence, 0)
        signal_score = signal_level / 2.0  # Normalize to 0-1

        # Agreement bonus
        agreement_num = int(signal_agreement.split('/')[0])
        agreement_score = agreement_num / 3.0

        # Combined score (weighted average)
        combined = (prophet_score * 0.5 +  # 50% weight on Prophet
                    signal_score * 0.3 +  # 30% weight on signal confidence
                    agreement_score * 0.2)  # 20% weight on agreement

        # Label
        if combined >= 0.75:
            label = 'VERY HIGH'
        elif combined >= 0.65:
            label = 'HIGH'
        elif combined >= 0.55:
            label = 'MEDIUM'
        else:
            label = 'LOW'

        return round(combined, 3), label

    def generate_recommendations(self,
                                 prophet_df: pd.DataFrame,
                                 signals_df: pd.DataFrame,
                                 risk_df: pd.DataFrame,
                                 features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trade recommendations by combining all data sources.
        """
        recommendations = []

        # Merge all data sources
        merged = prophet_df.merge(signals_df, on='symbol', how='inner')
        merged = merged.merge(risk_df[['symbol', 'directional_accuracy', 'mape']],
                              on='symbol', how='inner', suffixes=('_prophet', '_risk'))

        # Use the correct column after merge (handle suffix)
        accuracy_col = 'directional_accuracy' if 'directional_accuracy' in merged.columns else 'directional_accuracy_risk'
        merged = merged[merged[accuracy_col] >= self.min_prophet_accuracy]

        # Filter by signal confidence
        min_conf_level = self.confidence_levels[self.min_signal_confidence]
        merged['signal_conf_level'] = merged['confidence'].map(self.confidence_levels)
        merged = merged[merged['signal_conf_level'] >= min_conf_level]

        for _, row in merged.iterrows():
            symbol = row['symbol']

            # Get latest price from features
            stock_features = features_df[features_df['symbol'] == symbol].sort_values('date').tail(1)
            if stock_features.empty:
                continue

            current_price = stock_features['close'].iloc[0]
            atr = stock_features['atr'].iloc[0]

            # Calculate combined confidence
            prophet_acc = row.get('directional_accuracy',
                                  row.get('directional_accuracy_risk', row.get('directional_accuracy_prophet', 0)))
            combined_conf, conf_label = self.calculate_combined_confidence(
                prophet_acc,
                row['confidence'],
                row['agreement']
            )

            # Determine action
            if row['overall_signal'] == 'bullish':
                action = 'BUY'
            elif row['overall_signal'] == 'bearish':
                action = 'SELL/SHORT'
            else:
                continue  # Skip neutral

            # Calculate position size (based on combined confidence)
            position_pct = self._calculate_position_size(combined_conf, row[accuracy_col])
            position_value = self.total_capital * position_pct
            shares = int(position_value / current_price)

            if shares == 0:
                continue

            # Calculate stop loss (2x ATR)
            stop_loss = current_price - (2 * atr) if action == 'BUY' else current_price + (2 * atr)

            # Calculate target (based on predicted movement)
            # Estimate 1-2% move for conservative target
            target_pct = 0.015 if conf_label in ['MEDIUM', 'HIGH'] else 0.02
            target_price = current_price * (1 + target_pct) if action == 'BUY' else current_price * (1 - target_pct)

            # Risk calculation
            risk_per_share = abs(current_price - stop_loss)
            total_risk = risk_per_share * shares
            risk_pct = (total_risk / self.total_capital) * 100

            recommendation = {
                'symbol': symbol,
                'action': action,
                'current_price': round(current_price, 2),
                'shares': shares,
                'position_value': round(shares * current_price, 2),
                'position_pct': round(position_pct * 100, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target_price, 2),
                'risk_dollars': round(total_risk, 2),
                'risk_pct': round(risk_pct, 2),
                'prophet_accuracy': prophet_acc,
                'signal_confidence': row['confidence'],
                'signal_agreement': row['agreement'],
                'combined_confidence': combined_conf,
                'confidence_label': conf_label,
                'rsi_signal': row['rsi_signal'],
                'macd_signal': row['macd_signal'],
                'sma_signal': row['sma_signal']
            }

            recommendations.append(recommendation)

        # Convert to DataFrame and sort by combined confidence
        rec_df = pd.DataFrame(recommendations)

        if rec_df.empty:
            logger.warning("No recommendations generated!")
            return rec_df

        rec_df = rec_df.sort_values('combined_confidence', ascending=False)

        # Ensure total allocation doesn't exceed 90% of capital
        cumulative_pct = rec_df['position_pct'].cumsum()
        rec_df = rec_df[cumulative_pct <= 90]

        logger.info(f"Generated {len(rec_df)} recommendations")
        return rec_df

    def _calculate_position_size(self, combined_conf: float, prophet_acc: float) -> float:
        """
        Calculate position size as percentage of capital.
        Higher confidence = Larger position (with limits)
        """
        # Base allocation on combined confidence
        base_pct = combined_conf * 0.15  # Max 15% per position

        # Boost for very high Prophet accuracy
        if prophet_acc >= 80:
            base_pct *= 1.2
        elif prophet_acc >= 75:
            base_pct *= 1.1

        # Apply limits
        min_pct = 0.02  # Minimum 2%
        max_pct = 0.15  # Maximum 15%

        return max(min_pct, min(base_pct, max_pct))

    def generate_portfolio_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Generate summary statistics for the recommended portfolio."""
        if recommendations.empty:
            return {
                'total_positions': 0,
                'total_allocated': 0,
                'total_risk': 0,
                'avg_confidence': 0,
                'buy_count': 0,
                'sell_count': 0
            }

        return {
            'total_positions': len(recommendations),
            'total_allocated': recommendations['position_value'].sum(),
            'total_allocated_pct': recommendations['position_pct'].sum(),
            'cash_remaining': self.total_capital - recommendations['position_value'].sum(),
            'cash_remaining_pct': 100 - recommendations['position_pct'].sum(),
            'total_risk': recommendations['risk_dollars'].sum(),
            'total_risk_pct': recommendations['risk_pct'].sum(),
            'avg_confidence': recommendations['combined_confidence'].mean(),
            'avg_prophet_accuracy': recommendations['prophet_accuracy'].mean(),
            'buy_count': len(recommendations[recommendations['action'] == 'BUY']),
            'sell_count': len(recommendations[recommendations['action'] == 'SELL/SHORT']),
            'highest_confidence': recommendations['combined_confidence'].max(),
            'top_recommendation': recommendations.iloc[0]['symbol']
        }


# ============ EXAMPLE USAGE ============

def main():
    print("=" * 60)
    print("GENERATING DAILY TRADE RECOMMENDATIONS")
    print("=" * 60)

    # Initialize engine
    engine = RecommendationEngine(
        total_capital=10000,
        min_prophet_accuracy=70.0,  # Only stocks with >70% accuracy
        min_signal_confidence='MEDIUM'
    )

    # Load all data sources
    print("\nLoading data sources...")
    prophet_df = engine.load_prophet_predictions()
    signals_df = engine.load_signal_confluence()
    risk_df = engine.load_risk_profiles()

    # Load features for current prices
    features_df = pd.read_csv(ALL_CLEANED_CSV)
    features_df['date'] = pd.to_datetime(features_df['date'], utc=True)

    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = engine.generate_recommendations(
        prophet_df, signals_df, risk_df, features_df
    )

    if recommendations.empty:
        print("\n⚠ No recommendations meet the criteria today.")
        print("Try lowering min_prophet_accuracy or min_signal_confidence.")
    else:
        # Display recommendations
        print("\n" + "=" * 60)
        print(f"📊 TRADE RECOMMENDATIONS - {datetime.now().strftime('%B %d, %Y')}")
        print("=" * 60)

        print(f"\n🏆 HIGH CONFIDENCE TRADES (Showing top 10)")
        print("-" * 60)

        display_cols = ['symbol', 'action', 'shares', 'current_price',
                        'target_price', 'stop_loss', 'position_value',
                        'confidence_label', 'prophet_accuracy']

        print(recommendations[display_cols].head(10).to_string(index=False))

        # Portfolio summary
        summary = engine.generate_portfolio_summary(recommendations)

        print("\n" + "=" * 60)
        print("📈 PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Total Capital: ${engine.total_capital:,.2f}")
        print(f"Total Positions: {summary['total_positions']}")
        print(f"Total Allocated: ${summary['total_allocated']:,.2f} ({summary['total_allocated_pct']:.1f}%)")
        print(f"Cash Remaining: ${summary['cash_remaining']:,.2f} ({summary['cash_remaining_pct']:.1f}%)")
        print(f"Total Portfolio Risk: ${summary['total_risk']:,.2f} ({summary['total_risk_pct']:.2f}%)")
        print(f"Average Confidence: {summary['avg_confidence']:.3f}")
        print(f"Average Prophet Accuracy: {summary['avg_prophet_accuracy']:.1f}%")
        print(f"Buy Signals: {summary['buy_count']} | Sell Signals: {summary['sell_count']}")
        print(f"\n🌟 Top Recommendation: {summary['top_recommendation']}")

        # Save to CSV
        recommendations.to_csv(RECOMMEND_CSV, index=False)
        print("\n✅ Recommendations saved to: daily_recommendations.csv")

        # Save summary to JSON
        summary['date'] = datetime.now().isoformat()
        with open(RECOMMEND_JSON, 'w') as f:
            json.dump(summary, f, indent=2)
        print("✅ Summary saved to: recommendation_summary.json")

    print("\n" + "=" * 60)
    print("⚠ DISCLAIMER")
    print("=" * 60)
    print("This is for educational purposes only.")
    print("Not financial advice. Trade at your own risk.")
    print("Past performance does not guarantee future results.")
    print("=" * 60)