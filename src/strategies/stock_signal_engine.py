import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import json
import os
from src.utils.paths import ALL_CLEANED_CSV, ALL_Stocluster_CSV, CONFLUENCE_CSV


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalConfluenceEngine:
    """
    Analyzes multiple technical indicators and generates trading signals
    with confidence scores based on indicator agreement, stock risk profile,
    and news sentiment.
    """

    def __init__(self, risk_profiles: pd.DataFrame, news_summary_path: str = None):
        """
        Initialize with risk profiles from clustering and optional news sentiment.

        Args:
            risk_profiles: DataFrame with columns ['symbol', 'risk_label', 'cluster']
            news_summary_path: Path to news_summary.json (optional)
        """
        self.risk_profiles = risk_profiles.set_index('symbol')
        self.news_sentiment = self._load_news_sentiment(news_summary_path)

        # Thresholds for each indicator
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_neutral_low = 40
        self.rsi_neutral_high = 60

    def _load_news_sentiment(self, news_path: str = "filess_csv/news_enriched.csv") -> Dict:
        if not news_path or not os.path.exists(news_path):
            logger.warning("No news sentiment file found. Signals will not include news.")
            return {}

        try:
            df = pd.read_csv(news_path)
            sentiment_map = {}

            sentiment_scores = {'bullish': 1.0, 'neutral': 0.0, 'bearish': -1.0}
            impact_weights = {'High': 1.0, 'Medium': 0.6, 'Low': 0.3}

            for _, row in df.iterrows():
                tickers = [row['symbol']] if pd.notna(row['symbol']) else []
                sentiment_label = str(row['sentiment']).lower() if pd.notna(row['sentiment']) else 'neutral'
                impact_label = row['impact'] if pd.notna(row['impact']) else 'Low'
                title = row['title'] if pd.notna(row['title']) else ''

                sentiment_score = sentiment_scores.get(sentiment_label, 0.0)
                impact_weight = impact_weights.get(impact_label, 0.5)
                weighted_score = sentiment_score * impact_weight

                for ticker in tickers:
                    if ticker not in sentiment_map:
                        sentiment_map[ticker] = {'scores': [], 'items': []}
                    sentiment_map[ticker]['scores'].append(weighted_score)
                    sentiment_map[ticker]['items'].append({
                        'title': title,
                        'sentiment': sentiment_label,
                        'impact': impact_label
                    })

            for symbol in sentiment_map:
                scores = sentiment_map[symbol]['scores']
                sentiment_map[symbol]['avg_score'] = np.mean(scores) if scores else 0.0
                sentiment_map[symbol]['news_count'] = len(scores)

            logger.info(f"Loaded news sentiment for {len(sentiment_map)} symbols from CSV")
            return sentiment_map

        except Exception as e:
            logger.error(f"Error loading news sentiment from CSV: {e}")
            return {}

    def get_news_sentiment(self, symbol: str) -> Tuple[float, int]:
        """
        Get news sentiment score for a symbol.

        Returns:
            Tuple of (sentiment_score, news_count)
            sentiment_score: -1.0 (bearish) to 1.0 (bullish)
            news_count: number of news items
        """
        if symbol not in self.news_sentiment:
            return 0.0, 0

        return (
            self.news_sentiment[symbol]['avg_score'],
            self.news_sentiment[symbol]['news_count']
        )

    def analyze_rsi_signal(self, rsi_value: float) -> Tuple[str, float]:
        """
        Analyze RSI and return signal direction and strength.

        Returns:
            Tuple of (signal, strength) where:
            - signal: 'bullish', 'bearish', 'neutral'
            - strength: 0.0 to 1.0
        """
        if pd.isna(rsi_value):
            return 'neutral', 0.0

        if rsi_value < self.rsi_oversold:
            # Strong oversold - bullish signal
            strength = (self.rsi_oversold - rsi_value) / self.rsi_oversold
            return 'bullish', min(strength, 1.0)

        elif rsi_value > self.rsi_overbought:
            # Strong overbought - bearish signal
            strength = (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought)
            return 'bearish', min(strength, 1.0)

        elif self.rsi_neutral_low < rsi_value < self.rsi_neutral_high:
            # Neutral zone
            return 'neutral', 0.0

        elif rsi_value < self.rsi_neutral_low:
            # Mild oversold
            strength = (self.rsi_neutral_low - rsi_value) / (self.rsi_neutral_low - self.rsi_oversold)
            return 'bullish', strength * 0.5  # Weaker signal

        else:  # rsi_value > self.rsi_neutral_high
            # Mild overbought
            strength = (rsi_value - self.rsi_neutral_high) / (self.rsi_overbought - self.rsi_neutral_high)
            return 'bearish', strength * 0.5  # Weaker signal

    def analyze_macd_signal(self, macd: float, macd_signal: float, macd_histogram: float) -> Tuple[str, float]:
        """
        Analyze MACD and return signal direction and strength.
        """
        if pd.isna(macd) or pd.isna(macd_signal):
            return 'neutral', 0.0

        # Check if MACD is above or below signal line
        if macd > macd_signal and macd_histogram > 0:
            # Bullish: MACD above signal
            strength = min(abs(macd_histogram) / abs(macd) if macd != 0 else 0.5, 1.0)
            return 'bullish', strength

        elif macd < macd_signal and macd_histogram < 0:
            # Bearish: MACD below signal
            strength = min(abs(macd_histogram) / abs(macd) if macd != 0 else 0.5, 1.0)
            return 'bearish', strength

        else:
            return 'neutral', 0.0

    def analyze_sma_signal(self, close_price: float, sma_20: float, sma_50: float) -> Tuple[str, float]:
        """
        Analyze moving average crossovers and price position.
        """
        if pd.isna(close_price) or pd.isna(sma_20) or pd.isna(sma_50):
            return 'neutral', 0.0

        # Check if price is above/below moving averages
        price_vs_sma20 = (close_price - sma_20) / sma_20
        price_vs_sma50 = (close_price - sma_50) / sma_50
        sma_cross = (sma_20 - sma_50) / sma_50

        if close_price > sma_20 > sma_50:
            # Strong bullish: Price above both, golden cross
            strength = min(abs(sma_cross) * 10, 1.0)  # Scale the crossover
            return 'bullish', max(strength, 0.6)  # Minimum 0.6 for this pattern

        elif close_price < sma_20 < sma_50:
            # Strong bearish: Price below both, death cross
            strength = min(abs(sma_cross) * 10, 1.0)
            return 'bearish', max(strength, 0.6)

        elif close_price > sma_20:
            # Mild bullish: Price above short-term MA
            strength = min(abs(price_vs_sma20) * 5, 1.0)
            return 'bullish', strength * 0.5

        elif close_price < sma_20:
            # Mild bearish: Price below short-term MA
            strength = min(abs(price_vs_sma20) * 5, 1.0)
            return 'bearish', strength * 0.5

        else:
            return 'neutral', 0.0

    def calculate_confluence(self, row: pd.Series) -> Dict:
        """
        Calculate confluence score for a single data point (row).
        NOW INCLUDES NEWS SENTIMENT INTEGRATION.

        Returns a dict with signal analysis results.
        """
        symbol = row['symbol']

        # Get individual indicator signals
        rsi_signal, rsi_strength = self.analyze_rsi_signal(row['rsi'])
        macd_signal, macd_strength = self.analyze_macd_signal(
            row['macd'], row['macd_signal'], row['macd_histogram']
        )
        sma_signal, sma_strength = self.analyze_sma_signal(
            row['close'], row['sma_20'], row['sma_50']
        )

        # Get news sentiment
        news_score, news_count = self.get_news_sentiment(symbol)
        news_signal = 'bullish' if news_score > 0.2 else ('bearish' if news_score < -0.2 else 'neutral')

        # Count agreements
        signals = [rsi_signal, macd_signal, sma_signal]
        strengths = [rsi_strength, macd_strength, sma_strength]

        # Determine overall direction
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')
        neutral_count = signals.count('neutral')

        if bullish_count > bearish_count:
            overall_signal = 'bullish'
            agreement_count = bullish_count
        elif bearish_count > bullish_count:
            overall_signal = 'bearish'
            agreement_count = bearish_count
        else:
            overall_signal = 'neutral'
            agreement_count = 0

        # Calculate base confidence score
        if agreement_count == 3:
            confidence = 'HIGH'
            confidence_score = np.mean([s for s, sig in zip(strengths, signals) if sig == overall_signal])
        elif agreement_count == 2:
            confidence = 'MEDIUM'
            confidence_score = np.mean([s for s, sig in zip(strengths, signals) if sig == overall_signal]) * 0.7
        else:
            confidence = 'LOW'
            confidence_score = 0.3

        # ============ NEWS SENTIMENT ADJUSTMENT ============
        if news_count > 0:
            # Check if news aligns with technical signals
            if overall_signal == news_signal and overall_signal != 'neutral':
                # BOOST: News confirms technical signal
                boost = min(abs(news_score) * 0.15, 0.15)  # Max 15% boost
                confidence_score = min(confidence_score * (1 + boost), 1.0)

                # Upgrade confidence if strong alignment
                if news_count >= 3 and abs(news_score) > 0.5:
                    if confidence == 'MEDIUM':
                        confidence = 'HIGH'

                logger.debug(f"{symbol}: News CONFIRMS {overall_signal} (+{boost:.1%})")

            elif (overall_signal == 'bullish' and news_signal == 'bearish') or \
                    (overall_signal == 'bearish' and news_signal == 'bullish'):
                # REDUCE: News contradicts technical signal
                penalty = min(abs(news_score) * 0.12, 0.12)  # Max 12% penalty
                confidence_score = max(confidence_score * (1 - penalty), 0.1)

                # Downgrade confidence if strong contradiction
                if news_count >= 3 and abs(news_score) > 0.5:
                    if confidence == 'HIGH':
                        confidence = 'MEDIUM'
                    elif confidence == 'MEDIUM':
                        confidence = 'LOW'

                logger.debug(f"{symbol}: News CONTRADICTS {overall_signal} (-{penalty:.1%})")

        # Adjust for risk profile
        if symbol in self.risk_profiles.index:
            risk_label = self.risk_profiles.loc[symbol, 'risk_label']

            # High risk stocks need stronger confirmation
            if risk_label in ['High Risk', 'Medium High Risk']:
                if agreement_count < 3:
                    confidence = 'LOW'
                    confidence_score *= 0.6

            # Low risk stocks can trade on medium signals
            elif risk_label == 'Low Risk':
                if agreement_count == 2:
                    confidence_score *= 1.2  # Boost confidence for stable stocks

        return {
            'symbol': symbol,
            'date': row['date'],
            'overall_signal': overall_signal,
            'confidence': confidence,
            'confidence_score': round(min(confidence_score, 1.0), 3),
            'agreement': f"{agreement_count}/3",
            'rsi_signal': rsi_signal,
            'rsi_strength': round(rsi_strength, 3),
            'macd_signal': macd_signal,
            'macd_strength': round(macd_strength, 3),
            'sma_signal': sma_signal,
            'sma_strength': round(sma_strength, 3),
            'news_sentiment': round(news_score, 3) if news_count > 0 else None,
            'news_count': news_count,
            'close_price': round(row['close'], 2)
        }

    def analyze_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all rows in features DataFrame and return signals.
        """
        results = []

        for idx, row in features_df.iterrows():
            result = self.calculate_confluence(row)
            results.append(result)

        signals_df = pd.DataFrame(results)
        logger.info(f"Analyzed {len(signals_df)} data points")

        return signals_df

    def get_latest_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the most recent signal for each stock.
        """
        # Analyze all data
        all_signals = self.analyze_features(features_df)

        # Get latest date for each symbol
        latest_signals = all_signals.sort_values('date').groupby('symbol').tail(1)

        return latest_signals.reset_index(drop=True)

    def get_actionable_signals(self, features_df: pd.DataFrame, min_confidence: str = 'MEDIUM') -> pd.DataFrame:
        """
        Filter for actionable signals based on minimum confidence.

        Args:
            min_confidence: 'LOW', 'MEDIUM', or 'HIGH'
        """
        latest_signals = self.get_latest_signals(features_df)

        # Filter by confidence
        confidence_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        min_level = confidence_levels[min_confidence]

        actionable = latest_signals[
            latest_signals['confidence'].map(confidence_levels) >= min_level
            ]

        # Remove neutral signals
        actionable = actionable[actionable['overall_signal'] != 'neutral']

        return actionable.sort_values('confidence_score', ascending=False)


# ============ EXAMPLE USAGE ============

def main():
    # Load features from Phase A2
    print("Loading features...")
    features_df = pd.read_csv(ALL_CLEANED_CSV)
    features_df['date'] = pd.to_datetime(features_df['date'])

    # Load clustering results from Part 1
    print("Loading clustering results...")
    risk_profiles = pd.read_csv(ALL_Stocluster_CSV)

    print("\nRisk Profiles:")
    print(risk_profiles)

    # Initialize Signal Confluence Engine WITH NEWS
    print("\n" + "=" * 60)
    print("INITIALIZING SIGNAL CONFLUENCE ENGINE (WITH NEWS)")
    print("=" * 60)

    # Path to news summary JSON
    engine = SignalConfluenceEngine(risk_profiles)

    # Get latest signals for all stocks
    print("\nAnalyzing latest signals...")
    latest_signals = engine.get_latest_signals(features_df)

    print("\n" + "=" * 60)
    print("LATEST SIGNALS FOR ALL STOCKS (WITH NEWS SENTIMENT)")
    print("=" * 60)
    print(latest_signals[['symbol', 'overall_signal', 'confidence', 'confidence_score',
                          'agreement', 'news_sentiment', 'news_count', 'close_price']].to_string(index=False))

    # Get actionable signals (MEDIUM confidence or higher)
    print("\n" + "=" * 60)
    print("ACTIONABLE SIGNALS (Medium+ Confidence)")
    print("=" * 60)
    actionable = engine.get_actionable_signals(features_df, min_confidence='MEDIUM')

    if actionable.empty:
        print("No actionable signals at this time.")
    else:
        print(actionable[['symbol', 'overall_signal', 'confidence', 'confidence_score',
                          'agreement', 'rsi_signal', 'macd_signal', 'sma_signal',
                          'news_sentiment', 'news_count', 'close_price']].to_string(index=False))

    # Show detailed breakdown for one stock
    print("\n" + "=" * 60)
    print("DETAILED SIGNAL BREAKDOWN - AAPL")
    print("=" * 60)
    aapl_signal = latest_signals[latest_signals['symbol'] == 'AAPL'].iloc[0]
    print(f"Symbol: {aapl_signal['symbol']}")
    print(f"Date: {aapl_signal['date']}")
    print(f"Close Price: ${aapl_signal['close_price']}")
    print(f"\nOverall Signal: {aapl_signal['overall_signal'].upper()}")
    print(f"Confidence: {aapl_signal['confidence']} ({aapl_signal['confidence_score']})")
    print(f"Agreement: {aapl_signal['agreement']}")
    print(f"\nIndicator Breakdown:")
    print(f"  RSI: {aapl_signal['rsi_signal']} (strength: {aapl_signal['rsi_strength']})")
    print(f"  MACD: {aapl_signal['macd_signal']} (strength: {aapl_signal['macd_strength']})")
    print(f"  SMA: {aapl_signal['sma_signal']} (strength: {aapl_signal['sma_strength']})")
    print(f"  News: {aapl_signal['news_sentiment']} ({aapl_signal['news_count']} articles)")

    # Save results
    latest_signals.to_csv(CONFLUENCE_CSV, index=False)
    print("\n✅ Results saved to 'signal_confluence_results.csv'")


if __name__ == "__main__":
    main()