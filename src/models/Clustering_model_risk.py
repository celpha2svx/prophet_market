import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from src.utils.paths import ALL_CLEANED_CSV,ALL_Stocluster_CSV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockClusteringModel:
    """
    Clusters stocks based on risk and volatility characteristics.
    Uses features like volatility, drawdown, and return patterns.
    """

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = {0: 'Low Risk', 1: 'Medium Risk',  2: 'Medium High Risk', 3: 'High Risk'}

    def prepare_clustering_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregate risk metrics per stock for clustering.

        Returns DataFrame with one row per stock and risk metrics as columns.
        """
        # Group by symbol to get per-stock metrics
        stock_metrics = []

        for symbol in features_df['symbol'].unique():
            stock_data = features_df[features_df['symbol'] == symbol].copy()

            # Remove NaN values for calculations
            stock_data = stock_data.dropna(subset=['daily_return', 'realized_volatility', 'max_drawdown'])

            if len(stock_data) < 30:  # Need minimum data
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue

            metrics = {
                'symbol': symbol,
                # Volatility measures
                'avg_volatility': stock_data['realized_volatility'].mean(),
                'max_volatility': stock_data['realized_volatility'].max(),
                # Return measures
                'avg_return': stock_data['daily_return'].mean(),
                'return_std': stock_data['daily_return'].std(),
                # Risk measures
                'avg_drawdown': stock_data['max_drawdown'].mean(),
                'max_drawdown': stock_data['max_drawdown'].min(),  # Most negative
                # Price swings
                'return_range': stock_data['daily_return'].max() - stock_data['daily_return'].min(),
            }

            stock_metrics.append(metrics)

        clustering_df = pd.DataFrame(stock_metrics)
        logger.info(f"Prepared clustering features for {len(clustering_df)} stocks")

        return clustering_df

    def fit(self, features_df: pd.DataFrame):
        """
        Fit the clustering model on stock features.
        """
        # Prepare data
        self.clustering_data = self.prepare_clustering_features(features_df)

        if self.clustering_data.empty:
            logger.error("No data available for clustering")
            return

        # Select features for clustering (exclude symbol)
        feature_columns = [col for col in self.clustering_data.columns if col != 'symbol']
        X = self.clustering_data[feature_columns].values

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit KMeans
        self.kmeans.fit(X_scaled)

        # Assign clusters
        self.clustering_data['cluster'] = self.kmeans.labels_

        # Sort clusters by risk (0=low, 1=med, 2=high)
        self._reorder_clusters_by_risk()

        logger.info("Clustering model fitted successfully")

    def _reorder_clusters_by_risk(self):
        """
        Reorder cluster labels so 0=Low Risk, 1=Medium, 2=High
        based on average volatility in each cluster.
        """
        cluster_risk = self.clustering_data.groupby('cluster')['avg_volatility'].mean().sort_values()

        # Create mapping from old labels to new labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_risk.index)}

        # Apply mapping
        self.clustering_data['cluster'] = self.clustering_data['cluster'].map(label_mapping)

        # Add human-readable labels
        self.clustering_data['risk_label'] = self.clustering_data['cluster'].map(self.cluster_labels)


    def predict(self, symbols: list, save_path: str = ALL_Stocluster_CSV) -> dict:
        """Get cluster assignment for multiple symbols and save to CSV"""
        if self.clustering_data is None or self.clustering_data.empty:
            logger.error("Model not fitted yet")
            return None

        results = {}
        for symbol in symbols:
            stock_data = self.clustering_data[self.clustering_data['symbol'] == symbol]

            if stock_data.empty:
                logger.warning(f"Symbol {symbol} not found")
                results[symbol] = None
            else:
                results[symbol] = stock_data.iloc[0].to_dict()

        # Convert results to DataFrame for saving
        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.to_csv(save_path, index_label='symbol')

        logger.info(f"Cluster visualization saved to {save_path}")

        return results

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        """
        summary = self.clustering_data.groupby('risk_label').agg({
            'symbol': 'count',
            'avg_volatility': 'mean',
            'avg_return': 'mean',
            'max_drawdown': 'mean',
            'return_std': 'mean'
        }).round(4)

        summary.columns = ['num_stocks', 'avg_volatility', 'avg_return', 'avg_max_drawdown', 'avg_return_std']

        return summary

    def visualize_clusters(self, save_path: str = 'stock_clusters.png'):
        """
        Create visualization of stock clusters.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Volatility vs Return
        ax1 = axes[0]
        for cluster_id in range(self.n_clusters):
            cluster_data = self.clustering_data[self.clustering_data['cluster'] == cluster_id]
            ax1.scatter(cluster_data['avg_volatility'],
                        cluster_data['avg_return'],
                        label=self.cluster_labels[cluster_id],
                        s=200, alpha=0.6)

            # Add stock labels
            for _, row in cluster_data.iterrows():
                ax1.annotate(row['symbol'],
                             (row['avg_volatility'], row['avg_return']),
                             fontsize=10, ha='right')

        ax1.set_xlabel('Average Volatility', fontsize=12)
        ax1.set_ylabel('Average Return', fontsize=12)
        ax1.set_title('Stock Clusters: Risk vs Return', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Plot 2: Max Drawdown vs Volatility
        ax2 = axes[1]
        for cluster_id in range(self.n_clusters):
            cluster_data = self.clustering_data[self.clustering_data['cluster'] == cluster_id]
            ax2.scatter(cluster_data['avg_volatility'],
                        cluster_data['max_drawdown'],
                        label=self.cluster_labels[cluster_id],
                        s=200, alpha=0.6)

            # Add stock labels
            for _, row in cluster_data.iterrows():
                ax2.annotate(row['symbol'],
                             (row['avg_volatility'], row['max_drawdown']),
                             fontsize=10, ha='right')

        ax2.set_xlabel('Average Volatility', fontsize=12)
        ax2.set_ylabel('Max Drawdown', fontsize=12)
        ax2.set_title('Stock Clusters: Volatility vs Drawdown', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Cluster visualization saved to {save_path}")


# ============ EXAMPLE USAGE ============

def main():
    # Load features (from Phase A2)
    print("Loading features...")
    features_df = pd.read_csv(ALL_CLEANED_CSV)

    # Initialize clustering model
    print("\nInitializing clustering model...")
    clustering_model = StockClusteringModel(n_clusters=4)

    # Fit the model
    print("Fitting clustering model...")
    clustering_model.fit(features_df)

    # Get cluster assignments
    print("\n" + "=" * 60)
    print("STOCK CLUSTER ASSIGNMENTS")
    print("=" * 60)
    print(clustering_model.clustering_data[['symbol', 'risk_label', 'avg_volatility', 'avg_return', 'max_drawdown']])

    # Get cluster summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY STATISTICS")
    print("=" * 60)
    print(clustering_model.get_cluster_summary())

    # Get specific stock info
    print("\n" + "=" * 60)
    print("ALL RISK PROFILE")
    print("=" * 60)
    symbols =[ "AAPL", "MSFT", "TSLA", "JPM","NVDA","GOOGL", "AMD" ,
                "SNOW" , "COIN","META" ,"BAC" , "GS" ,"MA" , "UNH" ,
                "ABBV" , "CVX" , "WMT" ,"NKE" ,"PFE" , "PG","DIS","SCHW",
                "QCOM", "ADBE","CRM", "BMY","C","SBUX","UPS","BA","MCD",
                "BLK","MDT","TMO", "AVGO","INTC", "AXP","LLY","COST", "CAT","DE" ]

    all_profile = clustering_model.predict(symbols)
    if all_profile:
        for symbol, profile in all_profile.items():
            print(f"Symbol: {symbol}, Profile: {profile}")


    # Visualize
    #print("\nCreating cluster visualization...")
    #clustering_model.visualize_clusters('stock_cluster4.png')

    print("\n✅ Clustering complete!")