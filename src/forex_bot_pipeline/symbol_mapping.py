import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolMapper:
    """
    Maps database symbols to broker-specific symbols and vice versa.
    """

    def __init__(self, broker='exness'):
        """
        Initialize symbol mapper for specific broker.

        Args:
            broker: Broker name ('exness', 'fxcm', 'ic_markets', etc.)
        """
        self.broker = broker.lower()
        self.mapping = self._get_broker_mapping()

        # Create reverse mapping (broker → database)
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

        logger.info(f"✓ Symbol mapper initialized for {broker}")
        logger.info(f"  Mapped {len(self.mapping)} symbols")

    def _get_broker_mapping(self):
        """
        Get symbol mapping for specific broker.

        Returns:
            dict: {database_symbol: broker_symbol}
        """
        # Exness uses 'm' suffix for standard accounts
        if self.broker == 'exness':
            return {
                'EURUSD=X': 'EURUSDm',
                'GBPUSD=X': 'GBPUSDm',
                'USDJPY=X': 'USDJPYm',
                'AUDUSD=X': 'AUDUSDm',
                'USDCAD=X': 'USDCADm',
                'USDCHF=X': 'USDCHFm',
                'NZDUSD=X': 'NZDUSDm',
                'EURGBP=X': 'EURGBPm',
                'EURJPY=X': 'EURJPYm',
                'GBPJPY=X': 'GBPJPYm',
                'GC=F': 'XAUUSDm'  # Gold
            }

        # FXCM uses no suffix
        elif self.broker == 'fxcm':
            return {
                'EURUSD=X': 'EURUSD',
                'GBPUSD=X': 'GBPUSD',
                'USDJPY=X': 'USDJPY',
                'AUDUSD=X': 'AUDUSD',
                'USDCAD=X': 'USDCAD',
                'USDCHF=X': 'USDCHF',
                'NZDUSD=X': 'NZDUSD',
                'EURGBP=X': 'EURGBP',
                'EURJPY=X': 'EURJPY',
                'GBPJPY=X': 'GBPJPY',
                'GC=F': 'XAUUSD'
            }

        # IC Markets uses no suffix
        elif self.broker == 'ic_markets':
            return {
                'EURUSD=X': 'EURUSD',
                'GBPUSD=X': 'GBPUSD',
                'USDJPY=X': 'USDJPY',
                'AUDUSD=X': 'AUDUSD',
                'USDCAD=X': 'USDCAD',
                'USDCHF=X': 'USDCHF',
                'NZDUSD=X': 'NZDUSD',
                'EURGBP=X': 'EURGBP',
                'EURJPY=X': 'EURJPY',
                'GBPJPY=X': 'GBPJPY',
                'GC=F': 'XAUUSD'
            }

        # Default: strip =X suffix
        else:
            logger.warning(f"Unknown broker '{self.broker}', using default mapping")
            return {
                'EURUSD=X': 'EURUSD',
                'GBPUSD=X': 'GBPUSD',
                'USDJPY=X': 'USDJPY',
                'AUDUSD=X': 'AUDUSD',
                'USDCAD=X': 'USDCAD',
                'USDCHF=X': 'USDCHF',
                'NZDUSD=X': 'NZDUSD',
                'EURGBP=X': 'EURGBP',
                'EURJPY=X': 'EURJPY',
                'GBPJPY=X': 'GBPJPY',
                'GC=F': 'XAUUSD'
            }

    def to_broker(self, database_symbol):
        """
        Convert database symbol to broker symbol.

        Args:
            database_symbol: Symbol from database (e.g., 'EURUSD=X')

        Returns:
            str: Broker symbol (e.g., 'EURUSDm')
        """
        broker_symbol = self.mapping.get(database_symbol)

        if broker_symbol is None:
            logger.warning(f"No mapping found for {database_symbol}, returning as-is")
            # Fallback: strip =X suffix
            return database_symbol.replace('=X', '')

        return broker_symbol

    def to_database(self, broker_symbol):
        """
        Convert broker symbol to database symbol.

        Args:
            broker_symbol: Symbol from broker (e.g., 'EURUSDm')

        Returns:
            str: Database symbol (e.g., 'EURUSD=X')
        """
        database_symbol = self.reverse_mapping.get(broker_symbol)

        if database_symbol is None:
            logger.warning(f"No reverse mapping found for {broker_symbol}")
            # Fallback: add =X suffix
            return f"{broker_symbol}=X"

        return database_symbol

    def get_all_broker_symbols(self):
        """Get list of all broker symbols."""
        return list(self.mapping.values())

    def get_all_database_symbols(self):
        """Get list of all database symbols."""
        return list(self.mapping.keys())

    def display_mapping(self):
        """Display the symbol mapping table."""
        print(f"\n{'=' * 60}")
        print(f"SYMBOL MAPPING ({self.broker.upper()})")
        print(f"{'=' * 60}")
        print(f"{'Database Symbol':<20} → {'Broker Symbol':<20}")
        print(f"{'-' * 60}")

        for db_sym, broker_sym in self.mapping.items():
            print(f"{db_sym:<20} → {broker_sym:<20}")


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Initialize for Exness
    mapper = SymbolMapper(broker='exness')

    # Display mapping
    mapper.display_mapping()

    # Convert database → broker
    print("\n--- Convert Database → Broker ---")
    print(f"EURUSD=X → {mapper.to_broker('EURUSD=X')}")
    print(f"GC=F → {mapper.to_broker('GC=F')}")

    # Convert broker → database
    print("\n--- Convert Broker → Database ---")
    print(f"EURUSDm → {mapper.to_database('EURUSDm')}")
    print(f"XAUUSD → {mapper.to_database('XAUUSD')}")

    # Get all symbols
    print(f"\nAll broker symbols: {mapper.get_all_broker_symbols()}")