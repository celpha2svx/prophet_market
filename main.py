"""
Stock Prediction System - Master Pipeline
==========================================
Runs the complete end-to-end pipeline:
1. Data ingestion
2. Feature engineering
3. Data Cleaning
4. Risk clustering
5. Prophet forecasting
6. Signal confluence
7. News sentiment analysis
8. Recommendations generation
9. Email reporting
10. Dashboard update

Usage:
    python main.py              # Normal run
    python main.py --dry-run    # Test without executing
    python main.py --force      # Force run even if data is recent
"""

import logging
import datetime
import traceback
import time
import sys
import os
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional

# === Import your modules ===
from src.data_pipeline import schedular, cleaning_data
from src.featuress import stock_features
from src.models import Clustering_model_risk, propheT_engine_forcast
from src.strategies import stock_signal_engine, recomendation_engine
from src.newss_agent import news_pipeline
from src.web_Email_T import email_engine

# === Configuration ===
CONFIG_PATH = "config.yaml"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / f"pipeline_{datetime.datetime.now():%Y%m%d}.log"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)

# === Setup logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


class PipelineStats:
    """Track pipeline execution statistics"""

    def __init__(self):
        self.steps = {}
        self.start_time = time.time()

    def record_step(self, name: str, success: bool, duration: float, error: Optional[str] = None):
        """Record step execution result"""
        self.steps[name] = {
            'success': success,
            'duration': duration,
            'error': error,
            'timestamp': datetime.datetime.now().isoformat()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_duration = time.time() - self.start_time
        successes = sum(1 for s in self.steps.values() if s['success'])
        failures = len(self.steps) - successes

        return {
            'total_duration': round(total_duration, 2),
            'total_steps': len(self.steps),
            'successes': successes,
            'failures': failures,
            'steps': self.steps,
            'timestamp': datetime.datetime.now().isoformat()
        }

    def save_report(self, filename: str = "logs/pipeline_report.json"):
        """Save execution report to JSON"""
        summary = self.get_summary()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Pipeline report saved to {filename}")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {path} not found, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def send_failure_alert(step_name: str, error: str, config: Dict[str, Any]):
    """Send alert when pipeline step fails"""
    try:
        # Only send if configured
        if not config.get('alerts', {}).get('enabled', False):
            return

        subject = f"⚠ Pipeline Failed: {step_name}"
        body = f"""
        Pipeline Step Failed
        ====================

        Step: {step_name}
        Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Error:
        {error}

        Check logs at: {LOG_FILE}

        ---
        Stock Prediction System
        """

        # Use email_engine to send alert
        # email_engine.send_alert(subject, body)
        logger.info(f"Failure alert prepared for {step_name}")

    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


def run_step(name: str, func, stats: PipelineStats, config: Dict[str, Any],
             dry_run: bool = False, *args, **kwargs) -> Optional[Any]:
    """
    Execute a pipeline step with error handling and timing.

    Args:
        name: Step name for logging
        func: Function to execute
        stats: PipelineStats object to record results
        config: Configuration dictionary
        dry_run: If True, skip actual execution
        *args, **kwargs: Arguments to pass to func

    Returns:
        Function result or None if failed
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}START: {name}")
    print(f"\n{'🧪 ' if dry_run else '▶ '}{name}...")

    if dry_run:
        print(f"  Would execute: {func._module}.{func.name_}")
        stats.record_step(name, True, 0.0)
        return True

    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        stats.record_step(name, True, duration)
        logger.info(f"SUCCESS: {name} (took {duration:.2f}s)")
        print(f"✔ {name} completed in {duration:.2f}s")

        return result

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()

        stats.record_step(name, False, duration, error_msg)
        logger.error(f"FAILED: {name} after {duration:.2f}s")
        logger.error(f"Error: {error_msg}")
        logger.debug(f"Traceback:\n{error_trace}")

        print(f"✘ {name} failed after {duration:.2f}s")
        print(f"  Error: {error_msg}")

        # Send alert if configured
        send_failure_alert(name, error_trace, config)

        # Continue to next step (don't crash entire pipeline)
        return None


def check_prerequisites() -> bool:
    """Check if all prerequisites are met before running pipeline"""
    logger.info("Checking prerequisites...")

    checks = {
        'config_exists': os.path.exists(CONFIG_PATH),
        'data_dir_exists': os.path.exists('data'),
        'logs_dir_exists': os.path.exists('logs'),
    }

    all_passed = all(checks.values())

    if not all_passed:
        logger.error("Prerequisites check failed:")
        for check, passed in checks.items():
            status = "✔" if passed else "✘"
            logger.error(f"  {status} {check}")
    else:
        logger.info("All prerequisites met")

    return all_passed


def print_banner():
    """Print startup banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║          STOCK PREDICTION SYSTEM - DAILY PIPELINE         ║
    ║                                                            ║
    ║  🤖 AI-Powered Stock Analysis & Trading Recommendations   ║
    ║  📊 Prophet ML + Technical Signals + News Sentiment       ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log file: {LOG_FILE}")
    print("  " + "─" * 58)


def print_summary(stats: PipelineStats):
    """Print execution summary"""
    summary = stats.get_summary()

    print("\n" + "═" * 60)
    print("  PIPELINE EXECUTION SUMMARY")
    print("═" * 60)
    print(f"  Total Duration: {summary['total_duration']:.2f}s")
    print(f"  Steps Completed: {summary['successes']}/{summary['total_steps']}")

    if summary['failures'] > 0:
        print(f"  ⚠  Failed Steps: {summary['failures']}")
        print("\n  Failed steps:")
        for name, data in summary['steps'].items():
            if not data['success']:
                print(f"    ✘ {name}: {data['error']}")
    else:
        print("  ✔ All steps succeeded!")

    print("\n  Step Timings:")
    for name, data in summary['steps'].items():
        status = "✔" if data['success'] else "✘"
        print(f"    {status} {name:<30} {data['duration']:>6.2f}s")

    print("═" * 60)

    if summary['failures'] == 0:
        print("  🎉 Pipeline completed successfully!")
    else:
        print(f"  ⚠  Pipeline completed with {summary['failures']} error(s)")
        print(f"  Check logs: {LOG_FILE}")

    print("═" * 60 + "\n")


def main(dry_run: bool = False, force: bool = False):
    """
    Main pipeline execution function.

    Args:
        dry_run: If True, simulate execution without running
        force: If True, skip recency checks and force execution
    """
    # Print banner
    print_banner()

    # Load configuration
    config = load_config()

    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Aborting.")
        return 1

    # Initialize statistics tracker
    stats = PipelineStats()

    # Log run parameters
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'NORMAL'}")
    logger.info(f"Force: {force}")
    logger.info(f"Config: {CONFIG_PATH}")
    logger.info("=" * 60)

    # ========================================
    # PIPELINE STEPS
    # ========================================

    # Step 1: Data Ingestion
    run_step(
        "Data Ingestion",
        schedular.main,
        stats, config, dry_run
    )

    # Step 2: Feature Engineering
    run_step(
         "Feature Engineering",
        stock_features.main,
        stats, config, dry_run
    )

    # Step 3: Data Cleaning
    run_step(
        "Data Cleaning",
        cleaning_data.main,
        stats, config, dry_run
    )

    # Step 4: Risk Clustering
    run_step(
        "Risk Clustering",
        Clustering_model_risk.main,
        stats, config, dry_run
    )

    # Step 5: Prophet Forecasting
    run_step(
        "Prophet Forecasting",
        propheT_engine_forcast.main,
        stats, config, dry_run
    )

    # Step 6: News Sentiment Analysis
    run_step(
        "News Sentiment Pipeline",
        news_pipeline.main,
        stats, config, dry_run
    )

    # Step 7: Signal Confluence
    run_step(
        "Signal Confluence",
        stock_signal_engine.main,
        stats, config, dry_run
    )

    # Step 8: Generate Recommendations
    run_step(
        "Recommendation Engine",
        recomendation_engine.main,
        stats, config, dry_run
    )

    # Step 9: Send Email Report
    run_step(
        "Email Report",
        email_engine.main,
        stats, config, dry_run
    )

    # Step 10: Dashboard Refresh
    # Dashboard auto-refreshes by reading JSON/CSV files
    logger.info("Dashboard data updated (reads from generated files)")
    print("✔ Dashboard data ready")

    # ========================================
    # FINALIZATION
    # ========================================

    # Print summary
    print_summary(stats)

    # Save execution report
    stats.save_report()

    # Log completion
    summary = stats.get_summary()
    logger.info("=" * 60)
    logger.info("PIPELINE END")
    logger.info(f"Duration: {summary['total_duration']:.2f}s")
    logger.info(f"Success: {summary['successes']}/{summary['total_steps']}")
    logger.info("=" * 60)

    # Return exit code (0 = success, 1 = failure)
    return 0 if summary['failures'] == 0 else 1


if __name__ == "__main__":
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv

    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)

    # Run pipeline
    try:
        exit_code = main(dry_run=dry_run, force=force)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logger.critical(f"Unexpected error: {e}\n{traceback.format_exc()}")
        sys.exit(1)