import pytz
import datetime as dt
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from src.data_pipeline.data_injestion import DataIngestionManager, load_app_config  # adjust import if needed  # pyright: ignore[reportMissingImports]

def run_eod_job():
    cfg = load_app_config()
    ingestion = DataIngestionManager(db_path=cfg["database"]["path"])
    tz = pytz.timezone(cfg["app"]["timezone"])

    all_symbols = cfg["tickers"]["stocks"] + cfg["tickers"]["forex"]

    period = cfg["ingestion"]["yahoo"]["period"]
    res = ingestion.ingest_stock_data(all_symbols, data_source="yahoo", period=period)
    print(f"[{dt.datetime.now(tz)}] EOD ingestion result:", res)

def main():
    cfg = load_app_config()
    tz = pytz.timezone(cfg["app"]["timezone"])
    hh, mm = map(int, cfg["app"]["eod_time_local"].split(":"))

    sched = BlockingScheduler(timezone=tz)
    # Run every day at local EOD time
    sched.add_job(run_eod_job, CronTrigger(hour=hh, minute=mm))
    print(f"Scheduler started; job at {cfg['app']['eod_time_local']} {cfg['app']['timezone']} daily.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == "__main__":
    main()