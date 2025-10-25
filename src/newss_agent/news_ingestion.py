import hashlib
import time
import feedparser
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import pandas as pd
import yaml
import os
from dotenv import load_dotenv


class NewsIngestor:
    def __init__(self, config_path: str = "../../config.yaml"):
        self.config = self.load_config(config_path)
        self.sources = self.config["news"]["sources"]
        self.dedup_hours = self.config["news"]["dedup_hours"]
        self.output_csv_raw = self.config["news"]["output_csv_raw"]

    def load_config(self, path: str) -> Dict[str, Any]:
        load_dotenv()
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def normalize_text(self, s: str) -> str:
        return (s or "").strip().lower()

    def make_id(self, title: str, link: str, source: str) -> str:
        base = f"{self.normalize_text(title)}|{self.normalize_text(link)}|{self.normalize_text(source)}"
        return hashlib.sha256(base.encode()).hexdigest()

    def parse_feed(self, url: str) -> List[Dict[str, Any]]:
        parsed = feedparser.parse(url)
        items = []
        for e in parsed.entries:
            title = e.get("title", "")
            summary = e.get("summary", "")
            link = e.get("link", "")
            published = e.get("published_parsed") or e.get("updated_parsed")
            if published:
                dt = datetime(*published[:6], tzinfo=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
            items.append({
                "source": parsed.feed.get("title", url),
                "title": title,
                "summary": summary,
                "link": link,
                "published_utc": dt.isoformat(),
            })
        return items

    def ingest_news(self) -> pd.DataFrame:
        all_items = []
        for url in self.sources:
            try:
                all_items.extend(self.parse_feed(url))
                time.sleep(0.5)  # polite pause
            except Exception as e:
                print(f"[WARN] Failed {url}: {e}")

        df = pd.DataFrame(all_items)
        if df.empty:
            return df

        df["norm_title"] = df["title"].apply(self.normalize_text)
        df["id"] = df.apply(lambda r: self.make_id(r["title"], r["link"], r["source"]), axis=1)
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.dedup_hours)
        df = df[df["published_utc"] >= cutoff].copy()
        df = df.sort_values("published_utc", ascending=False)
        df = df.drop_duplicates(subset=["norm_title", "source"], keep="first")

        return df.drop(columns=["norm_title"])

    def save_csv(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self.output_csv_raw), exist_ok=True)
        df.to_csv(self.output_csv_raw, index=False)


def main():
    ingestor = NewsIngestor()
    df = ingestor.ingest_news()
    if df.empty:
        print("[INFO] No news fetched.")
    else:
        ingestor.save_csv(df)
        print(f"[OK] Saved raw news: {ingestor.output_csv_raw} ({len(df)} rows)")