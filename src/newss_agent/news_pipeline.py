import json
import os
import pandas as pd
from .news_ingestion import NewsIngestor
from .sentiment_analyzer import SentimentAnalyzer,AgenticAnalyzer
from datetime import datetime
import pytz
from src.utils.paths import CONFIG_PATH
#from agentic_AI import AgenticAI


class NewsPipeline:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.config = self.load_config()
        self.ns = self.config["news"]
        self.ingestor = NewsIngestor(config_path)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.agentic_analyzer = AgenticAnalyzer()

    def load_config(self):
        import yaml
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def save_json(self, obj, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)

    def run(self):
        # 1) Ingest news
        raw_df = self.ingestor.ingest_news()
        if raw_df.empty:
            print("[INFO] No news fetched.")
            return

        self.ingestor.save_csv(raw_df)
        print(f"[OK] Raw saved: {self.ns['output_csv_raw']} ({len(raw_df)} rows)")

        # 2) Enrich news with sentiment and entities
        enriched = self.sentiment_analyzer.enrich_news(raw_df)
        if enriched.empty:
            print("[INFO] Enrichment yielded no rows.")
            return

        # 3) Calculate impact and explain
        enriched["impact_score"] = enriched.apply(lambda r: self.agentic_analyzer.impact_score(r), axis=1)
        enriched["impact_label"] = enriched["impact_score"].apply(self.agentic_analyzer.impact_label)
        enriched["explainer"] = enriched.apply(lambda r: self.agentic_analyzer.short_explainer(r), axis=1)

        # 4) Save enriched CSV
        self.sentiment_analyzer.save_csv(enriched, self.ns["output_csv_enriched"])
        print(f"[OK] Enriched saved: {self.ns['output_csv_enriched']} ({len(enriched)} rows)")

        # 5) Save JSON summary
        rollup = {
            "generated_at": datetime.now(pytz.utc).isoformat(),
            "counts": {
                "total": int(len(enriched)),
                "bullish": int((enriched["sentiment_label"] == "bullish").sum()),
                "bearish": int((enriched["sentiment_label"] == "bearish").sum()),
                "neutral": int((enriched["sentiment_label"] == "neutral").sum()),
                "high_impact": int((enriched["impact_label"] == "High").sum()),
            },
            "top_items": [
                {
                    "title": r["title"],
                    "source": r["source"],
                    "link": r["link"],
                    "tickers": r["tickers"],
                    "sentiment": r["sentiment_label"],
                    "impact": r["impact_label"],
                    "explainer": r["explainer"],
                }
                for _, r in enriched.sort_values("impact_score", ascending=False).head(10).iterrows()
            ],
        }
        self.save_json(rollup, self.ns["output_json_summary"])
        print(f"[OK] Summary saved: {self.ns['output_json_summary']}")


def main():
    pipeline = NewsPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()