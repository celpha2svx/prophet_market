import json
from typing import Dict, List, Any
import pandas as pd
import spacy
from nltk.sentiment import vader
from nltk import download as nltk_download
import os
from src.utils.paths import TICKER_MAPPING


class SentimentAnalyzer:
    def __init__(self, ticker_map_path: str = TICKER_MAPPING):
        try:
            nltk_download('vader_lexicon')
        except:
            pass
        self.nlp = self.load_spacy()
        self.ticker_map = self.load_ticker_map(ticker_map_path)
        self.vader_analyzer = vader.SentimentIntensityAnalyzer()

    def load_spacy(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Install spaCy model: python -m spacy download en_core_web_sm")

    def load_ticker_map(self, path: str) -> Dict[str, List[str]]:
        with open(path, "r") as f:
            return json.load(f)

    def normalize_entities(self, entities: List[str]) -> List[str]:
        return [e.lower() for e in entities]

    def map_entities_to_tickers(self, entities: List[str]) -> List[str]:
        ents_norm = self.normalize_entities(entities)
        hits = set()
        for ticker, aliases in self.ticker_map.items():
            for a in aliases:
                a_norm = a.lower()
                if any(a_norm in e for e in ents_norm):
                    hits.add(ticker)
        return sorted(hits)

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text or "")
        return [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "GPE")]

    def vader_sentiment(self, text: str) -> Dict[str, float]:
        return self.vader_analyzer.polarity_scores(text or "")

    def classify_sentiment(self, compound: float, pos: float, neg: float) -> str:
        if compound >= 0.2 and pos > neg:
            return "bullish"
        if compound <= -0.2 and neg > pos:
            return "bearish"
        return "neutral"

    def enrich_news(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        records = []
        for _, row in df.iterrows():
            text = f"{row['title']} {row.get('summary', '')}"
            ents = self.extract_entities(text)
            senti = self.vader_sentiment(text)
            label = self.classify_sentiment(senti["compound"], senti["pos"], senti["neg"])
            tickers = self.map_entities_to_tickers(ents)
            records.append({
                **row,
                "entities": ", ".join(ents),
                "tickers": ", ".join(tickers),
                "sentiment_compound": senti["compound"],
                "sentiment_label": label,
            })
        return pd.DataFrame(records)

    def save_csv(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


class AgenticAnalyzer:
    @staticmethod
    def impact_score(row: Dict[str, Any]) -> float:
        mag = abs(row.get("sentiment_compound", 0.0))
        has_ticker = 1.0 if (row.get("tickers") or "").strip() else 0.5
        length_bonus = min(len((row.get("title") or "")) / 120.0, 1.0)
        recency_bonus = 0.2
        score = 0.5 * mag + 0.3 * has_ticker + 0.2 * length_bonus + recency_bonus
        return max(0.0, min(score, 1.0))

    @staticmethod
    def impact_label(score: float) -> str:
        if score >= 0.7:
            return "High"
        if score >= 0.45:
            return "Medium"
        return "Low"

    @staticmethod
    def short_explainer(row: Dict[str, Any]) -> str:
        s_label = row.get("sentiment_label", "neutral")
        tick = (row.get("tickers") or "market").split(",")[0].strip()
        title = (row.get("title") or "").strip()
        impact = AgenticAnalyzer.impact_label(AgenticAnalyzer.impact_score(row))
        return (
            f"This headline is {s_label} for {tick}. "
            f"Impact: {impact}. "
            f"Why: sentiment magnitude and entity match suggest a potential move. "
            f"Headline: “{title}”."
        )