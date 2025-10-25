import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import re


load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')


class AgenticAI:
    def __init__(self, model_name: str = "gemini-pro"):
        # The GoogleGenerativeAI class expects the API key to be in the environment variable GOOGLE_API_KEY
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',
                                    google_api_key =gemini_api_key,
                                      temperature=0.3)

        # Define the prompt template once
        self.template = """
Analyze this financial news headline.

Headline: {title}
Summary: {summary}
Entities: {tickers}
Sentiment: {sentiment}

Tasks:
1. Explain what this means for the listed tickers.
2. Give a confidence score (0–1).
3. Suggest if the impact is High/Medium/Low.

Respond in JSON with keys: analysis_text, confidence, impact_label.
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def analyze_with_gemini(self, title: str, summary: str, tickers: str, sentiment: str) -> dict:
        # Format prompt
        formatted_prompt = self.prompt.format(
            title=title,
            summary=summary,
            tickers=tickers,
            sentiment=sentiment
        )

        response_text = self.llm.invoke(formatted_prompt).content

        try:
            json_match = re.search(r'json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                cleaned_text = json_match.group(1).strip()
            else:
                cleaned_text = response_text.strip()

            return json.loads(cleaned_text)

        except json.JSONDecodeError as e:
            # If JSON invalid, fallback to default
            print(f"[WARN] JSON parsing failed (Error: {e}). Returning fallback for: {title}")
            return {
                "analysis_text": response_text,  # Keep the raw text for review
                "confidence": 0.5,
                "impact_label": "Medium",
            }


def main():

    agent = AgenticAI()

    title = "Company X beats quarterly earnings expectations"
    summary = "Strong sales growth in the North American market boosts revenue beyond forecasts."
    tickers = "COMPX"
    sentiment = "bullish"

    result = agent.analyze_with_gemini(title, summary, tickers, sentiment)
    print("Gemini analysis result:", result)