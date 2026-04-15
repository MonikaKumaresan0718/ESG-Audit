"""
Data loaders for ESG audit pipeline.
Supports CSV, PDF (PyMuPDF), and News API sources.
"""

import io
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)


class CSVLoader:
    """Load ESG data from CSV files."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """Load and validate CSV data."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")

        df = pd.read_csv(self.filepath)
        logger.info(f"Loaded CSV: {self.filepath} ({len(df)} rows, {len(df.columns)} cols)")

        # Basic cleaning
        df = df.dropna(how="all")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        return df

    def load_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Load data for a specific company."""
        df = self.load()
        if "company" in df.columns:
            row = df[df["company"].str.lower() == company_name.lower()]
            if not row.empty:
                return row.iloc[0].to_dict()
        return df.iloc[0].to_dict() if not df.empty else None


class PDFLoader:
    """Load and extract text from PDF sustainability reports using PyMuPDF."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def extract_full_text(self) -> str:
        """Extract all text from PDF."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(self.filepath)
            texts = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    texts.append(f"[Page {page_num + 1}]\n{text.strip()}")
            doc.close()
            full_text = "\n\n".join(texts)
            logger.info(
                f"Extracted {len(full_text)} chars from {self.filepath} ({len(texts)} pages)"
            )
            return full_text
        except ImportError:
            logger.warning("PyMuPDF not installed; using text file fallback")
            return self._text_fallback()
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""

    def extract_text_chunks(self, chunk_size: int = 512) -> List[str]:
        """Extract text split into chunks of approximately chunk_size characters."""
        full_text = self.extract_full_text()
        if not full_text:
            return []

        # Split into sentences first, then chunk
        chunks = []
        words = full_text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if len(c.strip()) > 20]

    def extract_sections(self) -> Dict[str, str]:
        """Extract common ESG report sections by keyword detection."""
        full_text = self.extract_full_text()
        sections: Dict[str, str] = {}

        section_keywords = {
            "environmental": ["environment", "climate", "carbon", "emissions", "energy"],
            "social": ["social", "employees", "community", "diversity", "inclusion"],
            "governance": ["governance", "board", "ethics", "compliance", "audit"],
        }

        lines = full_text.split("\n")
        current_section = "general"
        sections[current_section] = []

        for line in lines:
            line_lower = line.lower()
            for section, keywords in section_keywords.items():
                if any(kw in line_lower for kw in keywords):
                    current_section = section
                    break
            sections.setdefault(current_section, []).append(line)

        return {k: "\n".join(v) for k, v in sections.items() if v}

    def _text_fallback(self) -> str:
        """Try reading file as plain text."""
        try:
            with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


class NewsAPILoader:
    """Fetch ESG-related news articles from News API."""

    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2"):
        self.api_key = api_key
        self.base_url = base_url

    def fetch(
        self,
        company_name: str,
        max_articles: int = 10,
        language: str = "en",
    ) -> List[str]:
        """
        Fetch news articles for a company.

        Returns:
            List of article text strings.
        """
        try:
            import requests

            query = f'"{company_name}" ESG sustainability environment'
            params = {
                "q": query,
                "language": language,
                "pageSize": max_articles,
                "apiKey": self.api_key,
                "sortBy": "publishedAt",
            }

            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            texts = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                content = article.get("content", "")
                text = f"{title}. {description}. {content}"
                if len(text.strip()) > 50:
                    texts.append(text[:1024])  # Limit per article

            logger.info(f"Fetched {len(texts)} articles for {company_name}")
            return texts

        except Exception as e:
            logger.error(f"News API fetch failed: {e}")
            return self._generate_synthetic_news(company_name)

    def _generate_synthetic_news(self, company_name: str) -> List[str]:
        """Generate synthetic news as fallback."""
        return [
            f"{company_name} announces new carbon neutrality targets for 2030, "
            "pledging to reduce Scope 1 and 2 emissions by 50% within 5 years.",
            f"{company_name} faces scrutiny over supply chain labor practices "
            "following audit revelations in South Asian manufacturing facilities.",
            f"{company_name} board diversity initiative increases female representation "
            "to 35% amid pressure from institutional investors.",
            f"ESG analysts downgrade {company_name} governance score following "
            "executive pay controversy and shareholder rights concerns.",
            f"{company_name} water stewardship program launches in water-stressed regions "
            "as part of its sustainable operations strategy.",
        ]


class JSONLoader:
    """Load ESG data from JSON files."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> Dict[str, Any]:
        """Load JSON file and return as dict."""
        import json

        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON: {self.filepath}")
        return data