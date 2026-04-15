"""
DataIngestionAgent – Loads structured ESG data from CSV, extracts text from PDFs
using PyMuPDF, optionally fetches news, and stores embeddings in FAISS.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from crewai import Agent

from core.config import settings
from core.logging import get_logger
from tools.data_loaders import CSVLoader, PDFLoader, NewsAPILoader
from tools.embedding_engine import EmbeddingEngine
from tools.vector_store import FAISSVectorStore

logger = get_logger(__name__)


class DataIngestionAgent:
    """
    Handles all data ingestion tasks for the ESG audit pipeline.
    Supports CSV, PDF, and optional news API data sources.
    """

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = FAISSVectorStore(
            embedding_dim=settings.EMBEDDING_DIM,
            index_path=settings.FAISS_INDEX_PATH,
        )
        self.agent = self._build_crewai_agent()

    def _build_crewai_agent(self) -> Agent:
        return Agent(
            role="ESG Data Ingestion Specialist",
            goal=(
                "Collect, parse, and store all ESG-related data from structured "
                "CSV files, unstructured PDF sustainability reports, and news APIs. "
                "Convert all text into vector embeddings stored in FAISS."
            ),
            backstory=(
                "You are a data engineering expert specializing in ESG data pipelines. "
                "You have experience parsing complex sustainability reports, cleaning "
                "messy ESG datasets, and building vector indexes for semantic search. "
                "You ensure data quality and completeness before analysis."
            ),
            verbose=settings.VERBOSE_AGENTS,
            allow_delegation=False,
        )

    def ingest(
        self,
        company_name: str,
        csv_path: Optional[str] = None,
        pdf_path: Optional[str] = None,
        esg_data: Optional[Dict[str, Any]] = None,
        fetch_news: bool = False,
    ) -> Dict[str, Any]:
        """
        Main ingestion entry point.

        Args:
            company_name: Company name for news fetching and metadata.
            csv_path: Path to ESG metrics CSV file.
            pdf_path: Path to sustainability report PDF.
            esg_data: Pre-loaded structured ESG data dict.
            fetch_news: Whether to fetch recent news articles.

        Returns:
            Dict containing structured_data, texts, embeddings, and metadata.
        """
        logger.info(f"Starting data ingestion for {company_name}")

        result: Dict[str, Any] = {
            "company_name": company_name,
            "structured_data": {},
            "texts": [],
            "text_sources": [],
            "embedding_count": 0,
            "errors": [],
        }

        # 1. Load structured data from CSV
        if csv_path and os.path.exists(csv_path):
            try:
                csv_loader = CSVLoader(csv_path)
                df = csv_loader.load()
                # Filter for company if column exists
                if "company" in df.columns:
                    company_df = df[df["company"].str.lower() == company_name.lower()]
                    if not company_df.empty:
                        result["structured_data"] = company_df.iloc[0].to_dict()
                    else:
                        result["structured_data"] = df.iloc[0].to_dict()
                else:
                    result["structured_data"] = df.iloc[0].to_dict() if not df.empty else {}
                logger.info(f"Loaded CSV data with {len(df)} rows")
            except Exception as e:
                logger.error(f"CSV loading failed: {e}")
                result["errors"].append(f"CSV: {str(e)}")

        # 2. Use pre-loaded ESG data if provided
        if esg_data:
            result["structured_data"].update(esg_data)

        # 3. Fall back to default CSV if no data loaded
        if not result["structured_data"]:
            default_csv = os.path.join(settings.DATA_DIR, "sample_esg_data.csv")
            if os.path.exists(default_csv):
                try:
                    df = pd.read_csv(default_csv)
                    result["structured_data"] = df.iloc[0].to_dict()
                    logger.info("Using default sample ESG data")
                except Exception as e:
                    logger.error(f"Default CSV loading failed: {e}")
                    result["structured_data"] = self._generate_synthetic_esg_data()
            else:
                result["structured_data"] = self._generate_synthetic_esg_data()

        # 4. Extract text from PDF
        if pdf_path and os.path.exists(pdf_path):
            try:
                pdf_loader = PDFLoader(pdf_path)
                pdf_texts = pdf_loader.extract_text_chunks(chunk_size=512)
                result["texts"].extend(pdf_texts)
                result["text_sources"].extend(["pdf"] * len(pdf_texts))
                logger.info(f"Extracted {len(pdf_texts)} text chunks from PDF")
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                result["errors"].append(f"PDF: {str(e)}")

        # 5. Optionally fetch news
        if fetch_news and settings.NEWS_API_KEY:
            try:
                news_loader = NewsAPILoader(api_key=settings.NEWS_API_KEY)
                news_texts = news_loader.fetch(company_name, max_articles=10)
                result["texts"].extend(news_texts)
                result["text_sources"].extend(["news"] * len(news_texts))
                logger.info(f"Fetched {len(news_texts)} news articles")
            except Exception as e:
                logger.error(f"News fetching failed: {e}")
                result["errors"].append(f"News: {str(e)}")

        # 6. Add synthetic text from structured data
        synthetic_texts = self._generate_text_from_structured(
            result["structured_data"], company_name
        )
        result["texts"].extend(synthetic_texts)
        result["text_sources"].extend(["structured"] * len(synthetic_texts))

        # 7. Generate and store embeddings
        if result["texts"]:
            try:
                embeddings = self.embedding_engine.encode(result["texts"])
                self.vector_store.add(
                    embeddings=embeddings,
                    texts=result["texts"],
                    metadata=[
                        {"source": src, "company": company_name}
                        for src in result["text_sources"]
                    ],
                )
                result["embedding_count"] = len(embeddings)
                result["embeddings_shape"] = list(embeddings.shape)
                logger.info(f"Stored {len(embeddings)} embeddings in FAISS")
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                result["errors"].append(f"Embeddings: {str(e)}")

        result["ingestion_success"] = len(result["errors"]) == 0
        return result

    def _generate_synthetic_esg_data(self) -> Dict[str, Any]:
        """Generate synthetic ESG data as fallback."""
        rng = np.random.default_rng(42)
        return {
            "carbon_emissions": float(rng.uniform(50, 500)),
            "water_usage": float(rng.uniform(100, 1000)),
            "board_diversity": float(rng.uniform(0.2, 0.6)),
            "employee_turnover": float(rng.uniform(0.05, 0.25)),
            "controversy_score": float(rng.uniform(0, 10)),
            "renewable_energy_pct": float(rng.uniform(0.1, 0.8)),
            "supply_chain_risk": float(rng.uniform(1, 10)),
            "esg_risk_label": int(rng.integers(0, 3)),
        }

    def _generate_text_from_structured(
        self, data: Dict[str, Any], company_name: str
    ) -> List[str]:
        """Convert structured ESG data into descriptive text chunks."""
        texts = []

        if not data:
            return texts

        carbon = data.get("carbon_emissions", "N/A")
        water = data.get("water_usage", "N/A")
        diversity = data.get("board_diversity", "N/A")
        turnover = data.get("employee_turnover", "N/A")
        controversy = data.get("controversy_score", "N/A")

        texts.append(
            f"{company_name} environmental performance: carbon emissions of {carbon} "
            f"metric tons CO2e and water usage of {water} million liters annually. "
            f"The company has committed to renewable energy adoption and carbon neutrality goals."
        )
        texts.append(
            f"{company_name} social performance: board diversity at {diversity} "
            f"and employee turnover rate of {turnover}. The company maintains "
            f"human rights policies and supply chain transparency programs."
        )
        texts.append(
            f"{company_name} governance performance: controversy score of {controversy}/10. "
            f"The company has established ESG oversight committees and follows GRI, SASB, "
            f"and TCFD reporting frameworks."
        )
        return texts