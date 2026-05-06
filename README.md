# ESG Audit Platform

An AI-powered platform for automated Environmental, Social, and Governance (ESG) auditing, risk assessment, and report generation.

## Overview

The ESG Audit Platform streamlines ESG compliance analysis by combining machine learning, natural language processing, and interactive dashboards. It helps organizations evaluate ESG performance, identify risks, and generate comprehensive audit reports.

## Key Features

* Automated ESG data ingestion and processing
* Zero-shot classification for ESG issue detection
* Machine learning–based risk scoring
* Interactive dashboards and visual analytics
* Automated report generation
* Background task processing with Celery
* Vector search using FAISS for semantic analysis

## Technology Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Task Queue:** Celery with Redis
* **Database:** SQLAlchemy / SQLite
* **Machine Learning:** Hugging Face Transformers, Sentence Transformers
* **Vector Database:** FAISS
* **Testing:** Pytest

## Project Structure

```text
esg-audit/
├── core/                # Core configuration, database, logging, Celery setup
├── ui/                  # Streamlit frontend
│   ├── components/      # Reusable UI components
│   └── pages/           # Multipage app screens
├── tasks/               # Background task pipelines
├── tools/               # ML and NLP utilities
├── tests/               # Unit and integration tests
├── data/                # Generated indexes and datasets
├── logs/                # Application logs
└── requirements.txt     # Python dependencies
```

## Installation

```bash
git clone https://github.com/MonikaKumaresan0718/ESG-Audit.git
cd ESG-Audit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

### Start Redis

```bash
redis-server
```

### Start the FastAPI Backend

```bash
uvicorn api.main:app --reload
```

### Start the Celery Worker

```bash
celery -A core.celery_app:celery_app worker --loglevel=info
```

### Launch the Streamlit Frontend

```bash
streamlit run ui/app.py
```

## Usage

1. Launch the application.
2. Submit a company for ESG audit.
3. Monitor audit progress in the dashboard.
4. Review generated ESG scores and risk insights.
5. View or download the final audit report.

## Core Capabilities

* ESG risk assessment
* Compliance monitoring
* Automated scoring and benchmarking
* Report generation and visualization
* Semantic document analysis

## Future Enhancements

* Real-time ESG news integration
* Advanced predictive analytics
* Multi-user authentication and role-based access
* Cloud deployment support
* Export to PDF and Excel

## Contributors
- Monika Kumaresan
- Anusri Baskaran (Github:https://github.com/anusribaskaran2-arch)
- 
## Team Contributions

**Monika Kumaresan**
- Designed multi-agent ESG pipeline architecture and workflow orchestration
- Implemented RAG system using FAISS and LLM integration
- Built backend APIs (FastAPI) and async task processing (Celery, Redis)
- Integrated explainability (SHAP) and model inference pipeline

**Anusri baskaran**
- Designed data ingestion flow and ESG data preprocessing pipeline
- Engineered prompts and LLM interaction logic for ESG analysis
- Developed Streamlit UI and report generation module
- Led testing, validation, and deployment setup (Docker, environment config)
- 
## License

This project is licensed under the MIT License.

## Author

**Monika Kumaresan**

GitHub: [https://github.com/MonikaKumaresan0718](https://github.com/MonikaKumaresan0718)
