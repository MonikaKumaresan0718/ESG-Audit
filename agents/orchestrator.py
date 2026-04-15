"""
OrchestratorAgent – Coordinates the ESG audit workflow and manages task execution.
Uses CrewAI to define the planner/orchestrator agent with sequential task execution.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from crewai import Agent, Crew, Task
from crewai.process import Process

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class OrchestratorAgent:
    """
    Orchestrates the full ESG audit pipeline.
    Coordinates DataIngestion → ZeroShot → MLRisk → HybridFusion
    → Validation → ReportGeneration agents in a sequential workflow.
    """

    def __init__(self):
        self.audit_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.agent = None

    def _build_crewai_agent(self) -> Agent:
        """Construct the CrewAI orchestrator agent definition."""
        return  None

    def run_audit_pipeline(
        self,
        company_name: str,
        esg_data: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
        csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full ESG audit pipeline for a given company.

        Args:
            company_name: Name of the company being audited.
            esg_data: Optional pre-loaded structured ESG data dict.
            pdf_path: Optional path to a sustainability report PDF.
            csv_path: Optional path to ESG metrics CSV.

        Returns:
            Comprehensive ESG audit result dictionary.
        """
        self.audit_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()

        logger.info(
            "Starting ESG audit pipeline",
            extra={
                "audit_id": self.audit_id,
                "company": company_name,
                "has_pdf": pdf_path is not None,
                "has_csv": csv_path is not None,
            },
        )

        context = {
            "audit_id": self.audit_id,
            "company_name": company_name,
            "start_time": self.start_time.isoformat(),
            "esg_data": esg_data,
            "pdf_path": pdf_path,
            "csv_path": csv_path,
            "pipeline_stages": [],
        }

        try:
            # Stage 1: Data Ingestion
            context = self._stage_data_ingestion(context)

            # Stage 2: Zero-Shot Analysis
            context = self._stage_zero_shot_analysis(context)

            # Stage 3: ML Risk Modeling
            context = self._stage_ml_risk_modeling(context)

            # Stage 4: Hybrid Fusion
            context = self._stage_hybrid_fusion(context)

            # Stage 5: Validation & Explainability
            context = self._stage_validation(context)

            # Stage 6: Report Generation
            context = self._stage_report_generation(context)

            context["status"] = "completed"
            context["end_time"] = datetime.utcnow().isoformat()
            context["duration_seconds"] = (
                datetime.utcnow() - self.start_time
            ).total_seconds()

            logger.info(
                "ESG audit pipeline completed successfully",
                extra={"audit_id": self.audit_id, "company": company_name},
            )

        except Exception as exc:
            logger.error(
                "ESG audit pipeline failed",
                extra={"audit_id": self.audit_id, "error": str(exc)},
                exc_info=True,
            )
            context["status"] = "failed"
            context["error"] = str(exc)

        return context

    def _stage_data_ingestion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the DataIngestionAgent stage."""
        from agents.data_ingestion import DataIngestionAgent

        logger.info("Stage 1/6: Data Ingestion", extra={"audit_id": self.audit_id})
        agent = DataIngestionAgent()

        ingestion_result = agent.ingest(
            company_name=context["company_name"],
            csv_path=context.get("csv_path"),
            pdf_path=context.get("pdf_path"),
            esg_data=context.get("esg_data"),
        )

        context["ingestion"] = ingestion_result
        context["pipeline_stages"].append("data_ingestion")
        return context

    def _stage_zero_shot_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the ZeroShotAnalyzerAgent stage."""
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        logger.info("Stage 2/6: Zero-Shot Analysis", extra={"audit_id": self.audit_id})
        agent = ZeroShotAnalyzerAgent()

        texts = context.get("ingestion", {}).get("texts", [])
        zero_shot_result = agent.analyze(
            texts=texts,
            company_name=context["company_name"],
        )

        context["zero_shot"] = zero_shot_result
        context["pipeline_stages"].append("zero_shot_analysis")
        return context

    def _stage_ml_risk_modeling(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the MLRiskModelerAgent stage."""
        from agents.ml_risk_modeler import MLRiskModelerAgent

        logger.info("Stage 3/6: ML Risk Modeling", extra={"audit_id": self.audit_id})
        agent = MLRiskModelerAgent()

        structured_data = context.get("ingestion", {}).get("structured_data", {})
        ml_result = agent.predict(structured_data=structured_data)

        context["ml_risk"] = ml_result
        context["pipeline_stages"].append("ml_risk_modeling")
        return context

    def _stage_hybrid_fusion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the HybridFusionAgent stage."""
        from agents.hybrid_fusion import HybridFusionAgent

        logger.info("Stage 4/6: Hybrid Fusion", extra={"audit_id": self.audit_id})
        agent = HybridFusionAgent()

        fusion_result = agent.fuse(
            ml_result=context.get("ml_risk", {}),
            zero_shot_result=context.get("zero_shot", {}),
        )

        context["fusion"] = fusion_result
        context["pipeline_stages"].append("hybrid_fusion")
        return context

    def _stage_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the ValidationExplainabilityAgent stage."""
        from agents.validation_explainer import ValidationExplainabilityAgent

        logger.info(
            "Stage 5/6: Validation & Explainability",
            extra={"audit_id": self.audit_id},
        )
        agent = ValidationExplainabilityAgent()

        validation_result = agent.validate_and_explain(
            structured_data=context.get("ingestion", {}).get("structured_data", {}),
            ml_result=context.get("ml_risk", {}),
            fusion_result=context.get("fusion", {}),
        )

        context["validation"] = validation_result
        context["pipeline_stages"].append("validation_explainability")
        return context

    def _stage_report_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the ReportGeneratorAgent stage."""
        from agents.report_generator import ReportGeneratorAgent

        logger.info("Stage 6/6: Report Generation", extra={"audit_id": self.audit_id})
        agent = ReportGeneratorAgent()

        report_result = agent.generate(
            audit_id=self.audit_id,
            company_name=context["company_name"],
            ingestion=context.get("ingestion", {}),
            zero_shot=context.get("zero_shot", {}),
            ml_risk=context.get("ml_risk", {}),
            fusion=context.get("fusion", {}),
            validation=context.get("validation", {}),
        )

        context["report"] = report_result
        context["pipeline_stages"].append("report_generation")
        return context

    def build_crewai_crew(
        self,
        company_name: str,
        csv_path: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> Crew:
        """
        Build a full CrewAI Crew for demonstration/integration.
        Defines all agents and tasks in a sequential process.
        """
        from agents.data_ingestion import DataIngestionAgent
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent
        from agents.ml_risk_modeler import MLRiskModelerAgent
        from agents.hybrid_fusion import HybridFusionAgent
        from agents.validation_explainer import ValidationExplainabilityAgent
        from agents.report_generator import ReportGeneratorAgent

        # Build sub-agents
        ingestion_agent = DataIngestionAgent().agent
        zero_shot_agent = ZeroShotAnalyzerAgent().agent
        ml_agent = MLRiskModelerAgent().agent
        fusion_agent = HybridFusionAgent().agent
        validation_agent = ValidationExplainabilityAgent().agent
        report_agent = ReportGeneratorAgent().agent

        # Define tasks
        task_ingest = Task(
            description=f"Ingest ESG data for {company_name} from CSV={csv_path}, PDF={pdf_path}.",
            agent=ingestion_agent,
            expected_output="Structured ESG data dict with texts and embeddings stored.",
        )
        task_zero_shot = Task(
            description=f"Run zero-shot NLP classification on {company_name} sustainability texts.",
            agent=zero_shot_agent,
            expected_output="JSON with ESG category confidence scores and emerging risks.",
        )
        task_ml = Task(
            description=f"Predict ML-based ESG risk score for {company_name}.",
            agent=ml_agent,
            expected_output="Risk score, tier, and feature importances.",
        )
        task_fusion = Task(
            description="Combine ML predictions and zero-shot insights into composite ESG score.",
            agent=fusion_agent,
            expected_output="Composite ESG score (0–100) and risk tier.",
        )
        task_validation = Task(
            description="Validate ESG scores against GRI/SASB/TCFD thresholds and generate SHAP explanations.",
            agent=validation_agent,
            expected_output="Validation flags, confidence intervals, and SHAP values.",
        )
        task_report = Task(
            description=f"Generate a comprehensive ESG audit report for {company_name} in JSON and Markdown.",
            agent=report_agent,
            expected_output="ESG audit report files (JSON, Markdown, optional PDF).",
        )

        crew = Crew(
            agents=[
                self.agent,
                ingestion_agent,
                zero_shot_agent,
                ml_agent,
                fusion_agent,
                validation_agent,
                report_agent,
            ],
            tasks=[
                task_ingest,
                task_zero_shot,
                task_ml,
                task_fusion,
                task_validation,
                task_report,
            ],
            process=Process.sequential,
            verbose=settings.VERBOSE_AGENTS,
        )
        return crew
