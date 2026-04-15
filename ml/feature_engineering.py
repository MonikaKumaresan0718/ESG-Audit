"""
ESG Feature Engineering Pipeline.
Transforms raw ESG metrics into ML-ready features.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

# Default feature columns expected in the ESG dataset
ESG_FEATURE_COLUMNS = [
    "carbon_emissions",
    "water_usage",
    "board_diversity",
    "employee_turnover",
    "controversy_score",
    "renewable_energy_pct",
    "supply_chain_risk",
]

ESG_TARGET_COLUMN = "esg_risk_label"


class ESGFeatureEngineer:
    """
    Transforms raw ESG data dicts into engineered feature DataFrames.
    Implements normalization, interaction features, and risk proxies.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or ESG_FEATURE_COLUMNS
        self._scaler = None
        self._fitted = False

    def transform(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform a single ESG data dict into a feature DataFrame.

        Args:
            data: Dict of ESG metric values.

        Returns:
            DataFrame with engineered features.
        """
        # Extract base features
        base_features = self._extract_base_features(data)

        # Add interaction features
        interaction_features = self._compute_interaction_features(base_features)

        # Add risk proxies
        risk_proxies = self._compute_risk_proxies(base_features)

        # Combine all features
        all_features = {**base_features, **interaction_features, **risk_proxies}

        df = pd.DataFrame([all_features])

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(0)

        return df

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame of ESG records.

        Args:
            df: Input DataFrame with ESG columns.

        Returns:
            Feature-engineered DataFrame.
        """
        records = df.to_dict(orient="records")
        feature_records = [
            {
                **self._extract_base_features(r),
                **self._compute_interaction_features(self._extract_base_features(r)),
                **self._compute_risk_proxies(self._extract_base_features(r)),
            }
            for r in records
        ]
        result = pd.DataFrame(feature_records)
        result = result.fillna(result.median(numeric_only=True))
        result = result.fillna(0)
        return result

    def _extract_base_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and validate base ESG features."""
        features = {}
        for col in self.feature_columns:
            value = data.get(col, None)
            if value is not None:
                try:
                    features[col] = float(value)
                except (ValueError, TypeError):
                    features[col] = 0.0
            else:
                features[col] = self._default_value(col)
        return features

    def _default_value(self, column: str) -> float:
        """Return sensible default for missing feature."""
        defaults = {
            "carbon_emissions": 200.0,
            "water_usage": 500.0,
            "board_diversity": 0.35,
            "employee_turnover": 0.15,
            "controversy_score": 4.0,
            "renewable_energy_pct": 0.25,
            "supply_chain_risk": 5.0,
        }
        return defaults.get(column, 0.0)

    def _compute_interaction_features(
        self, base: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute interaction features between ESG dimensions."""
        carbon = base.get("carbon_emissions", 200)
        water = base.get("water_usage", 500)
        diversity = base.get("board_diversity", 0.35)
        turnover = base.get("employee_turnover", 0.15)
        controversy = base.get("controversy_score", 4)
        renewable = base.get("renewable_energy_pct", 0.25)
        supply = base.get("supply_chain_risk", 5)

        return {
            # Environmental composite
            "env_risk_composite": (carbon / 500 + water / 1000) / 2,
            # Social composite
            "social_risk_composite": (turnover / 0.3 + supply / 10) / 2,
            # Governance composite
            "gov_risk_composite": (1 - diversity) * controversy / 10,
            # Carbon intensity proxy
            "carbon_water_ratio": carbon / max(water, 1),
            # Renewable offset
            "carbon_adjusted": carbon * (1 - renewable),
            # Diversity-controversy interaction
            "div_controversy": (1 - diversity) * controversy,
            # Turnover-supply interaction
            "turnover_supply": turnover * supply / 10,
        }

    def _compute_risk_proxies(
        self, base: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute high-level ESG risk proxy scores (0–1)."""
        carbon = base.get("carbon_emissions", 200)
        water = base.get("water_usage", 500)
        diversity = base.get("board_diversity", 0.35)
        turnover = base.get("employee_turnover", 0.15)
        controversy = base.get("controversy_score", 4)
        renewable = base.get("renewable_energy_pct", 0.25)
        supply = base.get("supply_chain_risk", 5)

        # Normalize to 0–1 risk scores
        env_risk = min(
            (carbon / 500 * 0.5 + water / 1000 * 0.3 + (1 - renewable) * 0.2), 1.0
        )
        social_risk = min(
            (turnover / 0.3 * 0.4 + (1 - diversity) * 0.3 + supply / 10 * 0.3), 1.0
        )
        gov_risk = min(
            (controversy / 10 * 0.6 + (1 - diversity) * 0.4), 1.0
        )
        overall_risk = (env_risk * 0.35 + social_risk * 0.35 + gov_risk * 0.30)

        return {
            "env_risk_proxy": round(env_risk, 4),
            "social_risk_proxy": round(social_risk, 4),
            "gov_risk_proxy": round(gov_risk, 4),
            "overall_risk_proxy": round(overall_risk, 4),
        }

    def get_feature_names(self) -> List[str]:
        """Return all feature names after transformation."""
        dummy = self._extract_base_features({})
        interaction = self._compute_interaction_features(dummy)
        proxies = self._compute_risk_proxies(dummy)
        return list({**dummy, **interaction, **proxies}.keys())


def load_and_prepare_dataset(
    csv_path: str,
    target_col: str = ESG_TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load CSV, engineer features, and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from {csv_path}")

    engineer = ESGFeatureEngineer()
    X = engineer.transform_batch(df)

    if target_col in df.columns:
        y = df[target_col].astype(int)
    else:
        # Derive synthetic labels from risk proxies
        y = (X["overall_risk_proxy"] * 3).astype(int).clip(0, 2)
        logger.warning("Target column not found; using synthetic labels")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test