"""Operational condition orchestrator for condition-aware data splitting.

Splits time-series data into operational condition bands (power x head) so that
each data-driven model is trained and evaluated on a homogeneous operating regime.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "operational_conditions.json"


@dataclass
class OperationalConditionConfig:
    """Configuration for operational condition splitting."""

    power_tag: str
    head_tag: str
    power_thresholds: list[float]
    head_thresholds: list[float]
    min_training_points: int


class OperationalConditionOrchestrator:
    """Splits DataFrames by operational condition (power band x head band).

    Power thresholds [t1, t2, t3] create 4 bands: P1 (<t1), P2 (t1<=x<t2), P3 (t2<=x<t3), P4 (>=t3).
    Head thresholds [t1, t2] create 3 bands: Q1 (<t1), Q2 (t1<=x<t2), Q3 (>=t2).
    Combined: 12 labels P1_Q1 .. P4_Q3.
    """

    def __init__(self, config: OperationalConditionConfig) -> None:
        """Initialize with an operational condition configuration."""
        self.config = config

    @classmethod
    def from_json(cls, key: str, json_path: Optional[str | Path] = None) -> "OperationalConditionOrchestrator":
        """Load config by key (e.g. 'UGA') from JSON file.

        Args:
            key: Plant/UG identifier.
            json_path: Path to JSON config. Defaults to data/operational_conditions.json.

        Returns:
            Configured orchestrator instance.

        Raises:
            KeyError: If key not found in config file.

        """
        path = Path(json_path) if json_path else _DEFAULT_CONFIG_PATH
        with open(path) as f:
            all_configs = json.load(f)

        if key not in all_configs:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(all_configs.keys())}")

        raw = all_configs[key]
        config = OperationalConditionConfig(
            power_tag=raw["power_tag"],
            head_tag=raw["head_tag"],
            power_thresholds=raw["power_thresholds"],
            head_thresholds=raw["head_thresholds"],
            min_training_points=raw["min_training_points"],
        )
        return cls(config)

    def compute_head(self, df: pd.DataFrame) -> pd.Series:
        """Return the net head series from the configured head tag.

        Args:
            df: DataFrame containing the head tag column.

        Returns:
            Series with head values.

        """
        return df[self.config.head_tag]

    def classify_single(self, power: float, head: float) -> str:
        """Hard-assign one observation to a condition label.

        Args:
            power: Power value.
            head: Net head value.

        Returns:
            Condition label like 'P1_Q1' or 'UNCLASSIFIED' if NaN.

        """
        if np.isnan(power) or np.isnan(head):
            return "UNCLASSIFIED"

        p_band = self._assign_band(power, self.config.power_thresholds, "P")
        q_band = self._assign_band(head, self.config.head_thresholds, "Q")
        return f"{p_band}_{q_band}"

    def assign_conditions(self, df: pd.DataFrame) -> pd.Series:
        """Label every row with its operational condition.

        Args:
            df: DataFrame containing power and head tag columns.

        Returns:
            Series of condition labels (e.g. 'P2_Q1', 'UNCLASSIFIED').

        """
        power = df[self.config.power_tag]
        head = self.compute_head(df)

        p_bands = self._assign_band_vectorized(power, self.config.power_thresholds, "P")
        q_bands = self._assign_band_vectorized(head, self.config.head_thresholds, "Q")

        # Combine bands; mark NaN rows as UNCLASSIFIED
        nan_mask = power.isna() | head.isna()
        labels = p_bands + "_" + q_bands
        labels[nan_mask] = "UNCLASSIFIED"

        return labels

    def split_for_training(self, df: pd.DataFrame, sensor_columns: list[str]) -> dict[str, pd.DataFrame]:
        """Split data by condition for training, enforcing min_training_points.

        Takes the first ``min_training_points`` rows per condition.
        Conditions with insufficient data are skipped with a warning.

        Args:
            df: Full DataFrame with sensor + power/head columns.
            sensor_columns: Columns to include in output (excludes power/head metadata).

        Returns:
            Dict mapping condition label to sensor-only DataFrame.

        """
        conditions = self.assign_conditions(df)
        splits: dict[str, pd.DataFrame] = {}
        min_pts = self.config.min_training_points

        for label in self.get_all_condition_labels():
            mask = conditions == label
            cond_df = df.loc[mask, sensor_columns]

            if len(cond_df) < min_pts:
                logger.warning(f"Condition {label}: {len(cond_df)} points < min {min_pts}, skipping for training")
                continue

            splits[label] = cond_df.iloc[:min_pts].copy()

        return splits

    def split_for_detection(self, df: pd.DataFrame, sensor_columns: list[str]) -> dict[str, pd.DataFrame]:
        """Split data by condition for detection (no min-points check, no truncation).

        Preserves original DatetimeIndex for recombination.

        Args:
            df: Full DataFrame with sensor + power/head columns.
            sensor_columns: Columns to include in output.

        Returns:
            Dict mapping condition label to sensor-only DataFrame (preserving index).

        """
        conditions = self.assign_conditions(df)
        splits: dict[str, pd.DataFrame] = {}

        for label in self.get_all_condition_labels():
            mask = conditions == label
            cond_df = df.loc[mask, sensor_columns]
            if len(cond_df) > 0:
                splits[label] = cond_df

        return splits

    def get_all_condition_labels(self) -> list[str]:
        """Return all 12 condition labels in order.

        Returns:
            List of labels: ['P1_Q1', 'P1_Q2', ..., 'P4_Q3'].

        """
        n_power = len(self.config.power_thresholds) + 1
        n_head = len(self.config.head_thresholds) + 1
        labels = []
        for p in range(1, n_power + 1):
            for q in range(1, n_head + 1):
                labels.append(f"P{p}_Q{q}")
        return labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_band(value: float, thresholds: list[float], prefix: str) -> str:
        """Assign a single value to a band.

        Bands use < for lower thresholds: band 1 is value < t[0],
        band 2 is t[0] <= value < t[1], ..., last band is value >= t[-1].
        """
        for i, t in enumerate(thresholds):
            if value < t:
                return f"{prefix}{i + 1}"
        return f"{prefix}{len(thresholds) + 1}"

    @staticmethod
    def _assign_band_vectorized(series: pd.Series, thresholds: list[float], prefix: str) -> pd.Series:
        """Vectorized band assignment for a Series."""
        # Start with last band
        result = pd.Series(f"{prefix}{len(thresholds) + 1}", index=series.index)
        # Assign from highest threshold down so lower thresholds overwrite
        for i in range(len(thresholds) - 1, -1, -1):
            result[series < thresholds[i]] = f"{prefix}{i + 1}"
        return result
