"""Reference Moving Window Principal Component Analysis with Schur Complement (RMWPCA_SC) implementation of PCA."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from fault_diagnosis.technique import BaseDetector


class RMWPCA_SC(TransformerMixin, BaseDetector):
    """Robust Moving Window Principal Component Analysis with Schur Complement (RMWPCA_SC) class."""

    # Default configuration constants
    DEFAULT_WINDOW_SIZE: int = 10
    DEFAULT_NUM_COMPONENTS: int = 3
    DEFAULT_NUM_STDS: int = 3
    DEFAULT_ROLLING_WINDOW: int = 6
    PREDICTION_ROLLING_WINDOW: int = 200
    PREDICTION_HORIZONS: Dict[str, int] = {
        "less_than_15_days": 15,
        "less_than_30_days": 30,
        "less_than_180_days": 180,
    }

    def __init__(self) -> None:
        """Initialize the RMWPCA_SC class with default values for attributes."""
        self.eigenvectors: Optional[np.ndarray] = None
        self.ref_data: Optional[np.ndarray] = None
        self.health_data: Optional[pd.DataFrame] = None
        self.window_size: Optional[int] = None
        self.num_components: Optional[int] = None
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_centered: Optional[np.ndarray] = None
        self.ref_covariance: Optional[np.ndarray] = None
        self.limits: Optional[Dict[str, Any]] = None
        self.model: Optional[Any] = None  # For compatibility with BaseDetector explainer methods

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> "RMWPCA_SC":
        """Fit the RMWPCA_SC model to the provided health and reference data.

        Args:
            data (pd.DataFrame): Health data to fit the model.
            **kwargs: Additional parameters:
                - ref_data (Optional[np.ndarray]): Reference data for PCA. Default is None.
                - window_size (int): Size of the moving window. Default is DEFAULT_WINDOW_SIZE.

        Returns:
            RMWPCA_SC: The fitted RMWPCA_SC model.

        """
        ref_data: Optional[np.ndarray] = kwargs.get("ref_data", None)
        window_size: int = kwargs.get("window_size", self.DEFAULT_WINDOW_SIZE)

        # Calculate num_components dynamically based on number of features
        n_features = data.shape[1]
        num_components = min(self.DEFAULT_NUM_COMPONENTS, n_features)

        self.ref_data = ref_data
        self.health_data = data
        self.window_size = window_size
        self.num_components = num_components

        if ref_data is not None:
            self.ref_mean = np.mean(ref_data, axis=0)
            self.ref_centered = ref_data - self.ref_mean
            self.ref_covariance = np.cov(self.ref_centered, rowvar=False)

        self.limits = self._calculate_health_limits(data, window_size)

        return self

    def _perform_pca(self, data: np.ndarray, cov_matrix: np.ndarray, num_components: int) -> tuple:
        """Perform PCA on the provided data.

        Args:
            data (np.ndarray): Data to perform PCA on.
            cov_matrix (np.ndarray): Covariance matrix of the data.
            num_components (int): Number of principal components to retain.

        Returns:
            tuple: Transformed data, eigenvalues (top-k), eigenvectors, inverse eigenvalues, and full eigenvalues.

        Raises:
            ValueError: If covariance matrix contains invalid values or is ill-conditioned.

        """
        # Validate covariance matrix
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            raise ValueError("Covariance matrix contains NaN or Inf values")

        # Add regularization to improve numerical stability
        regularization = 1e-8
        cov_matrix_reg = cov_matrix + regularization * np.eye(cov_matrix.shape[0])

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_reg)
        except np.linalg.LinAlgError as e:
            # Fallback to SVD if eigenvalue decomposition fails
            try:
                U, S, Vt = np.linalg.svd(cov_matrix_reg, hermitian=True)
                eigenvalues = S
                eigenvectors = U
            except np.linalg.LinAlgError:
                raise ValueError(f"Failed to decompose covariance matrix: {e}") from e

        # Sort eigenvalues and eigenvectors in descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Preserve full eigenvalues for PCW normalization (sum of ALL eigenvalues)
        full_eigenvalues = eigenvalues.copy()

        # Select only the first num_components principal components
        eigenvalues = eigenvalues[:num_components]
        eigenvectors = eigenvectors[:, :num_components]

        # Project the data into the principal component space
        transformed_data = data @ eigenvectors
        eigenvalues = np.where(eigenvalues > 1e-10, eigenvalues, 1e-10)
        eigenvalues_inv = np.diag(1 / eigenvalues)

        return transformed_data, eigenvalues, eigenvectors, eigenvalues_inv, full_eigenvalues

    def transform(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Transform the current data using the fitted PCA model.

        Args:
            current_data (pd.DataFrame): Current data to transform.

        Returns:
            Dict[str, Any]: Calculated metrics.

        Raises:
            ValueError: If current_data contains NaN or Inf values.
            ValueError: If model has not been fitted (num_components is None).

        """
        if self.num_components is None:
            raise ValueError("Model has not been fitted. Call fit() before transform().")
        num_components = self.num_components

        # Validate input data
        if current_data.isnull().any().any():
            nan_counts = current_data.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            raise ValueError(
                f"Current data contains NaN values. Columns with NaNs: {nan_cols.to_dict()}. "
                "Please configure PISystem with handle_nans=True or clean the data before transformation."
            )
        if np.any(np.isinf(current_data.values)):
            raise ValueError("Current data contains Inf values")

        # Center the current data
        current_mean = np.mean(current_data, axis=0)
        current_centered = current_data - current_mean
        input_data = current_centered.copy()

        # Augmented matrix Z = [Y | X]
        if self.ref_data is not None:
            input_data = np.hstack((self.ref_centered, input_data))

        cov_matrix = np.cov(input_data, rowvar=False)
        current_covariance = np.cov(current_centered, rowvar=False)

        transformed_data, eigenvalues, eigenvectors, eigenvalues_inv, full_eigenvalues = self._perform_pca(
            input_data, cov_matrix, num_components
        )

        metrics = self._calculate_metrics(
            input_data=input_data,
            current_centered=current_centered,
            transformed_data=transformed_data,
            eigenvectors=eigenvectors,
            eigenvalues_inv=eigenvalues_inv,
            eigenvalues=eigenvalues,
            full_eigenvalues=full_eigenvalues,
            cov_matrix=current_covariance,
            num_components=num_components,
        )

        metrics["index"] = current_data.index[-1]
        return metrics

    def _calculate_t2(self, transformed_data: np.ndarray, eigenvalues_inv: np.ndarray) -> float:
        """Calculate the T² metric.

        Args:
            transformed_data (np.ndarray): Transformed data.
            eigenvalues_inv (np.ndarray): Inverse eigenvalues.

        Returns:
            float: Calculated T² metric.

        """
        T2_vector = np.array([transformed_data @ eigenvalues_inv @ transformed_data.T])
        return abs(T2_vector - T2_vector.mean()).mean()

    def _calculate_q(self, input_data: np.ndarray, transformed_data: np.ndarray, eigenvectors: np.ndarray) -> float:
        """Calculate the Q metric.

        Args:
            input_data (np.ndarray): Input data.
            transformed_data (np.ndarray): Transformed data.
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            float: Calculated Q metric.

        """
        Z_reconstructed = self.inverse_transform(transformed_data, eigenvectors)
        residuals = input_data - Z_reconstructed
        Q_vector = np.sum(residuals**2, axis=1)
        return abs(Q_vector - Q_vector.mean()).mean()

    def calculate_reconstruction_error_per_feature(
        self, input_data: np.ndarray, transformed_data: np.ndarray, eigenvectors: np.ndarray
    ) -> np.ndarray:
        """Calculate reconstruction error contribution per feature.

        Args:
            input_data (np.ndarray): Original input data.
            transformed_data (np.ndarray): Transformed data (in PCA space).
            eigenvectors (np.ndarray): Eigenvectors used for projection.

        Returns:
            np.ndarray: Mean squared error per feature across all samples.

        """
        Z_reconstructed = self.inverse_transform(transformed_data, eigenvectors)
        residuals = input_data - Z_reconstructed
        feature_errors = np.mean(residuals**2, axis=0)
        return feature_errors

    def _calculate_pcw(self, eigenvalues: np.ndarray, full_eigenvalues: np.ndarray, num_components: int) -> np.ndarray:
        """Calculate the PCW metric using direct eigenvalues normalized over the full spectrum.

        Args:
            eigenvalues (np.ndarray): Top-k direct eigenvalues (num_components).
            full_eigenvalues (np.ndarray): All eigenvalues from the decomposition (2×n_sensors).
            num_components (int): Number of principal components.

        Returns:
            np.ndarray: Calculated PCW metric (explained variance percentage per component).

        """
        selected = eigenvalues[:num_components]
        total = np.sum(full_eigenvalues)
        normalized_pcw = 100 * (selected / total)

        return normalized_pcw

    def _calculate_metrics(
        self,
        input_data: np.ndarray,
        current_centered: np.ndarray,
        transformed_data: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues_inv: np.ndarray,
        eigenvalues: np.ndarray,
        full_eigenvalues: np.ndarray,
        cov_matrix: np.ndarray,
        num_components: int,
    ) -> Dict[str, Any]:
        """Calculate the metrics for the current data.

        Args:
            input_data (np.ndarray): Input data.
            current_centered (np.ndarray): Centered current data.
            transformed_data (np.ndarray): Transformed data.
            eigenvectors (np.ndarray): Eigenvectors.
            eigenvalues_inv (np.ndarray): Inverse eigenvalues (used for T²).
            eigenvalues (np.ndarray): Top-k direct eigenvalues (used for PCW).
            full_eigenvalues (np.ndarray): All eigenvalues from decomposition (used for PCW normalization).
            cov_matrix (np.ndarray): Covariance matrix.
            num_components (int): Number of principal components.

        Returns:
            Dict[str, Any]: Calculated metrics.

        """
        # Projection of the current data into the principal component space
        current_transformed = current_centered @ eigenvectors[: current_centered.shape[1]]

        return {
            "T2": self._calculate_t2(transformed_data, eigenvalues_inv),
            "T2_current": self._calculate_t2(current_transformed, eigenvalues_inv),
            "Q": self._calculate_q(input_data, transformed_data, eigenvectors),
            "PCW": self._calculate_pcw(eigenvalues, full_eigenvalues, num_components),
            "schur": self._calculate_schur_complement(self.ref_covariance, cov_matrix, num_components),
        }

    def _calculate_schur_complement(
        self, ref_cov: Optional[np.ndarray], current_cov: np.ndarray, num_components: int
    ) -> float:
        """Calculate the Schur Complement metric.

        Args:
            ref_cov (Optional[np.ndarray]): Reference covariance matrix. If None, returns 0.
            current_cov (np.ndarray): Current covariance matrix.
            num_components (int): Number of principal components.

        Returns:
            float: Calculated Schur Complement metric.

        """
        # Guard against None reference covariance
        if ref_cov is None:
            return 0.0

        try:
            ref_cov_reduced = ref_cov[:num_components, :num_components]
            current_cov_reduced = current_cov[:num_components, :num_components]
            schur_complement = ref_cov_reduced - ref_cov_reduced @ np.linalg.inv(current_cov_reduced) @ ref_cov_reduced
            schur_metric = np.linalg.norm(schur_complement, ord="fro")  # Frobenius norm
        except np.linalg.LinAlgError:
            schur_metric = 0.0  # Define as zero if the matrix is not invertible
        except (IndexError, TypeError):
            schur_metric = 0.0  # There is no enough data to calculate the Schur Complement

        return schur_metric

    @staticmethod
    def inverse_transform(data: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        """Inverse transform the data using the provided eigenvectors.

        Args:
            data (np.ndarray): Data to inverse transform.
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            np.ndarray: Inverse transformed data.

        """
        return data @ eigenvectors.T

    def _predict_error(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate reconstruction error for SHAP explainer compatibility.

        This method computes the reconstruction error (Q metric) for each sample,
        which is used by the BaseDetector SHAP explainer methods.

        Args:
            data (pd.DataFrame): Input data to calculate reconstruction error for.

        Returns:
            np.ndarray: Array of reconstruction errors for each sample.

        """
        if self.num_components is None:
            raise ValueError("Model has not been fitted. Call fit() before _predict_error().")

        errors = []
        num_components = self.num_components

        # Use health data covariance for consistent reconstruction
        if self.health_data is not None and len(self.health_data) > num_components:
            health_centered = self.health_data.values - np.mean(self.health_data.values, axis=0)
            cov_matrix = np.cov(health_centered, rowvar=False)
        else:
            cov_matrix = None

        for i in range(len(data)):
            sample = data.iloc[[i]].values

            # Center the sample using the health data mean if available
            if self.health_data is not None:
                health_mean = np.mean(self.health_data.values, axis=0)
                sample_centered = sample - health_mean
            else:
                sample_centered = sample - np.mean(sample, axis=0)

            try:
                # Use pre-computed covariance or compute from sample
                if cov_matrix is None:
                    # Fallback: use identity-like covariance
                    current_cov = np.eye(sample_centered.shape[1])
                else:
                    current_cov = cov_matrix

                _, _, eigenvectors, _, _ = self._perform_pca(
                    np.vstack([sample_centered] * max(2, num_components + 1)),  # Need at least 2 samples for PCA
                    current_cov,
                    num_components,
                )

                # Project and reconstruct single sample
                transformed = sample_centered @ eigenvectors
                reconstructed = self.inverse_transform(transformed, eigenvectors)
                residuals = sample_centered - reconstructed
                error = float(np.sum(residuals**2))
            except (ValueError, np.linalg.LinAlgError):
                error = 0.0

            errors.append(error)

        return np.array(errors)

    def _calculate_health_limits(
        self, health_data: pd.DataFrame, window_size: int, num_stds: int = None
    ) -> Dict[str, Any]:
        """Calculate the limits for the health data.

        Args:
            health_data (pd.DataFrame): Health data.
            window_size (int): Size of the moving window.
            num_stds (int, optional): Number of standard deviations for limit calculation.
                Defaults to DEFAULT_NUM_STDS.

        Returns:
            Dict[str, Any]: Calculated limits including intermediate metric lists for plotting.

        """
        if num_stds is None:
            num_stds = self.DEFAULT_NUM_STDS

        metrics_list = []
        for i in range(len(health_data) - window_size):
            data_health_window = health_data.iloc[i : i + window_size]
            metrics = self.transform(data_health_window)
            metrics_list.append(metrics)

        T2_list = [metrics["T2"] for metrics in metrics_list]
        T2_current_list = [metrics["T2_current"] for metrics in metrics_list]
        Q_list = [metrics["Q"] for metrics in metrics_list]
        PCW_raw_list = [metrics["PCW"] for metrics in metrics_list]
        # Transpose PCW list: from list of arrays to list of lists per component
        PCW_list = [[arr[i] for arr in PCW_raw_list] for i in range(self.num_components)]
        schur_list = [metrics["schur"] for metrics in metrics_list]

        def calculate_limits(metric_list: list) -> Dict[str, float]:
            mean = np.mean(metric_list)
            std_dev = np.std(metric_list)
            return {"low": mean - num_stds * std_dev, "high": mean + num_stds * std_dev}

        def calculate_pcw_limits(pcw_list: list) -> Dict[str, float]:
            return {f"PCW_{i+1}": calculate_limits(component) for i, component in enumerate(pcw_list)}

        limits = {
            "T2": calculate_limits(T2_list),
            "T2_current": calculate_limits(T2_current_list),
            "Q": calculate_limits(Q_list),
            "PCW": calculate_pcw_limits(PCW_list),
            "schur": calculate_limits(schur_list),
            "T2_list": T2_list,
            "T2_current_list": T2_current_list,
            "Q_list": Q_list,
            "PCW_list": PCW_raw_list,  # Original format for plotting
            "schur_list": schur_list,
        }

        return limits

    def calculate_health_limits(
        self, health_data: pd.DataFrame, window_size: int, num_stds: int = None
    ) -> Dict[str, Any]:
        """Calculate the limits for the health data (public interface).

        Args:
            health_data (pd.DataFrame): Health data.
            window_size (int): Size of the moving window.
            num_stds (int, optional): Number of standard deviations for limit calculation.
                Defaults to DEFAULT_NUM_STDS.

        Returns:
            Dict[str, Any]: Calculated limits including intermediate metric lists for plotting.

        """
        return self._calculate_health_limits(health_data, window_size, num_stds)

    def transform_current_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Transform the current data using the fitted PCA model.

        Args:
            data (pd.DataFrame): Current data to transform.

        Returns:
            Dict[str, Any]: Transformed data metrics.

        """
        metrics_list = []
        for i in range(len(data) - self.window_size):
            current_data = data.iloc[i : i + self.window_size]
            metrics = self.transform(current_data)
            metrics_list.append(metrics)

        return {
            "index": [metrics["index"] for metrics in metrics_list],
            "T2": [metrics["T2"] for metrics in metrics_list],
            "T2_current": [metrics["T2_current"] for metrics in metrics_list],
            "Q": [metrics["Q"] for metrics in metrics_list],
            "PCW": [metrics["PCW"] for metrics in metrics_list],
            "schur": [metrics["schur"] for metrics in metrics_list],
        }

    def explain_reconstruction_error(self, data: pd.DataFrame, feature_names: list[str] = None) -> Dict[str, Any]:
        """Explain reconstruction error by calculating per-feature contributions.

        Args:
            data (pd.DataFrame): Current data to explain (typically the most recent window).
            feature_names (list[str], optional): Names of the features. If None, uses data column names.

        Returns:
            Dict[str, Any]: Dictionary containing feature contributions and visualization data.

        """
        # Check if we have enough data
        n_samples = len(data)
        if n_samples < self.window_size:
            raise ValueError(
                f"Data has only {n_samples} samples but model was trained with window_size={self.window_size}. "
                f"Need at least {self.window_size} samples for explanation. "
                f"This typically happens when data aggregation returns fewer samples than expected."
            )

        # Use the last window for explanation
        current_data = data.iloc[-self.window_size :].copy()  # Use copy to avoid SettingWithCopyWarning

        # Convert to numpy for calculations to avoid pandas alignment issues
        current_values = current_data.values
        current_mean = np.mean(current_values, axis=0)
        current_centered = current_values - current_mean

        # Prepare input data
        input_data = current_centered.copy()
        if self.ref_data is not None:
            input_data = np.hstack((self.ref_centered, input_data))

        # Perform PCA transformation
        cov_matrix = np.cov(input_data, rowvar=False)
        transformed_data, _, eigenvectors, _, _ = self._perform_pca(input_data, cov_matrix, self.num_components)

        # Calculate per-feature reconstruction error
        feature_errors = self.calculate_reconstruction_error_per_feature(input_data, transformed_data, eigenvectors)

        # Extract errors for current data features only (not reference data)
        if self.ref_data is not None:
            n_ref_features = self.ref_centered.shape[1]
            feature_errors = feature_errors[n_ref_features:]

        # Get feature names - use current_data columns to match the window used for calculation
        if feature_names is None:
            feature_names = (
                list(current_data.columns)
                if isinstance(current_data, pd.DataFrame)
                else [f"Feature_{i}" for i in range(len(feature_errors))]
            )
        else:
            # Ensure feature_names match the size of feature_errors
            # feature_names is from the full data, but we only use current_data window
            feature_names = list(current_data.columns) if isinstance(current_data, pd.DataFrame) else feature_names

        # Validate that feature_names and feature_errors have matching lengths
        if len(feature_names) != len(feature_errors):
            raise ValueError(
                f"Length mismatch: feature_names has {len(feature_names)} items but "
                f"feature_errors has {len(feature_errors)} items. "
                f"This may be due to feature extraction or reference data processing. "
                f"Using current_data window with {len(current_data.columns)} columns."
            )

        # Calculate total Q metric for context
        total_q = np.sum(feature_errors)

        # Calculate percentage contributions
        feature_contributions = [
            {
                "feature": name,
                "error": float(error),
                "percentage": float(error / total_q * 100) if total_q > 0 else 0.0,
            }
            for name, error in zip(feature_names, feature_errors)
        ]

        # Sort by error magnitude (descending)
        feature_contributions = sorted(feature_contributions, key=lambda x: x["error"], reverse=True)

        return {
            "total_Q": float(total_q),
            "Q_threshold": self.limits["Q"] if self.limits else None,
            "feature_contributions": feature_contributions,
            "n_features": len(feature_names),
            "timestamp": current_data.index[-1] if hasattr(current_data, "index") else None,
        }

    def __repr__(self) -> str:
        """Return the string representation of the RMWPCA_SC class."""
        return f"RMWPCA_SC(num_components={self.num_components}, window_size={self.window_size})"

    def predict(
        self,
        transformed_data: Dict[str, Any],
        metric: str = "Q",
        threshold: float = 0.5,
        rolling_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict the fault probability using rolling boolean method.

        This method aligns with the BaseDetector.predict() interface while providing
        additional probability distribution over time horizons.

        Args:
            transformed_data (Dict[str, Any]): Transformed data metrics.
            metric (str, optional): Metric to use for prediction. Defaults to "Q".
            threshold (float, optional): Threshold for fault probability. Defaults to 0.5.
            rolling_window (int, optional): Size of the rolling window.
                Defaults to DEFAULT_ROLLING_WINDOW.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'fault_detected': Boolean indicating if fault probability exceeds threshold
                - 'probability_distribution': Probability distribution over time horizons
                - 'rolling_predictions': The rolling boolean predictions series

        """
        if rolling_window is None:
            rolling_window = self.DEFAULT_ROLLING_WINDOW

        rolling_bool_predictions = self.calculate_rolling_metrics(transformed_data, rolling_window=rolling_window)

        pred = rolling_bool_predictions[f"{metric}_rolling_bool"].fillna(0)
        pred.index = rolling_bool_predictions["index"]

        # Calculate probability distribution over time horizons
        probability_distribution = self._calculate_probability_distribution(pred)

        # Determine if fault is detected based on threshold
        max_probability = max(
            probability_distribution.get("less_than_15_days", 0),
            probability_distribution.get("less_than_30_days", 0),
        )
        fault_detected = max_probability > threshold

        return {
            "fault_detected": fault_detected,
            "probability_distribution": probability_distribution,
            "rolling_predictions": rolling_bool_predictions[f"{metric}_rolling_bool"],
        }

    def predict_with_horizon(
        self,
        transformed_data: Dict[str, Any],
        metric: str = "Q",
        rolling_window: Optional[int] = None,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Predict fault probability distribution over time horizons.

        This is the legacy prediction method that returns probability distribution directly.

        Args:
            transformed_data (Dict[str, Any]): Transformed data metrics.
            metric (str, optional): Metric to use for prediction. Defaults to "Q".
            rolling_window (int, optional): Size of the rolling window.
                Defaults to DEFAULT_ROLLING_WINDOW.
            reference_time (datetime, optional): Reference time for horizon calculation.
                Defaults to datetime.now(). Useful for testing with deterministic results.

        Returns:
            Dict[str, float]: Probability distribution over time horizons:
                - 'less_than_15_days': Probability of fault within 15 days
                - 'less_than_30_days': Probability of fault between 15-30 days
                - 'less_than_180_days': Probability of fault between 30-180 days
                - 'more_than_180_days': Probability of fault after 180 days

        """
        if rolling_window is None:
            rolling_window = self.DEFAULT_ROLLING_WINDOW
        if reference_time is None:
            reference_time = datetime.now()

        rolling_bool_predictions = self.calculate_rolling_metrics(transformed_data, rolling_window=rolling_window)

        pred = rolling_bool_predictions[f"{metric}_rolling_bool"].fillna(0)
        pred.index = rolling_bool_predictions["index"]

        return self._calculate_probability_distribution(pred, reference_time=reference_time)

    def _calculate_probability_distribution(
        self, pred: pd.Series, reference_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Calculate probability distribution over time horizons.

        Args:
            pred (pd.Series): Rolling boolean predictions with datetime index.
            reference_time (datetime, optional): Reference time for horizon calculation.
                Defaults to datetime.now().

        Returns:
            Dict[str, float]: Normalized probability distribution over time horizons.

        """
        if reference_time is None:
            reference_time = datetime.now()

        # Filter predictions after reference time
        pred_future = pred[pred.index > reference_time]
        rolling_avg = pred_future.rolling(window=self.PREDICTION_ROLLING_WINDOW).mean().fillna(0)

        horizons = self.PREDICTION_HORIZONS
        proba_sum = {
            "less_than_15_days": rolling_avg[
                rolling_avg.index < reference_time + timedelta(days=horizons["less_than_15_days"])
            ].sum(),
            "less_than_30_days": rolling_avg[
                rolling_avg.index < reference_time + timedelta(days=horizons["less_than_30_days"])
            ].sum(),
            "less_than_180_days": rolling_avg[
                rolling_avg.index < reference_time + timedelta(days=horizons["less_than_180_days"])
            ].sum(),
            "more_than_180_days": rolling_avg[
                rolling_avg.index > reference_time + timedelta(days=horizons["less_than_180_days"])
            ].sum(),
        }

        # Convert cumulative to incremental probabilities
        proba_sum["less_than_30_days"] = proba_sum["less_than_30_days"] - proba_sum["less_than_15_days"]
        proba_sum["less_than_180_days"] = (
            proba_sum["less_than_180_days"] - proba_sum["less_than_30_days"] - proba_sum["less_than_15_days"]
        )

        # Handle case where no predictions are available
        if sum(proba_sum.values()) == 0:
            proba_sum["more_than_180_days"] = 1.0

        # Normalize to get probabilities
        total = sum(proba_sum.values())
        if total > 0:
            proba_sum = {k: float(v / total) for k, v in proba_sum.items()}

        return proba_sum
