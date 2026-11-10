"""Base class for fault detection techniques."""

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from plotly.subplots import make_subplots


class BaseDetector(ABC):
    """Base class for fault detection techniques."""

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit the detector to the data."""
        pass

    def __check_limits(self, x, metric):
        return (x > self.limits[metric]["high"]) or (x < self.limits[metric]["low"])

    def __calculate_pcw_bool(self, transformed_data: dict[str, Any]) -> dict[str, Any]:
        pcws = {f"PCW_{i+1}": [arr[i] for arr in transformed_data["PCW"]] for i in range(self.num_components)}
        pcws = pd.DataFrame(pcws)

        def make_pcw_checker(component_idx: int):
            def check_pcw(x: float) -> bool:
                pcw_key = f"PCW_{component_idx+1}"
                return x > self.limits["PCW"][pcw_key]["high"] or x < self.limits["PCW"][pcw_key]["low"]

            return check_pcw

        transformed_data["PCW_bool"] = {
            f"PCW_{i+1}_bool": pd.Series(pcws[f"PCW_{i+1}"]).apply(make_pcw_checker(i))
            for i in range(self.num_components)
        }

        return pd.DataFrame(transformed_data["PCW_bool"]).sum(axis=1) > (self.num_components / 2)

    def calculate_rolling_metrics(self, transformed_data: dict[str, Any], rolling_window: int = 6) -> dict[str, Any]:
        """Calculate the rolling metrics for the transformed data.

        Args:
            transformed_data (Dict[str, Any]): Transformed data metrics.
            rolling_window (int, optional): Size of the rolling window. Defaults to 6.

        Returns:
            Dict[str, Any]: Transformed data with rolling metrics.

        """
        metrics = ["T2", "T2_current", "Q", "schur"]
        for metric in metrics:
            transformed_data[f"{metric}_bool"] = pd.Series(transformed_data[metric]).apply(
                lambda x, m=metric: self.__check_limits(x, m)
            )

        transformed_data["PCW_bool"] = self.__calculate_pcw_bool(transformed_data)

        metrics = ["T2", "T2_current", "Q", "schur", "PCW"]
        for metric in metrics:
            transformed_data[f"{metric}_rolling_bool"] = (
                transformed_data[f"{metric}_bool"].rolling(window=rolling_window).apply(lambda x: all(x), raw=True)
            )

        return transformed_data

    def plot_rolling_bool_metrics(self, transformed_data: dict[str, Any]) -> None:
        """Plot the rolling metrics.

        Args:
            transformed_data (Dict[str, Any]): Transformed data with rolling metrics.

        """
        fig = px.line(title="Rolling Metrics")
        fig.add_scatter(
            x=transformed_data["index"], y=transformed_data["T2_rolling_bool"], mode="lines", name="T2 Rolling"
        )
        fig.add_scatter(
            x=transformed_data["index"],
            y=transformed_data["T2_current_rolling_bool"],
            mode="lines",
            name="T2 Current Rolling",
        )
        fig.add_scatter(
            x=transformed_data["index"], y=transformed_data["Q_rolling_bool"], mode="lines", name="Q Rolling"
        )
        fig.add_scatter(
            x=transformed_data["index"], y=transformed_data["schur_rolling_bool"], mode="lines", name="Schur Rolling"
        )
        fig.update_layout(title="Rolling Metrics", xaxis_title="Index", yaxis_title="Value", legend_title="Legend")
        fig.show()

    def predict(
        self,
        transformed_data: dict[str, Any],
        metric: str = "Q",
        threshold: float = 0.5,
        rolling_window: int = 6,
    ) -> dict[str, Any]:
        """Predict the fault probability.

        Args:
            transformed_data (Dict[str, Any]): Transformed data metrics.
            metric (str, optional): Metric to use for prediction. Defaults to "Q".
            threshold (float, optional): Threshold for the sigmoid probability. Defaults to 0.5.
            rolling_window (int, optional): Size of the rolling window. Defaults to 6.

        Returns:
            Dict[str, Any]: Transformed data with fault probability.

        """
        transformed_data = self.calculate_rolling_metrics(transformed_data, rolling_window=rolling_window)
        return transformed_data[f"{metric}_rolling_bool"] > threshold

    def __plot_pcw_series(self, transformed_data: dict[str, Any]) -> None:
        """Plot the PCW series.

        Args:
            transformed_data (dict[str, Any]): Transformed data.

        """
        PCW_series = pd.DataFrame(transformed_data["PCW"], columns=[f"PCW_{i+1}" for i in range(self.num_components)])
        fig_PCW = px.line(title="PCW Series")
        colors = [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "cyan",
            "magenta",
            "yellow",
            "brown",
            "pink",
            "lime",
            "teal",
            "indigo",
            "maroon",
            "navy",
        ][: self.num_components]
        for column, color in zip(PCW_series.columns, colors):
            fig_PCW.add_scatter(
                x=transformed_data["index"],
                y=PCW_series[column],
                mode="lines",
                name=column,
                line_color=color,
            )
            fig_PCW.add_hline(
                y=self.limits["PCW"][column]["high"],
                line_dash="dash",
                line_color=color,
                annotation_text=f"High {column} Limit",
                annotation_position="top left",
            )
            fig_PCW.add_hline(
                y=self.limits["PCW"][column]["low"],
                line_dash="dash",
                line_color=color,
                annotation_text=f"Low {column} Limit",
                annotation_position="bottom left",
            )
        fig_PCW.show()

    def plot_health_components(self) -> None:
        """Plot the health components using the calculated limits."""
        if self.limits is None:
            raise ValueError("Limits not fitted. Call fit() method first.")

        # Convert PCW_list to a DataFrame
        PCW_series = pd.DataFrame(self.limits["PCW_list"], columns=[f"PCW{i+1}" for i in range(self.num_components)])

        # Plotting T², T² current, Q and PCW series
        fig_T2 = px.line(title="T² Series")
        fig_T2.add_scatter(y=self.limits["T2_list"], mode="lines", name="T²", line_color="blue")
        fig_T2.show()

        fig_T2_current = px.line(title="T² Current Series")
        fig_T2_current.add_scatter(
            y=self.limits["T2_current_list"], mode="lines", name="T² Current", line_color="green"
        )

        fig_Q = px.line(title="Q Series")
        fig_Q.add_scatter(y=self.limits["Q_list"], mode="lines", name="Q", line_color="orange")
        fig_Q.show()

        # Plotting all PCW components in the same graph
        fig_PCW = px.line(title="PCW Series")
        for column in PCW_series.columns:
            fig_PCW.add_scatter(y=PCW_series[column], mode="lines", name=column)
        fig_PCW.show()

        # Plotting schur metric series
        fig_schur = px.line(title="Schur Complement Metric Series")
        fig_schur.add_scatter(y=self.limits["schur_list"], mode="lines", name="schur Metric", line_color="blue")
        fig_schur.show()

    def plot_transformed_data(self, transformed_data: dict[str, Any]) -> None:
        """Plot the transformed data with critical limits.

        Args:
            transformed_data (Dict[str, Any]): Transformed data metrics.

        """

        def plot_metric(
            metric_key: str,
            title: str,
            scatter_name: str,
            scatter_color: str,
            high_text: str,
            high_pos: str,
            low_text: str,
            low_pos: str,
        ) -> None:
            fig = px.line(title=title)
            fig.add_scatter(
                x=transformed_data["index"],
                y=transformed_data[metric_key],
                mode="lines",
                name=scatter_name,
                line_color=scatter_color,
            )
            fig.add_hline(
                y=self.limits[metric_key]["high"],
                line_dash="dash",
                line_color="red",
                annotation_text=high_text,
                annotation_position=high_pos,
            )
            fig.add_hline(
                y=self.limits[metric_key]["low"],
                line_dash="dash",
                line_color="green",
                annotation_text=low_text,
                annotation_position=low_pos,
            )
            fig.show()

        # Plot T² Metrics
        plot_metric(
            metric_key="T2",
            title="T² Metrics with Critical Limits",
            scatter_name="T²",
            scatter_color="yellow",
            high_text="High T² Limit",
            high_pos="top left",
            low_text="Low T² Limit",
            low_pos="bottom left",
        )

        # Plot T² Current Metrics
        plot_metric(
            metric_key="T2_current",
            title="T² Current Metrics with Critical Limits",
            scatter_name="T² Current",
            scatter_color="purple",
            high_text="High T² Current Limit",
            high_pos="top left",
            low_text="Low T² Current Limit",
            low_pos="bottom left",
        )

        # Plot Q Metrics
        plot_metric(
            metric_key="Q",
            title="Q Metrics with Critical Limits",
            scatter_name="Q",
            scatter_color="yellow",
            high_text="High Q Limit",
            high_pos="top right",
            low_text="Low Q Limit",
            low_pos="bottom right",
        )

        self.__plot_pcw_series(transformed_data)

        plot_metric(
            metric_key="schur",
            title="Schur Complement Metric with Critical Limits",
            scatter_name="Schur Metric",
            scatter_color="purple",
            high_text="High Schur Limit",
            high_pos="top left",
            low_text="Low Schur Limit",
            low_pos="bottom left",
        )

    def create_explainer_plot(
        self,
        model_type: str,
        data: pd.DataFrame,
        plot_type: str,
        background_size: int = 50,
        index: int = 0,
        fast: bool = True,
        max_summary_samples: int = 100,
        nsamples: int = 100,
        show: bool = False,
    ):
        """Generate SHAP plots for different model types (autoencoder, regressor, classifier, isolation_forest).

        Args:
            model_type (str): Type of model to explain ('autoencoder', 'regressor_error', 'classification',
            'isolation_forest').
            data (pd.DataFrame): Input features to explain.
            plot_type (str): Type of plot ('summary', 'summary_plotly', 'waterfall', 'waterfall_plotly', 'all').
            background_size (int): Number of samples to use for background when needed. Default is 50.
            index (int): Index of the sample to explain with waterfall. Default is 0.
            fast (bool): Whether to limit data size for faster computation. Default is True.
            max_summary_samples (int): Max number of samples for summary SHAP. Used when fast=True.
            nsamples (int): Number of samples for SHAP kernel estimation. Used when fast=True.
            show (bool): Whether to display the plot (only for Plotly plots). Default is False.

        Returns:
            Figure or None: A Plotly Figure if using 'summary_plotly' or 'waterfall_plotly'; None otherwise.

        """
        if index >= len(data):
            raise ValueError(f"Index {index} >= len(X) = {len(data)}")

        if plot_type not in ["summary", "waterfall", "summary_plotly", "waterfall_plotly", "all"]:
            raise ValueError(
                f"Invalid plot type: {plot_type}. Must be one of "
                "['summary', 'waterfall', 'summary_plotly', 'waterfall_plotly', 'all']"
            )

        background = data if len(data) <= background_size else data.sample(background_size, random_state=42)

        model_explainers = {
            "autoencoder": lambda: self._explain_kernel_model(
                data,
                plot_type,
                background,
                index,
                prediction_fn=self._predict_error,
                max_summary_samples=max_summary_samples if fast else len(data),
                nsamples=nsamples if fast else "auto",
                show=show,
            ),
            "regressor_error": lambda: self._explain_generic_model(data, plot_type, background, index, show=show),
            "classification": lambda: self._explain_generic_model(data, plot_type, background, index, show=show),
            "isolation_forest": lambda: self._explain_kernel_model(
                data,
                plot_type,
                background,
                index,
                prediction_fn=self.model.decision_function,
                max_summary_samples=max_summary_samples if fast else len(data),
                nsamples=nsamples if fast else "auto",
                show=show,
            ),
        }

        explainer_fn = model_explainers.get(model_type)
        if explainer_fn is None:
            raise ValueError(f"Invalid model type: {model_type}")

        return explainer_fn()

    def _explain_kernel_model(
        self,
        data: pd.DataFrame,
        plot_type: str,
        background: pd.DataFrame,
        index: int,
        prediction_fn: callable,
        max_summary_samples: int,
        nsamples: int,
        show: bool = False,
    ):
        """Explain a model using SHAP KernelExplainer (for autoencoders or kernel-compatible models).

        Args:
            data (pd.DataFrame): Input dataset.
            plot_type (str): Type of plot ('summary', 'waterfall', 'summary_plotly', 'waterfall_plotly', 'all').
            background (pd.DataFrame): Background dataset for SHAP.
            index (int): Index of instance to explain (for waterfall).
            prediction_fn (Callable): Prediction function for the model.
            max_summary_samples (int): Number of samples for SHAP summary.
            nsamples (Union[int, str]): Number of samples for KernelExplainer.
            show (bool): Whether to display the plot.

        Returns:
            plotly.graph_objs.Figure or None: Plotly figure if plotly mode, else None.

        """

        def predict_shap(X):
            return prediction_fn(pd.DataFrame(X, columns=data.columns))

        explainer = shap.KernelExplainer(predict_shap, background)
        summary_data = data if len(data) <= max_summary_samples else data.sample(max_summary_samples, random_state=42)
        shap_values_all, shap_values_one, fig, echarts_data = None, None, None, None

        if plot_type in ["summary", "summary_plotly", "all"]:
            shap_values_all = explainer.shap_values(summary_data, nsamples=nsamples)
        if plot_type in ["waterfall", "waterfall_plotly", "all"]:
            shap_values_one = explainer.shap_values(data.iloc[[index]], nsamples=nsamples)

        if plot_type == "summary":
            shap.summary_plot(shap_values_all, summary_data, feature_names=summary_data.columns, show=True)
        elif plot_type == "waterfall":
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_one[0],
                    base_values=explainer.expected_value,
                    data=data.iloc[index],
                    feature_names=data.columns,
                )
            )
        elif plot_type == "summary_plotly":
            fig = self._plot_shap_summary_plotly(shap_values_all, data, show=show)

            aggregated = {feature: [] for feature in data.columns}
            for i in range(shap_values_all.shape[0]):
                for j, feature in enumerate(data.columns):
                    aggregated[feature].append(
                        {"shap_value": float(shap_values_all[i, j]), "value": float(summary_data.iloc[i, j])}
                    )

            mean_abs_shap = {
                feature: float(np.mean([abs(p["shap_value"]) for p in points]))
                for feature, points in aggregated.items()
            }
            sorted_features = sorted(mean_abs_shap, key=mean_abs_shap.get, reverse=True)

            echarts_data = {
                "type": "scatter",
                "features": [{"feature": feature, "points": aggregated[feature]} for feature in sorted_features],
            }

        elif plot_type == "waterfall_plotly":
            fig = self._plot_shap_waterfall_plotly(
                shap_values_one[0],
                data.iloc[index],
                explainer.expected_value,
                title=f"SHAP Waterfall Plot - Sample: {index}",
                show=show,
            )

            abs_order = np.argsort(np.abs(shap_values_one[0]))[::-1]
            echarts_data = {
                "type": "waterfall",
                "base_value": float(explainer.expected_value),
                "final_prediction": float(explainer.expected_value + shap_values_one[0].sum()),
                "features": [
                    {
                        "name": data.columns[i],
                        "value": float(data.iloc[index, i]),
                        "shap_value": float(shap_values_one[0][i]),
                    }
                    for i in abs_order
                ],
            }
        elif plot_type == "all":
            shap.summary_plot(shap_values_all, summary_data, feature_names=summary_data.columns, show=True)
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_one[0],
                    base_values=explainer.expected_value,
                    data=data.iloc[index],
                    feature_names=data.columns,
                )
            )
            self._plot_shap_summary_plotly(shap_values_all, data, show=show)
            self._plot_shap_waterfall_plotly(
                shap_values_one[0],
                data.iloc[index],
                explainer.expected_value,
                title=f"SHAP Waterfall Plot - Sample: {index}",
                show=show,
            )
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

        return {"fig": fig, "echarts_data": echarts_data}

    def _explain_generic_model(
        self, data: pd.DataFrame, plot_type: str, background: int, index: int, show: bool = False
    ):
        """Explain a generic model (tree, linear or kernel) depending on compatibility.

        Args:
            data (pd.DataFrame): Input features to explain.
            plot_type (str): Type of plot ('summary', 'summary_plotly', 'waterfall', 'waterfall_plotly', 'all').
            background (int): Number of samples to use for background when needed.
            index (int): Index of the sample to explain with waterfall.
            show (bool): Whether to display the plot (only for Plotly plots).

        Returns:
            Figure or None: The Plotly Figure if using 'summary_plotly' or 'waterfall_plotly'; None otherwise.

        """
        try:
            explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                masker = shap.maskers.Independent(data)
                explainer = shap.LinearExplainer(self.model, masker=masker, feature_perturbation="interventional")
            except Exception:

                def predict_shap(X):
                    return self.model.predict(X)

                explainer = shap.KernelExplainer(predict_shap, background)

        shap_values = explainer(data)
        shap_array = shap_values.values if hasattr(shap_values, "values") else shap_values
        fig = echarts_data = None

        if plot_type == "summary":
            shap.summary_plot(shap_values)
        elif plot_type == "waterfall":
            shap.waterfall_plot(shap_values[index])
        elif plot_type == "summary_plotly":
            fig = self._plot_shap_summary_plotly(shap_array, data, show=show)

            aggregated = {feature: [] for feature in data.columns}
            for i in range(shap_array.shape[0]):
                for j, feature in enumerate(data.columns):
                    aggregated[feature].append({"shap_value": float(shap_array[i, j]), "value": float(data.iloc[i, j])})

            mean_abs_shap = {
                feature: float(np.mean([abs(p["shap_value"]) for p in points]))
                for feature, points in aggregated.items()
            }
            sorted_features = sorted(mean_abs_shap, key=mean_abs_shap.get, reverse=True)

            echarts_data = {
                "type": "scatter",
                "features": [{"feature": feature, "points": aggregated[feature]} for feature in sorted_features],
            }

        elif plot_type == "waterfall_plotly":
            fig = self._plot_shap_waterfall_plotly(
                shap_array[index],
                data.iloc[index],
                explainer.expected_value,
                title=f"SHAP Waterfall Plot - Sample: {index}",
                show=show,
            )

            abs_order = np.argsort(np.abs(shap_array[index]))[::-1]
            echarts_data = {
                "type": "waterfall",
                "base_value": float(explainer.expected_value),
                "final_prediction": float(explainer.expected_value + shap_array[index].sum()),
                "features": [
                    {
                        "name": data.columns[i],
                        "value": float(data.iloc[index, i]),
                        "shap_value": float(shap_array[index][i]),
                    }
                    for i in abs_order
                ],
            }

        elif plot_type == "all":
            shap.summary_plot(shap_values)
            shap.waterfall_plot(shap_values[index])
            self._plot_shap_summary_plotly(shap_array, data, show=show)
            self._plot_shap_waterfall_plotly(
                shap_array[index],
                data.iloc[index],
                explainer.expected_value,
                title=f"SHAP Waterfall Plot - Sample: {index}",
                show=show,
            )
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

        return {"fig": fig, "echarts_data": echarts_data}

    def _plot_shap_summary_plotly(
        self, shap_values: np.ndarray, X: pd.DataFrame, title: str = "SHAP Summary Plot", show: bool = False
    ):
        """Generate a Plotly-based SHAP summary plot.

        Args:
            shap_values (np.ndarray): SHAP values matrix.
            X (pd.DataFrame): The input feature data.
            title (str): Plot title.
            show (bool): Whether to display the figure.

        Returns:
            plotly.graph_objs.Figure: Plotly figure object.

        """
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)
        fig = make_subplots(rows=1, cols=1)
        for i, idx in enumerate(sorted_idx):
            feature_vals = X.iloc[:, idx].values
            svals = shap_values[:, idx]
            fig.add_trace(
                go.Scatter(
                    x=svals,
                    y=[i] * len(svals),
                    mode="markers",
                    marker=dict(color=feature_vals, colorscale="RdBu_r", showscale=bool(i == sorted_idx[-1]), size=7),
                    name=X.columns[idx],
                    showlegend=False,
                )
            )
        feature_names_sorted = [X.columns[idx] for idx in sorted_idx]
        fig.update_layout(
            title=title,
            xaxis=dict(title="SHAP Value"),
            yaxis=dict(
                tickmode="array", tickvals=list(range(len(sorted_idx))), ticktext=feature_names_sorted, title="Features"
            ),
            height=600,
            width=900,
        )
        if show:
            fig.show()
        return fig

    def _plot_shap_waterfall_plotly(
        self,
        shap_values: np.ndarray,
        instance: pd.Series,
        base_value: float,
        title: str = "SHAP Waterfall Plot",
        max_features: int = 10,
        show: bool = False,
    ):
        """Generate a Plotly-based SHAP waterfall plot.

        Args:
            shap_values (np.ndarray): 1D SHAP values for a single instance.
            instance (pd.Series): Original feature values of the instance.
            base_value (float): SHAP base value.
            title (str): Plot title.
            max_features (int): Number of top features to display.
            show (bool): Whether to display the figure.

        Returns:
            plotly.graph_objs.Figure: Plotly figure object.

        """
        shap_values = np.array(shap_values).flatten()
        assert shap_values.ndim == 1, "shap_values precisa ser 1D para waterfall"
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:max_features]
        feature_names = instance.index[sorted_idx]
        shap_sorted = shap_values[sorted_idx]
        feature_vals_sorted = instance.values[sorted_idx]
        x_labels = (
            ["Base Value"]
            + [f"{name} = {val:.3g}" for name, val in zip(feature_names, feature_vals_sorted)]
            + ["Prediction"]
        )
        y_values = [float(base_value)] + list(shap_sorted.astype(float)) + [float(base_value + np.sum(shap_sorted))]
        measure = ["absolute"] + ["relative"] * len(shap_sorted) + ["total"]
        fig = go.Figure(
            go.Waterfall(
                name="SHAP Waterfall",
                orientation="v",
                measure=measure,
                x=x_labels,
                y=y_values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "rgba(31, 119, 180, 0.8)"}},
                increasing={"marker": {"color": "rgba(214, 39, 40, 0.8)"}},
                totals={"marker": {"color": "rgba(0, 204, 150, 0.8)"}},
            )
        )
        final_prediction = base_value + np.sum(shap_sorted)
        fig.update_layout(
            title=(f"{title}<br>Base Value = {base_value:.4f}, Prediction/Error = {final_prediction:.4f}"),
            height=max(500, 40 * len(x_labels)),
            width=900,
        )
        if show:
            fig.show()
        return fig

    def check_duration(self, flagged_data: pd.DataFrame, duration: timedelta, rule_name: str) -> list[dict[str, Any]]:
        """Check if flagged anomalies persist for the required duration.

        Args:
            flagged_data (pd.DataFrame): Filtered DataFrame with flagged anomalies.
            duration (timedelta): Minimum duration for a valid anomaly.
            rule_name (str): Name of the rule being evaluated.

        Returns:
            List[Dict[str, Any]]: List of detected anomalies with start and end times.

        """
        anomalies = []
        if not flagged_data.empty:
            grouped = flagged_data.groupby((flagged_data["Timestamp"].diff() > duration).cumsum())
            for _, group in grouped:
                if group["Timestamp"].max() - group["Timestamp"].min() >= duration:
                    anomalies.append(
                        {
                            "rule": rule_name,
                            "start_time": group["Timestamp"].min(),
                            "end_time": group["Timestamp"].max(),
                        }
                    )

        return anomalies
