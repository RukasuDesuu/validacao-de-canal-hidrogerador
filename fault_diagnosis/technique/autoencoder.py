"""AutoEncoder implementation class."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fault_diagnosis.technique.base_detector import BaseDetector


class AutoEncoder(nn.Module):
    """AutoEncoder with MLP architecture (fully connected network)."""

    VALID_ACTIVATIONS = ("relu", "linear")

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layers: Tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
    ):
        """Initialize the AutoEncoder with the specified architecture.

        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space representation.
            hidden_layers (Tuple[int, ...], optional): MLP architecture. Defaults to (128, 64, 32).
            activation (str, optional): Activation function between layers.
                "relu": nn.ReLU (default, nonlinear).
                "linear": nn.Identity (no activation, PCA-equivalent).

        """
        super().__init__()

        activation_fn = nn.ReLU() if activation == "relu" else nn.Identity()

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(activation_fn)
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(activation_fn)
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_recon (torch.Tensor): Reconstructed input tensor of shape (batch_size, input_dim).
                - z (torch.Tensor): Latent representation tensor of shape (batch_size, latent_dim).

        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class CNNAutoEncoder(nn.Module):
    """1D CNN-based AutoEncoder.

    This class implements a 1D Convolutional Neural Network (CNN) based AutoEncoder. It assumes that the input
    tensor has the shape (batch_size, 1, input_dim) and produces an output tensor of the same shape.

    The architecture consists of an encoder and a decoder:
    - The encoder compresses the input data into a latent space representation.
    - The decoder reconstructs the input data from the latent space representation.

    """

    def __init__(self, input_dim: int, latent_dim: int = 3):
        """Initialize the CNNAutoEncoder with the specified architecture.

        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int, optional): Dimension of the latent space representation. Defaults to 3.

        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc_enc = nn.Linear(32 * input_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 32 * input_dim)
        self.tconv2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.tconv1 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_recon (torch.Tensor): Reconstructed input tensor of shape (batch_size, 1, input_dim).
                - z (torch.Tensor): Latent representation tensor of shape (batch_size, latent_dim).

        """
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # flatten -> (batch_size, 32*input_dim)
        z = self.fc_enc(x)

        # Decoder
        x_dec = self.fc_dec(z)  # (batch_size, 32*input_dim)
        x_dec = x_dec.view(batch_size, 32, self.input_dim)  # unflatten -> (batch_size, 32, input_dim)
        x_dec = self.relu(self.tconv2(x_dec))
        x_recon = self.tconv1(x_dec)  # (batch_size, 1, input_dim)

        return x_recon, z


class NonlinearAE:
    """1-hidden-layer tanh autoencoder (numpy SGD, matching MATLAB trainAE_nonlinear).

    Architecture: Xn -> tanh(Xn·W1 + b1) -> H·W2 + b2
    Training uses mini-batch SGD with L2 weight decay and Xavier/Glorot init.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        epochs: int = 30,
        lr: float = 1e-2,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        random_seed: int = 1,
    ):
        """Initialize the NonlinearAE.

        Args:
            hidden_size: Number of neurons in the hidden layer.
            epochs: Number of training epochs.
            lr: Learning rate for SGD.
            batch_size: Mini-batch size.
            weight_decay: L2 regularization coefficient.
            random_seed: Random seed for reproducibility.

        """
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_seed = random_seed

        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None

    def fit(self, D_train: np.ndarray) -> "NonlinearAE":
        """Train: z-score normalize, Xavier init, mini-batch SGD with L2.

        Args:
            D_train: Training data, shape (N, d). Typically stacked double-centered blocks.

        Returns:
            Self after training.

        """
        rng = np.random.RandomState(self.random_seed)

        # Per-column z-score normalization (ddof=0 matching MATLAB)
        self.mu = D_train.mean(axis=0)
        self.sigma = D_train.std(axis=0, ddof=0)
        self.sigma[self.sigma == 0] = 1.0  # avoid division by zero
        Xn = (D_train - self.mu) / self.sigma

        n_samples, input_dim = Xn.shape
        h = self.hidden_size

        # Xavier/Glorot initialization
        limit1 = np.sqrt(6.0 / (input_dim + h))
        self.W1 = rng.uniform(-limit1, limit1, (input_dim, h))
        self.b1 = np.zeros(h)

        limit2 = np.sqrt(6.0 / (h + input_dim))
        self.W2 = rng.uniform(-limit2, limit2, (h, input_dim))
        self.b2 = np.zeros(input_dim)

        # Mini-batch SGD
        for _epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                Xb = Xn[batch_idx]
                mb = len(Xb)

                # Forward
                Z1 = Xb @ self.W1 + self.b1
                H = np.tanh(Z1)
                Y = H @ self.W2 + self.b2

                # Backward (MSE loss) — matches MATLAB trainAE_nonlinear exactly:
                # dY already has 1/m factor; MATLAB does NOT divide again in dW/db.
                dY = (2.0 / mb) * (Y - Xb)

                dW2 = H.T @ dY + self.weight_decay * self.W2
                db2 = dY.sum(axis=0)

                dH = dY @ self.W2.T
                dZ1 = dH * (1.0 - H**2)  # tanh derivative

                dW1 = Xb.T @ dZ1 + self.weight_decay * self.W1
                db1 = dZ1.sum(axis=0)

                # SGD update
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

        return self

    def reconstruct(self, D: np.ndarray) -> np.ndarray:
        """Forward pass: normalize -> tanh -> linear -> denormalize.

        Args:
            D: Input data, shape (N, d).

        Returns:
            Reconstructed data in original scale, shape (N, d).

        """
        Xn = (D - self.mu) / self.sigma
        H = np.tanh(Xn @ self.W1 + self.b1)
        Y = H @ self.W2 + self.b2
        return Y * self.sigma + self.mu

    def compute_q(self, D: np.ndarray) -> float:
        """Compute Q_NL = mean(mean((D - reconstruct(D))^2, axis=1)).

        Args:
            D: Input data, shape (N, d).

        Returns:
            Scalar MSE reconstruction error.

        """
        D_hat = self.reconstruct(D)
        return float(np.mean(np.mean((D - D_hat) ** 2, axis=1)))


class MWAutoEncoder(TransformerMixin, BaseDetector):
    """Moving Window AutoEncoder (MWAE) that can use MLP or CNN architecture.

    This class implements a moving window-based autoencoder for anomaly detection.
    It supports both MLP (fully connected) and CNN (convolutional) architectures.
    """

    # Default configuration constants
    DEFAULT_HIDDEN_LAYERS: Tuple[int, ...] = (128, 64, 32)
    DEFAULT_LATENT_DIM: int = 3
    DEFAULT_NUM_EPOCHS: int = 50
    DEFAULT_LEARNING_RATE: float = 1e-3
    DEFAULT_NUM_STDS: int = 3
    DEFAULT_WINDOW_SIZE: int = 10
    DEFAULT_BATCH_SIZE: int = 32

    VALID_WAE_METHODS = ("abs_ratio", "variance_ratio", "pcw")
    VALID_ACTIVATIONS = ("relu", "linear")

    def __init__(
        self,
        model_arch: str = "mlp",
        hidden_layers: Optional[Tuple[int, ...]] = None,
        use_scaler: bool = True,
        use_gating: bool = False,
        wae_method: str = "abs_ratio",
        center_per_window: bool = False,
        activation: str = "relu",
    ) -> None:
        """Initialize the Autoencoder class with the specified model architecture and hidden layers.

        Args:
            model_arch (str): "mlp" or "cnn". Defines which architecture to use. Defaults to "mlp".
            hidden_layers (Tuple[int, ...], optional): Architecture of the MLP if model_arch is "mlp".
                Defaults to DEFAULT_HIDDEN_LAYERS (128, 64, 32).
            use_scaler (bool): Whether to use StandardScaler for data normalization. Defaults to True.
            use_gating (bool): Whether to apply gating to Q. When True, Q is forced to 0 if all
                W_AE components are within their limits. Defaults to False.
            wae_method (str): Method for computing W_AE latent weight percentages.
                "abs_ratio": |z_last| / sum(|z_last|) * 100 (default, per-sample).
                "variance_ratio": var(z_k) / sum(var(z_k)) * 100 across all window samples,
                    producing PCA-like variance explained percentages with wider spread.
                "pcw": Analytical PCA on augmented [ref, Xw] matrix. Computes W_AE as
                    eigenvalue/sum*100 and Q as PCA reconstruction error. No neural network
                    is trained — matches the MATLAB PCA-AE pipeline exactly. Requires
                    ref_data kwarg in fit().
            center_per_window (bool): When True, subtracts column-wise mean from each window
                before feeding to the model. Matches Excel/RMWPCASC per-window centering and
                makes detection invariant to slow drifts. Defaults to False.
            activation (str): Activation function for MLP layers (only applies to model_arch="mlp").
                "relu": nn.ReLU (default, nonlinear).
                "linear": nn.Identity (no activation, PCA-equivalent).

        Attributes:
            model_arch (str): The chosen model architecture.
            hidden_layers (Tuple[int, ...]): The hidden layers configuration for the MLP.
            use_scaler (bool): Flag indicating if StandardScaler is used.
            use_gating (bool): Flag indicating if gating is applied to Q.
            wae_method (str): The W_AE computation method ("abs_ratio" or "variance_ratio").
            center_per_window (bool): Flag indicating if per-window centering is applied.
            activation (str): The activation function for MLP layers ("relu" or "linear").
            scaler (Optional[StandardScaler]): The fitted StandardScaler instance, initialized as None.
            model (Optional[nn.Module]): The neural network model, initialized as None.
            health_data (Optional[pd.DataFrame]): The health data, initialized as None.
            window_size (Optional[int]): The window size, initialized as None.
            latent_dim (Optional[int]): The latent dimension, initialized as None.
            limits (Optional[Dict[str, Any]]): The limits dictionary, initialized as None.
            device (torch.device): The device to run the model on, either "cuda" or "cpu".
            fitted (bool): Whether the model has been fitted.

        Raises:
            ValueError: If wae_method or activation is not one of the valid values.

        """
        if wae_method not in self.VALID_WAE_METHODS:
            raise ValueError(f"Invalid wae_method '{wae_method}'. Must be one of {self.VALID_WAE_METHODS}.")
        if activation not in self.VALID_ACTIVATIONS:
            raise ValueError(f"Invalid activation '{activation}'. Must be one of {self.VALID_ACTIVATIONS}.")

        self.model_arch = model_arch.lower()
        self.hidden_layers = hidden_layers if hidden_layers is not None else self.DEFAULT_HIDDEN_LAYERS
        self.use_scaler = use_scaler
        self.use_gating = use_gating
        self.wae_method = wae_method
        self.center_per_window = center_per_window
        self.activation = activation

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[nn.Module] = None
        self.health_data: Optional[pd.DataFrame] = None
        self.window_size: Optional[int] = None
        self.latent_dim: Optional[int] = None
        self.limits: Optional[Dict[str, Any]] = None
        self.fitted: bool = False

        # PCW mode attributes (analytical PCA on augmented [ref, Xw] matrix)
        self.ref_data: Optional[np.ndarray] = None
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_centered: Optional[np.ndarray] = None
        self.num_components: Optional[int] = None

        # Nonlinear AE for Q_NL (optional, trained when nl_hidden_size kwarg is passed to fit)
        self.nl_ae: Optional[NonlinearAE] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> "MWAutoEncoder":
        """Train the AutoEncoder (MLP or CNN) on the provided health data.

        Args:
            data (pd.DataFrame): Training data with features as columns.
            **kwargs: Additional parameters:
                - window_size (int): Size of the sliding window. Defaults to DEFAULT_WINDOW_SIZE.
                - latent_dim (int): Dimension of the latent space. Defaults to DEFAULT_LATENT_DIM.
                - num_epochs (int): Number of training epochs. Defaults to DEFAULT_NUM_EPOCHS.
                - learning_rate (float): Learning rate for optimizer. Defaults to DEFAULT_LEARNING_RATE.
                - batch_size (int): Batch size for training. Defaults to DEFAULT_BATCH_SIZE.
                    Use None to process entire dataset at once (legacy behavior).
                - random_seed (int, optional): Random seed for deterministic training. Defaults to None.
                - ref_data (np.ndarray or pd.DataFrame, optional): Reference window for PCW mode.
                    Used only when wae_method="pcw". The reference is centered and concatenated
                    with each current window before PCA. Defaults to None.

        Returns:
            MWAutoEncoder: The fitted autoencoder instance.

        Raises:
            ValueError: If data contains NaN or Inf values, or if model_arch is invalid.

        """
        # Extract kwargs with defaults
        window_size: int = kwargs.get("window_size", self.DEFAULT_WINDOW_SIZE)
        latent_dim: int = kwargs.get("latent_dim", self.DEFAULT_LATENT_DIM)
        num_epochs: int = kwargs.get("num_epochs", self.DEFAULT_NUM_EPOCHS)
        learning_rate: float = kwargs.get("learning_rate", self.DEFAULT_LEARNING_RATE)
        batch_size: Optional[int] = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        random_seed: Optional[int] = kwargs.get("random_seed", None)
        ref_data: Optional[np.ndarray] = kwargs.get("ref_data", None)

        # PCW mode: analytical PCA — no neural network needed
        if self.wae_method == "pcw":
            self._validate_input_data(data)
            self.limits = None
            self.fitted = False
            self.health_data = data
            self.window_size = window_size
            self.latent_dim = latent_dim
            self.num_components = latent_dim

            if ref_data is not None:
                ref_arr = ref_data if isinstance(ref_data, np.ndarray) else ref_data.values
                self.ref_data = ref_arr.astype(float)
                self.ref_mean = np.mean(self.ref_data, axis=0)
                self.ref_centered = self.ref_data - self.ref_mean

            # Train nonlinear AE for Q_NL (optional, enabled by nl_hidden_size kwarg)
            nl_hidden = kwargs.get("nl_hidden_size", None)
            if ref_data is not None and nl_hidden is not None:
                nl_baseline_start = kwargs.get("nl_baseline_start", None)
                D_train = self._build_nl_training_data(data, window_size, nl_baseline_start)
                self.nl_ae = NonlinearAE(
                    hidden_size=nl_hidden,
                    epochs=kwargs.get("nl_epochs", 30),
                    lr=kwargs.get("nl_lr", 1e-2),
                    batch_size=kwargs.get("nl_batch_size", 256),
                    weight_decay=kwargs.get("nl_weight_decay", 1e-4),
                    random_seed=kwargs.get("nl_random_seed", 1),
                )
                self.nl_ae.fit(D_train)

            self.limits = self._calculate_limits(self.health_data, window_size)
            self.fitted = True
            return self

        # Set random seeds for deterministic training
        if random_seed is not None:
            import random

            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Validate input data
        self._validate_input_data(data)

        # Reset limits so gating is disabled during _calculate_limits
        self.limits = None
        self.fitted = False

        # Store original health_data for limits calculation
        self.health_data = data
        self.window_size = window_size
        self.latent_dim = latent_dim

        # Apply StandardScaler if enabled
        if self.use_scaler:
            self.scaler = StandardScaler()
            health_data_scaled = pd.DataFrame(
                self.scaler.fit_transform(data),
                index=data.index,
                columns=data.columns,
            )
        else:
            health_data_scaled = data

        # Center training data by global column mean (approximates per-window centering)
        if self.center_per_window:
            training_data = health_data_scaled - health_data_scaled.mean()
        else:
            training_data = health_data_scaled

        input_dim = training_data.shape[1]

        if self.model_arch == "mlp":
            self.model = AutoEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_layers=self.hidden_layers,
                activation=self.activation,
            ).to(self.device)
        elif self.model_arch == "cnn":
            self.model = CNNAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        else:
            raise ValueError(f"Invalid model_arch '{self.model_arch}'. Use 'mlp' or 'cnn'.")

        # Train the model
        self._train_model(training_data, num_epochs, learning_rate, batch_size)

        # Calculate limits using ORIGINAL data (transform will apply scaler internally)
        # This prevents double scaling: calculate_limits -> transform -> scaler.transform
        self.limits = self._calculate_limits(self.health_data, window_size)
        self.fitted = True
        return self

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data for NaN and Inf values.

        Args:
            data (pd.DataFrame): Data to validate.

        Raises:
            ValueError: If data contains NaN or Inf values.

        """
        if data.isnull().any().any():
            nan_counts = data.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            raise ValueError(
                f"Data contains NaN values. Columns with NaNs: {nan_cols.to_dict()}. "
                "Please clean the data before fitting."
            )
        if np.any(np.isinf(data.values)):
            raise ValueError("Data contains Inf values. Please clean the data before fitting.")

    def _train_model(
        self,
        data: pd.DataFrame,
        num_epochs: int,
        learning_rate: float,
        batch_size: Optional[int],
    ) -> None:
        """Train the autoencoder model.

        Args:
            data (pd.DataFrame): Scaled training data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimizer.
            batch_size (Optional[int]): Batch size. If None, uses entire dataset.

        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Prepare data tensor
        data_array = data.values.astype(np.float32)
        data_tensor = torch.tensor(data_array, dtype=torch.float32)

        if self.model_arch == "cnn":
            data_tensor = data_tensor.unsqueeze(1)  # (N, 1, input_dim)

        # Use DataLoader for batch processing if batch_size is specified
        if batch_size is not None and batch_size < len(data_tensor):
            dataset = TensorDataset(data_tensor, data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_x, batch_target in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_target = batch_target.to(self.device)

                    # Denoising AE: Inject Gaussian noise (σ = 0.01) to inputs
                    noise = torch.randn_like(batch_x) * 0.01
                    noisy_batch_x = batch_x + noise

                    optimizer.zero_grad()
                    x_recon, _ = self.model(noisy_batch_x)
                    loss = criterion(x_recon, batch_target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
        else:
            # Legacy behavior: process entire dataset at once
            data_tensor = data_tensor.to(self.device)

            self.model.train()
            for epoch in range(num_epochs):
                # Denoising AE: Inject Gaussian noise (σ = 0.01) to inputs
                noise = torch.randn_like(data_tensor) * 0.01
                noisy_data_tensor = data_tensor + noise

                optimizer.zero_grad()
                x_recon, _ = self.model(noisy_data_tensor)
                loss = criterion(x_recon, data_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    def _pca_transform(self, current_data_values: np.ndarray) -> Dict[str, Any]:
        """Perform analytical PCA on (optionally augmented) window data.

        Used by wae_method='pcw' to compute W_AE (eigenvalue percentages) and Q
        (reconstruction error) matching the MATLAB PCA-AE pipeline.

        Args:
            current_data_values (np.ndarray): Window data, shape (W, d).

        Returns:
            Dict[str, Any]: Dictionary with 'W_AE' (eigenvalue %) and 'Q' (reconstruction error).

        """
        # Center current data
        current_mean = np.mean(current_data_values, axis=0)
        current_centered = current_data_values - current_mean

        # Augment with reference if available: [ref_centered, current_centered]
        if self.ref_centered is not None:
            augmented = np.hstack([self.ref_centered, current_centered])
        else:
            augmented = current_centered

        # Covariance and eigendecomposition
        cov_matrix = np.cov(augmented, rowvar=False)
        cov_matrix += 1e-8 * np.eye(cov_matrix.shape[0])  # regularization

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort descending
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # W_AE: top-k eigenvalue percentages
        full_eigenvalues = eigenvalues.copy()
        top_k_eigenvalues = eigenvalues[: self.num_components]
        top_k_eigenvectors = eigenvectors[:, : self.num_components]
        total = np.sum(full_eigenvalues)
        W_AE = 100.0 * (top_k_eigenvalues / total) if total > 0 else np.zeros(self.num_components)

        # Q: PCA reconstruction error (same formula as RMWPCA_SC._calculate_q)
        transformed = augmented @ top_k_eigenvectors
        reconstructed = transformed @ top_k_eigenvectors.T
        residuals = augmented - reconstructed
        Q_per_sample = np.sum(residuals**2, axis=1)
        Q = float(np.mean(np.abs(Q_per_sample - Q_per_sample.mean())))

        result = {"W_AE": W_AE, "Q": Q}
        if self.nl_ae is not None:
            result["Q_NL"] = self._compute_q_nl(current_data_values)
        return result

    def _build_nl_training_data(
        self, health_data: pd.DataFrame, window_size: int, baseline_start: Optional[int] = None
    ) -> np.ndarray:
        """Build D_train for nonlinear AE (stacked double-centered augmented windows).

        Args:
            health_data: Health data used for training.
            window_size: Size of the sliding window.
            baseline_start: 0-indexed sample start for baseline windows. When provided,
                uses windows starting from baseline_start+1 through baseline_start+window_size
                (matching MATLAB's baseline_first_out_row). When None, uses all N-W windows.

        Returns:
            Stacked double-centered blocks, shape (n_windows * window_size, 2*d).

        """
        all_data = health_data.values.astype(float) if hasattr(health_data, "values") else health_data
        blocks = []

        if baseline_start is not None:
            # MATLAB: baseline_starts = (baseline_first_out_row+1):(baseline_first_out_row+W)
            # These are 1-indexed sample starts. In 0-indexed: baseline_start .. baseline_start+W-1
            start_range = range(baseline_start, baseline_start + window_size)
        else:
            start_range = range(len(all_data) - window_size)

        for i in start_range:
            end = i + window_size
            if end > len(all_data):
                break
            Xw = all_data[i:end]
            Z = np.hstack([self.ref_data, Xw])
            D = Z - Z.mean(axis=0)
            D = D - D.mean(axis=0)
            blocks.append(D)
        return np.vstack(blocks)

    def _compute_q_nl(self, current_data_values: np.ndarray) -> float:
        """Compute Q_NL for one window using trained nonlinear AE.

        Args:
            current_data_values: Window data, shape (W, d).

        Returns:
            Scalar Q_NL (MSE reconstruction error from nonlinear AE).

        """
        Z = np.hstack([self.ref_data, current_data_values])
        D = Z - Z.mean(axis=0)
        D = D - D.mean(axis=0)
        return self.nl_ae.compute_q(D)

    def transform(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply the trained model (MLP or CNN) to current_data, calculating Q for the last sample.

        Also extracts the latent vector z and computes W_AE (latent weight percentages)
        for the last sample in the window.

        Args:
            current_data (pd.DataFrame): Current data window to transform.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - Q: reconstruction error (SSE) of the last sample
                - index: timestamp of the last sample
                - z: latent vector of the last sample (numpy array)
                - W_AE: latent weight percentages. Computed as |z_i|/sum(|z_j|)*100
                    when wae_method="abs_ratio", or var(z_k)/sum(var(z_k))*100
                    when wae_method="variance_ratio", or eigenvalue/sum*100
                    when wae_method="pcw".

        Raises:
            ValueError: If model is not fitted or data contains invalid values.

        """
        # PCW mode: analytical PCA
        if self.wae_method == "pcw":
            self._validate_input_data(current_data)
            pca_result = self._pca_transform(current_data.values.astype(float))
            Q = pca_result["Q"]
            W_AE = pca_result["W_AE"]

            # Gating: force Q=0 when all W_AE components are within their limits
            if self.use_gating and self.limits is not None and "W_AE" in self.limits:
                any_w_ae_outside = False
                for k in range(len(W_AE)):
                    w_key = f"W_AE_{k + 1}"
                    if w_key in self.limits["W_AE"]:
                        w_limits = self.limits["W_AE"][w_key]
                        if W_AE[k] > w_limits["high"] or W_AE[k] < w_limits["low"]:
                            any_w_ae_outside = True
                            break
                if not any_w_ae_outside:
                    Q = 0.0

            result = {
                "Q": Q,
                "index": current_data.index[-1],
                "z": np.zeros(self.num_components),
                "W_AE": W_AE,
            }
            if "Q_NL" in pca_result:
                result["Q_NL"] = pca_result["Q_NL"]
            return result

        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Validate input data
        self._validate_input_data(current_data)

        # Apply scaler if it was used during training
        if self.scaler is not None:
            current_data_scaled = pd.DataFrame(
                self.scaler.transform(current_data),
                index=current_data.index,
                columns=current_data.columns,
            )
        else:
            current_data_scaled = current_data

        # Per-window centering: subtract column-wise mean of this window
        if self.center_per_window:
            current_data_scaled = current_data_scaled - current_data_scaled.mean()

        self.model.eval()

        data_array = current_data_scaled.to_numpy().astype(np.float32)
        data_tensor = torch.tensor(data_array, dtype=torch.float32)
        if self.model_arch == "cnn":
            data_tensor = data_tensor.unsqueeze(1)  # (batch_size, 1, input_dim)

        data_tensor = data_tensor.to(self.device)
        with torch.no_grad():
            x_recon, z = self.model(data_tensor)

        if self.model_arch == "mlp":
            # shape (batch, input_dim) — SSE per sample
            recon_error = torch.sum((data_tensor - x_recon) ** 2, dim=1)
        else:
            recon_error = torch.sum((data_tensor - x_recon) ** 2, dim=[1, 2])

        Q = recon_error[-1].item()

        # Extract latent vector of the last sample
        z_last = z[-1].cpu().numpy()

        # Compute W_AE weight percentages
        if self.wae_method == "variance_ratio":
            # Variance of each latent dimension across all window samples
            z_all = z.cpu().numpy()  # shape (window_size, latent_dim)
            z_var = np.var(z_all, axis=0)
            z_var_sum = z_var.sum()
            W_AE_raw = (z_var / z_var_sum * 100) if z_var_sum > 0 else np.zeros_like(z_var)
        else:
            # abs_ratio: |z_last| / sum(|z_last|) * 100
            z_abs = np.abs(z_last)
            z_sum = z_abs.sum()
            W_AE_raw = (z_abs / z_sum * 100) if z_sum > 0 else np.zeros_like(z_abs)

        # Sort W_AE descending so W_AE_1 is always the dominant component
        # (matches Excel convention where W_AE1 >= W_AE2 >= W_AE3)
        sort_idx = np.argsort(W_AE_raw)[::-1]
        W_AE = W_AE_raw[sort_idx].astype(np.float64)
        w_sum = W_AE.sum()
        if w_sum > 0:
            W_AE = W_AE * (100.0 / w_sum)
            W_AE[-1] = 100.0 - W_AE[:-1].sum()

        # Gating: force Q=0 when all W_AE components are within their limits
        if self.use_gating and self.limits is not None and "W_AE" in self.limits:
            any_w_ae_outside = False
            for k in range(len(W_AE)):
                w_key = f"W_AE_{k + 1}"
                if w_key in self.limits["W_AE"]:
                    w_limits = self.limits["W_AE"][w_key]
                    if W_AE[k] > w_limits["high"] or W_AE[k] < w_limits["low"]:
                        any_w_ae_outside = True
                        break
            if not any_w_ae_outside:
                Q = 0.0

        return {
            "Q": Q,
            "index": current_data.index[-1],
            "z": z_last,
            "W_AE": W_AE,
        }

    def _calculate_limits(
        self, health_data: pd.DataFrame, window_size: int, num_stds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run sliding windows on 'health_data', calling self.transform() on each window.

        Calculates Q (reconstruction error) and W_AE (latent weight) limits
        based on the mean ± num_stds*std.

        Args:
            health_data (pd.DataFrame): The health data to calculate limits for.
            window_size (int): The size of the sliding window.
            num_stds (int, optional): The number of standard deviations to use for defining limits.
                Defaults to DEFAULT_NUM_STDS.

        Returns:
            Dict[str, Any]: Dictionary with Q and W_AE limits and intermediate lists for plotting.

        """
        if num_stds is None:
            num_stds = self.DEFAULT_NUM_STDS

        metrics_list = []
        # PCW mode uses N-W windows (matches MATLAB/RMWPCA_SC), other modes use N-W+1
        n_windows = len(health_data) - window_size + (0 if self.wae_method == "pcw" else 1)
        for i in range(n_windows):
            data_window = health_data.iloc[i : i + window_size]
            metrics = self.transform(data_window)
            metrics_list.append(metrics)

        Q_list = [m["Q"] for m in metrics_list]
        W_AE_list = [m["W_AE"] for m in metrics_list]

        def calculate_metric_limits(metric_list: List[float]) -> Dict[str, float]:
            mean = float(np.mean(metric_list))
            std_dev = float(np.std(metric_list, ddof=1))
            return {"low": mean - num_stds * std_dev, "high": mean + num_stds * std_dev}

        # Build per-component W_AE limits
        latent_dim = len(W_AE_list[0])
        wae_limits = {}
        wae_component_lists = {}
        for k in range(latent_dim):
            component_values = [w[k] for w in W_AE_list]
            wae_limits[f"W_AE_{k + 1}"] = calculate_metric_limits(component_values)
            wae_component_lists[f"W_AE_{k + 1}"] = component_values

        limits = {
            "Q": calculate_metric_limits(Q_list),
            "Q_list": Q_list,  # Store intermediate list for plotting
            "W_AE": wae_limits,
            "W_AE_lists": wae_component_lists,
        }

        # Add Q_NL limits when nonlinear AE is trained
        if any("Q_NL" in m for m in metrics_list):
            q_nl_list = [m["Q_NL"] for m in metrics_list]
            limits["Q_NL"] = calculate_metric_limits(q_nl_list)
            limits["Q_NL_list"] = q_nl_list

        return limits

    def calculate_limits(
        self, health_data: pd.DataFrame, window_size: int, num_stds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run sliding windows on 'health_data', calculating Q limits (public interface).

        Args:
            health_data (pd.DataFrame): The health data to calculate limits for.
            window_size (int): The size of the sliding window.
            num_stds (int, optional): The number of standard deviations to use for defining limits.
                Defaults to DEFAULT_NUM_STDS.

        Returns:
            Dict[str, Any]: Dictionary with Q limits and intermediate lists for plotting.

        """
        return self._calculate_limits(health_data, window_size, num_stds)

    def transform_current_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Apply the transform() method on sliding windows of size self.window_size along 'data'.

        Args:
            data (pd.DataFrame): The data to transform.

        Returns:
            Dict[str, Any]: Dictionary with Q and W_AE values for each window.

        Raises:
            ValueError: If model is not fitted.

        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        metrics_list = []
        # PCW mode uses N-W windows (matches MATLAB/RMWPCA_SC), other modes use N-W+1
        n_windows = len(data) - self.window_size + (0 if self.wae_method == "pcw" else 1)
        for i in range(n_windows):
            current_data = data.iloc[i : i + self.window_size]
            metrics = self.transform(current_data)
            metrics_list.append(metrics)

        result = {
            "index": [m["index"] for m in metrics_list],
            "Q": [m["Q"] for m in metrics_list],
            "W_AE": [m["W_AE"] for m in metrics_list],
        }
        if any("Q_NL" in m for m in metrics_list):
            result["Q_NL"] = [m.get("Q_NL") for m in metrics_list]
        return result

    def compute_qall_filtered(
        self,
        data: pd.DataFrame,
        majority_threshold: int = 2,
        relaxed_margin: float = 0.05,
        use_strict_detection: bool = False,
    ) -> Dict[str, Any]:
        """Compute Qall_Filtered with majority voting gating based on W_AE component detections.

        This method implements the MATLAB-aligned Qall_Filtered computation that filters Q_NL
        based on majority voting from W_AE component detections. Q_NL values are kept only when
        >= majority_threshold components are detected as anomalous, otherwise set to 0.

        Args:
            data (pd.DataFrame): Input data for sliding window analysis.
            majority_threshold (int): Minimum number of W_AE components that must be out of limits
                for Q_NL to be kept (not filtered to 0). Defaults to 2.
            relaxed_margin (float): Margin for relaxed detection thresholds. When > 0,
                applies tighter thresholds: UCL×(1-margin), LCL×(1+margin). This matches
                MATLAB's 5% margin (0.05) for enhanced sensitivity. Defaults to 0.05.
            use_strict_detection (bool): If True, uses exact limit boundaries for component detection.
                If False (default), uses relaxed_margin for enhanced detection sensitivity.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "Q_NL": Array of Q_NL values for each window (only available if nl_ae is trained)
                - "W_AE": Array of W_AE component arrays for each window
                - "W_AE_detections": Array of per-component detection arrays (bool)
                - "soma_det": Array of detection counts per window (int)
                - "majority_voting": Array of boolean majority voting results
                - "Qall_Filtered": Array of filtered Q_NL values (Q_NL where majority voting=True, 0 otherwise)
                - "index": Array of timestamps for each window
                - "limits_used": Dictionary of W_AE limits used for detection

        Raises:
            ValueError: If model is not fitted, Q_NL is not available, or data is invalid.

        Note:
            - Requires Q_NL to be available (nl_hidden_size must have been set during training)
            - Uses the same W_AE limits computed during model fitting
            - Follows the MATLAB ae_aenl pipeline logic for enhanced fault detection

        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.nl_ae is None:
            raise ValueError("Q_NL not available. Model must be trained with nl_hidden_size parameter.")

        if self.limits is None:
            raise ValueError("Limits not available. Ensure model was properly fitted with health data.")

        # Get transformed data with Q_NL and W_AE
        transformed_data = self.transform_current_data(data)

        if "Q_NL" not in transformed_data:
            raise ValueError("Q_NL not found in transformed data. Ensure nl_ae is properly trained.")

        q_nl_array = np.array(transformed_data["Q_NL"])
        wae_array = np.array(transformed_data["W_AE"])  # Shape: (n_windows, n_components)
        n_windows, n_components = wae_array.shape

        # Extract W_AE limits from fitted model
        wae_limits = {}
        for i in range(n_components):
            wae_key = f"W_AE_{i+1}"  # Formato correto com underscore
            if wae_key in self.limits["W_AE"]:
                wae_limits[i] = self.limits["W_AE"][wae_key]
            else:
                raise ValueError(
                    f"W_AE limits not found for component {i+1}. Available keys: {list(self.limits['W_AE'].keys())}"
                )

        # Compute per-component detections
        det_components = []
        for k in range(n_components):
            limits_k = wae_limits[k]
            low_limit = limits_k["low"]
            high_limit = limits_k["high"]

            if use_strict_detection:
                # Strict detection: exact limit boundaries
                det_k = (wae_array[:, k] < low_limit) | (wae_array[:, k] > high_limit)
            else:
                # Relaxed detection: tighter thresholds for enhanced sensitivity
                # MATLAB ae_aenl uses UCL×0.95, LCL×1.05 → tighter detection bounds
                relaxed_low = low_limit * (1 + relaxed_margin) if low_limit < 0 else low_limit / (1 + relaxed_margin)
                relaxed_high = (
                    high_limit * (1 - relaxed_margin) if high_limit > 0 else high_limit / (1 - relaxed_margin)
                )
                det_k = (wae_array[:, k] < relaxed_low) | (wae_array[:, k] > relaxed_high)

            det_components.append(det_k)

        # Convert to numpy array for easier manipulation
        det_array = np.array(det_components)  # Shape: (n_components, n_windows)

        # Compute detection sum per window (majority voting basis)
        soma_det = np.sum(det_array, axis=0)  # Shape: (n_windows,)

        # Majority voting: keep Q_NL only when >= majority_threshold components are anomalous
        majority_voting = soma_det >= majority_threshold

        # Qall_Filtered: Q_NL where majority voting is True, 0 otherwise
        qall_filtered = np.where(majority_voting, q_nl_array, 0.0)

        return {
            "Q_NL": q_nl_array,
            "W_AE": wae_array,
            "W_AE_detections": det_array.T,  # Transpose for (n_windows, n_components) consistency
            "soma_det": soma_det,
            "majority_voting": majority_voting,
            "Qall_Filtered": qall_filtered,
            "index": transformed_data["index"],
            "limits_used": wae_limits,
        }

    def plot_health_components(self) -> None:
        """Plot the Q series (reconstruction error) of the 'health_data' used for training."""
        if self.limits is None:
            raise ValueError("Limits not fitted. Call fit() method first.")

        fig_Q = px.line(title="Q Series (Reconstruction Error)")
        fig_Q.add_scatter(y=self.limits["Q_list"], mode="lines", name="Q", line_color="orange")
        fig_Q.show()

    def plot_transformed_data(self, transformed_data: dict[str, Any], view_mode: str = "normal") -> None:
        """Plot Q (reconstruction error) of the transformed data.

        Args:
            transformed_data (dict[str, Any]): The transformed data to plot.
            view_mode (str, optional): The view mode for the plot. Can be "normal" or "reduced".

        """
        if self.limits is None:
            raise ValueError("Limits not fitted. Call fit() method first.")

        if view_mode not in ["normal", "reduced"]:
            raise ValueError("view_mode must be 'normal' or 'reduced'")

        # Q
        fig_Q = px.line(title="Q (Reconstruction Error) with Critical Limits")
        fig_Q.add_scatter(
            x=transformed_data["index"], y=transformed_data["Q"], mode="lines", name="Q", line_color="orange"
        )
        fig_Q.add_hline(
            y=self.limits["Q"]["high"],
            line_dash="dash",
            line_color="red",
            annotation_text="High Q Limit",
            annotation_position="top right",
        )
        fig_Q.add_hline(
            y=self.limits["Q"]["low"],
            line_dash="dash",
            line_color="green",
            annotation_text="Low Q Limit",
            annotation_position="bottom right",
        )

        fig_Q.update_layout(template="plotly_white")

        if view_mode == "reduced":
            fig_Q.update_yaxes(
                range=[
                    2 * self.limits["Q"]["low"] if self.limits["Q"]["low"] < 0 else (self.limits["Q"]["low"]) / 2,
                    2 * self.limits["Q"]["high"],
                ]
            )

        fig_Q.show()

    def __repr__(self) -> str:
        """Return string representation of the class."""
        return (
            f"MWAutoEncoder(model_arch={self.model_arch}, "
            f"hidden_layers={self.hidden_layers}, use_scaler={self.use_scaler}, "
            f"use_gating={self.use_gating}, wae_method={self.wae_method}, "
            f"center_per_window={self.center_per_window}, activation={self.activation})"
        )

    def _predict_error(self, data: pd.DataFrame) -> np.ndarray:
        """Return the reconstruction error (SSE) for each sample in data.

        Used internally by the SHAP methods (KernelExplainer) in BaseDetector.

        Args:
            data (pd.DataFrame): Input data for which the reconstruction error will be calculated.

        Returns:
            np.ndarray: An array containing the reconstruction error (SSE) for each sample.

        Raises:
            ValueError: If the model is not trained (i.e., self.model is None).

        """
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")

        # Apply scaler if it was used during training
        if self.scaler is not None:
            data_scaled = pd.DataFrame(
                self.scaler.transform(data),
                index=data.index,
                columns=data.columns,
            )
        else:
            data_scaled = data

        # Per-window centering: subtract column-wise mean of this batch
        if self.center_per_window:
            data_scaled = data_scaled - data_scaled.mean()

        self.model.eval()

        data_array = data_scaled.to_numpy().astype(np.float32)
        data_tensor = torch.tensor(data_array, dtype=torch.float32)

        if self.model_arch == "cnn":
            # For CNNs, we need an extra dimension for channels
            data_tensor = data_tensor.unsqueeze(1)

        data_tensor = data_tensor.to(self.device)

        with torch.no_grad():
            x_recon, _ = self.model(data_tensor)

        if self.model_arch == "mlp":
            recon_error = torch.sum((data_tensor - x_recon) ** 2, dim=1)
        else:
            recon_error = torch.sum((data_tensor - x_recon) ** 2, dim=[1, 2])

        return recon_error.cpu().numpy()

    def predict(
        self,
        transformed_data: Dict[str, Any],
        metric: str = "Q",
        threshold: float = 0.5,
        rolling_window: int = 1,
        detection_side: str = "both",
    ) -> Dict[str, Any]:
        """Predict anomalies based on reconstruction errors.

        This method aligns with the BaseDetector interface.

        Args:
            transformed_data (Dict[str, Any]): Transformed data metrics from transform_current_data().
            metric (str, optional): Metric to use for prediction. Defaults to "Q".
            threshold (float, optional): Not used for autoencoder (uses limits instead). Defaults to 0.5.
            rolling_window (int, optional): Size of rolling window for smoothing detections. Defaults to 1.
            detection_side (str, optional): Which side of the limit to detect.
                "upper", "lower", or "both" (default). See detect() for details.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'fault_detected': Boolean indicating if any anomaly was detected
                - 'detections': Series of boolean detections for each sample
                - 'index': Timestamps/indices of the samples

        Raises:
            RuntimeError: If model is not fitted.

        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        detection_result = self.detect(
            transformed_data, metric=metric, rolling_window=rolling_window, detection_side=detection_side
        )

        # Determine if any fault was detected
        detections = detection_result["detections"]
        if hasattr(detections, "any"):
            fault_detected = bool(detections.any())
        else:
            fault_detected = bool(any(detections))

        return {
            "fault_detected": fault_detected,
            "detections": detection_result["detections"],
            "index": detection_result["index"],
        }

    def detect(
        self,
        transformed_data: Dict[str, Any],
        metric: str = "Q",
        rolling_window: int = 1,
        detection_side: str = "both",
    ) -> Dict[str, Any]:
        """Detect anomalies based on reconstruction errors.

        Args:
            transformed_data (Dict[str, Any]): Transformed data containing metric values.
            metric (str, optional): Metric to be used for detections. Defaults to "Q".
            rolling_window (int, optional): Size of rolling window for smoothing detections. Defaults to 1.
            detection_side (str, optional): Which side of the limit to detect.
                "upper": Only detect values above high limit.
                "lower": Only detect values below low limit.
                "both": Detect values outside either limit (default, backward compatible).

        Returns:
            Dict[str, Any]: Dictionary containing index and detection arrays.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If detection_side is invalid.

        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling detect.")

        if detection_side not in ("upper", "lower", "both"):
            raise ValueError(f"Invalid detection_side '{detection_side}'. Use 'upper', 'lower', or 'both'.")

        index = transformed_data["index"]
        errors = transformed_data[metric]

        # Convert to numpy array if needed
        if isinstance(errors, list):
            errors = np.array(errors)

        if detection_side == "upper":
            detections = errors > self.limits[metric]["high"]
        elif detection_side == "lower":
            detections = errors < self.limits[metric]["low"]
        else:
            detections = (errors < self.limits[metric]["low"]) | (errors > self.limits[metric]["high"])

        if rolling_window > 1:
            detections = (
                pd.Series(detections)
                .rolling(window=rolling_window)
                .apply(lambda x: all(x), raw=True)
                .fillna(0)
                .astype(bool)
            )

        return {
            "index": index,
            "detections": detections,
        }

    def plot_detections(self, detection_result: Dict[str, Any]) -> None:
        """Plot anomaly detection results.

        Args:
            detection_result (Dict[str, Any]): Dictionary containing 'index' and 'detections' arrays.

        """
        index = detection_result["index"]
        detection_values = detection_result["detections"]

        fig = go.Figure()
        fig.update_layout(
            title="Detections Over Time",
            yaxis_title="Detection Status",
            legend_title="Legend",
            template="plotly_white",
        )
        fig.add_trace(go.Scatter(x=index, y=detection_values, mode="lines+markers", name="Detections"))
        fig.show()
