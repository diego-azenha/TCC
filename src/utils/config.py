"""Configuration dataclasses for NeuralFactors model."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class EncoderConfig:
    """DeepSet encoder hyperparameters.

    φ network: MLP(4 + F → phi_hidden → phi_hidden, GELU) — per-stock, shared weights.
    After masked mean-pooling:
    ρ network: MLP(phi_hidden → rho_hidden → rho_hidden, GELU) — context aggregation.
    Output heads: μ_q (rho_hidden → F), log σ_q (rho_hidden → F).
    """
    phi_hidden: int = 64    # Width of φ MLP hidden layers
    rho_hidden: int = 128   # Width of ρ MLP hidden layers

    def __post_init__(self):
        if self.phi_hidden <= 0:
            raise ValueError(f"phi_hidden must be positive, got {self.phi_hidden}")
        if self.rho_hidden <= 0:
            raise ValueError(f"rho_hidden must be positive, got {self.rho_hidden}")


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    # Core architecture
    num_factors: int = 16   # F; 16 for Brazilian equity universe
    hidden_size: int = 256  # h — hidden dimension for StockEmbedder transformer

    # Input dimensions (must be set based on data)
    d_ts: int = None      # Dimension of time-series features per timestep
    d_static: int = None  # Dimension of static features

    # Sequence model parameters
    lookback: int = 256   # L — lookback window size
    nhead: int = 4        # Number of attention heads
    num_layers: int = 2   # Number of transformer encoder layers
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.25

    # Output parameter constraints
    sigma_min: float = 0.1    # σ lower bound (normalised space)
    sigma_max: float = 3.0    # σ upper bound (normalised space)
    alpha_max: float = 3.0    # α clamp bound (normalised space)

    # Sub-configuration (initialised in __post_init__)
    encoder_config: 'EncoderConfig' = None

    def __post_init__(self):
        if self.d_ts is None:
            raise ValueError("d_ts must be specified")
        if self.d_static is None:
            raise ValueError("d_static must be specified")
        if self.num_factors <= 0:
            raise ValueError(f"num_factors must be positive, got {self.num_factors}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.lookback <= 0:
            raise ValueError(f"lookback must be positive, got {self.lookback}")
        if self.nhead <= 0:
            raise ValueError(f"nhead must be positive, got {self.nhead}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")
        if self.sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {self.sigma_min}")
        if self.sigma_max <= self.sigma_min:
            raise ValueError(f"sigma_max must be > sigma_min")
        if self.alpha_max <= 0:
            raise ValueError(f"alpha_max must be positive, got {self.alpha_max}")
        if self.encoder_config is None:
            self.encoder_config = EncoderConfig()


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6

    # Training procedure
    max_steps: int = 250_000
    batch_size: int = 1

    # Validation
    val_every_n_steps: int = 10_000

    # Polyak averaging
    use_polyak: bool = True
    polyak_start_step: int = None   # Auto: max_steps // 2
    polyak_alpha: float = 0.999

    # Posterior collapse prevention
    free_bits_lambda: float = 0.1   # Nats/factor; 0 disables

    # Detached sigma loss (prevents alpha shortcut / sigma collapse)
    lambda_sigma: float = 1.0      # Weight on L_sigma calibration loss
    sigma_ref_ema: float = 0.99    # EMA momentum for sigma_ref buffer

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Data normalisation
    normalize_returns: bool = True
    returns_std: float = None

    # Reproducibility / device
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.polyak_start_step is None:
            self.polyak_start_step = self.max_steps // 2
        if self.polyak_start_step >= self.max_steps:
            raise ValueError("polyak_start_step must be < max_steps")
        if not 0.0 < self.polyak_alpha < 1.0:
            raise ValueError(f"polyak_alpha must be in (0, 1), got {self.polyak_alpha}")
        if self.free_bits_lambda < 0.0:
            raise ValueError(f"free_bits_lambda must be >= 0, got {self.free_bits_lambda}")
        if self.lambda_sigma < 0.0:
            raise ValueError(f"lambda_sigma must be >= 0, got {self.lambda_sigma}")
        if not 0.0 < self.sigma_ref_ema < 1.0:
            raise ValueError(f"sigma_ref_ema must be in (0, 1), got {self.sigma_ref_ema}")


def get_default_config(d_ts: int, d_static: int) -> tuple:
    """Get default configuration with specified feature dimensions.

    Returns:
        Tuple of (ModelConfig, EncoderConfig)
    """
    model_config = ModelConfig(d_ts=d_ts, d_static=d_static)
    return model_config, model_config.encoder_config


__all__ = [
    "ModelConfig",
    "EncoderConfig",
    "TrainingConfig",
    "get_default_config",
]
