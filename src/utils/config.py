"""Configuration dataclasses for NeuralFactors model.

All hyperparameter values are based on the paper:
"NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities"
by Achintya Gopal (arXiv:2408.01499v1)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture hyperparameters.
    
    References from paper (Section 3.5):
    - F (num_factors): 64 factors found optimal in ablation study (Table 3)
    - h (hidden_size): 256 used throughout with dropout 0.25
    - lookback: 256 days optimal (Table 3, Section 5.1.4)
    - nhead, num_layers: 4 heads, 2 layers for transformer
    - activation: GELU used in ablations
    """
    
    # Core architecture
    num_factors: int = 64  # F in paper, number of latent factors
    hidden_size: int = 256  # h in paper, hidden dimension for all layers
    
    # Input dimensions (must be set based on data)
    d_ts: int = None  # Dimension of time-series features per timestep
    d_static: int = None  # Dimension of static features
    
    # Sequence model parameters
    lookback: int = 256  # L in paper, lookback window size
    nhead: int = 4  # Number of attention heads
    num_layers: int = 2  # Number of transformer encoder layers
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.25  # Dropout rate (paper Section 3.5)
    
    # Output parameter constraints
    sigma_eps: float = 1e-6  # Minimum sigma (scale parameter)
    nu_offset: float = 4.0  # Minimum nu (degrees of freedom > 4 for finite kurtosis)
    
    # Numerical stability
    use_fp64: bool = False  # Use float64 for encoder/decoder numerical stability
    
    # Sub-configurations (initialized in __post_init__)
    prior_config: 'PriorConfig' = None
    encoder_config: 'EncoderConfig' = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.d_ts is None:
            raise ValueError("d_ts (time-series feature dimension) must be specified")
        if self.d_static is None:
            raise ValueError("d_static (static feature dimension) must be specified")
        
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
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.nu_offset < 4.0:
            raise ValueError(f"nu_offset must be >= 4.0 for finite kurtosis, got {self.nu_offset}")
        
        # Initialize sub-configs if not provided
        if self.prior_config is None:
            self.prior_config = PriorConfig()
        if self.encoder_config is None:
            self.encoder_config = EncoderConfig()


@dataclass
class PriorConfig:
    """Prior distribution hyperparameters.
    
    The prior is p(z) ~ Student-T(nu_z, mu_z, sigma_z) (Section 3.1, Equation 5).
    These are the initial values; the actual prior parameters are learnable.
    
    References from paper:
    - Prior is time-homogeneous (Section 3.5)
    - Student-T distribution used for heavy tails
    - Paper sets mu_z = 0 without loss of generality (Section 3.2)
    """
    
    # Initial values for learnable prior parameters
    mu_z_init: float = 0.0  # Initial mean (paper sets this to 0)
    sigma_z_init: float = 1.0  # Initial scale
    nu_z_init: float = 10.0  # Initial degrees of freedom (>4 required)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.sigma_z_init <= 0:
            raise ValueError(f"sigma_z_init must be positive, got {self.sigma_z_init}")
        if self.nu_z_init <= 4.0:
            raise ValueError(f"nu_z_init must be > 4.0 for finite kurtosis, got {self.nu_z_init}")


@dataclass
class EncoderConfig:
    """Encoder (variational posterior) hyperparameters.
    
    The encoder computes q(z|r, F) ≈ N(mu_q, Sigma_q) using analytical
    closed-form solution (Section 3.3, Equation 8).
    """
    
    # Numerical stability parameters
    eps: float = 1e-8  # Epsilon for inverse sigma computation
    jitter_init: float = 1e-6  # Initial Cholesky jitter
    jitter_max: float = 1e-1  # Maximum Cholesky jitter
    jitter_multiplier: float = 2.0  # Jitter increase factor (changed from 10x to 2x for stability)
    use_fp64: bool = True  # Use float64 for numerical stability in encoder
    
    def __post_init__(self):
        """Validate configuration."""
        if self.jitter_init <= 0 or self.jitter_init >= self.jitter_max:
            raise ValueError(f"Must have 0 < jitter_init < jitter_max, got {self.jitter_init}, {self.jitter_max}")
        if self.jitter_multiplier <= 1.0:
            raise ValueError(f"jitter_multiplier must be > 1.0, got {self.jitter_multiplier}")


@dataclass
class TrainingConfig:
    """Training hyperparameters from paper Section 3.5.
    
    References:
    - Optimizer: Adam with lr=1e-4, weight_decay=1e-6
    - Batch size: 1 (one batch = all stocks from one day)
    - Total training: 100,000 gradient updates
    - Validation: Every 1,000 steps
    - Polyak averaging: Starts at step 50,000 for stability
    - IWAE: k=20 importance samples for training
    """
    
    # Optimizer parameters
    learning_rate: float = 1e-4  # Adam learning rate
    weight_decay: float = 1e-6  # L2 regularization
    
    # Training procedure
    max_steps: int = 100_000  # Total gradient updates
    num_iwae_samples: int = 20  # k in IWAE loss (Equation 3)
    batch_size: int = 1  # Number of days per batch (paper: 1 day = all stocks)
    
    # Validation and checkpointing
    val_every_n_steps: int = 1_000  # Validation frequency
    
    # Polyak (exponential moving average) for stability
    use_polyak: bool = True
    polyak_start_step: int = 50_000  # When to start Polyak averaging
    polyak_alpha: float = 0.999  # EMA decay rate (not specified in paper, common default)
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Data normalization (from paper Section 5)
    normalize_returns: bool = True
    returns_std: float = 0.02672357  # Approx std of returns in training period
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.num_iwae_samples <= 0:
            raise ValueError(f"num_iwae_samples must be positive, got {self.num_iwae_samples}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.polyak_start_step >= self.max_steps:
            raise ValueError(f"polyak_start_step must be < max_steps")
        if not 0.0 < self.polyak_alpha < 1.0:
            raise ValueError(f"polyak_alpha must be in (0, 1), got {self.polyak_alpha}")


def get_default_config(d_ts: int, d_static: int) -> tuple[ModelConfig, PriorConfig, EncoderConfig]:
    """Get default configuration with specified feature dimensions.
    
    Args:
        d_ts: Dimension of time-series features per timestep
        d_static: Dimension of static features
        
    Returns:
        Tuple of (ModelConfig, PriorConfig, EncoderConfig) with paper defaults
    """
    model_config = ModelConfig(d_ts=d_ts, d_static=d_static)
    prior_config = PriorConfig()
    encoder_config = EncoderConfig()
    return model_config, prior_config, encoder_config


__all__ = [
    "ModelConfig",
    "PriorConfig", 
    "EncoderConfig",
    "TrainingConfig",
    "get_default_config",
]
