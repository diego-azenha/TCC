# NeuralFactors: Replication Implementation

Implementation of "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities" by Achintya Gopal (arXiv:2408.01499v1).

**Project Status**: ✅ Model architecture complete, training pipeline operational, data quality issues resolved.

## Recent Updates

- **2026-02-07**: Fixed critical data quality issue causing NaN loss during training
  - Root cause: Infinite values in log returns from zero/negative prices
  - Solution: Replace `±Inf` with `NaN` in `compute_returns()` for proper masking
  - Training now proceeds stably with all model parameters learnable
  
- **Architecture**: All components implemented and tested (Stock Embedder, Encoder, Decoder, Prior, NeuralFactors)
- **Training**: CIWAE loss with k=20 importance samples, Polyak averaging, gradient clipping
- **Data**: Brazilian IBX stocks (2005-2025), ~100 constituents, 12 time-series + 3 static features

## Repository Structure

```
TCC/
├── data/
│   ├── _raw_data/          # Raw data from LSEG Refinitiv/FRED
│   │   ├── daily/          # Daily frequency data
│   │   └── quarterly/      # Quarterly frequency data
│   ├── cleaned/            # Cleaned and normalized data
│   ├── parquets/           # Parquet format for efficient loading
│   └── processing/         # Data cleaning scripts
│       ├── csv_cleaning.py
│       └── parquets.py
├── src/
│   ├── models/             # Core model components
│   │   ├── stock_embedder.py    # [DONE] Stock feature encoder
│   │   ├── encoder.py           # [DONE] Variational posterior q(z|r)
│   │   ├── decoder.py           # [DONE] Likelihood p(r|z)
│   │   ├── prior.py             # [DONE] Prior distribution p(z)
│   │   ├── neuralfactors.py     # [DONE] Main model integration
│   │   ├── lightning_module.py  # [DONE] PyTorch Lightning wrapper
│   │   └── __init__.py          # [DONE] Package exports
│   ├── utils/              # Configuration and utilities
│   │   ├── config.py            # [DONE] Model & training config
│   │   ├── data_utils.py        # [DONE] Data loading & preprocessing
│   │   ├── dataset.py           # [DONE] PyTorch Dataset implementation
│   │   └── __init__.py          # [DONE] Package exports
│   └── evaluation/         # Evaluation metrics and analysis
├── scripts/
│   ├── train.py            # [DONE] Training script with CLI
│   └── test.py             # [DONE] Testing and evaluation with article metrics
├── checkpoints/            # Saved model checkpoints
├── logs/                   # TensorBoard training logs
├── NEURALFACTORSARTICLE.txt  # Original paper text
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

**Legend**: [DONE] Complete | [WIP] In Progress | [TODO] Not Started

---

## Model Components

### 1. Stock Embedder (`src/models/stock_embedder.py`) [DONE]

**Purpose**: Encodes stock features into factor-specific parameters for each asset.

**Architecture**:
- **Input**: 
  - `S[N, L, d_ts]`: Time-series features with lookback window (L=256 days)
  - `S_static[N, d_static]`: Static features (industry, financials)
- **Output**: 
  - `alpha[N]`: Idiosyncratic return component
  - `beta[N, F]`: Factor exposures (F=64 factors)
  - `sigma[N]`: Scale parameters (Student-t)
  - `nu[N]`: Degrees of freedom (Student-t)

**Pipeline**:
1. Per-timestep projection: `d_ts → h` (h=256)
2. Transformer encoder: Processes temporal sequence with attention
3. Feature fusion: Concatenates temporal state with static features
4. Two-layer MLP: Produces intermediate representation
5. Linear heads: Outputs parameters with softplus constraints

**Key Features**:
- Handles variable number of stocks per day (N can vary)
- Validates lookback window size (paper optimal: L=256)
- Integrates with `ModelConfig` for centralized hyperparameters
- Paper-compliant defaults: F=64, h=256, dropout=0.25

**Design Notes**:
- Batch semantics: One forward pass = all stocks from ONE day
- Not batched over multiple days (as per paper Section 3.5)
- Factor exposures β are time-varying, conditioned on features

---

### 2. Encoder (`src/models/encoder.py`) [DONE]

**Purpose**: Computes variational posterior q(z|r) for latent factors given observed returns.

**Implementation**: Analytical closed-form solution using linear regression formulation.

**Key Formula** (Paper Equation 8):
```
q(z|r) ≈ N(μ_q, Σ_q)
where:
  Σ_q = (Σ_z^-1 + B^T Σ_x^-1 B)^-1
  μ_q = Σ_q (Σ_z^-1 μ_z + B^T Σ_x^-1 (r - α))
```

**Architecture**:
- **Input**: 
  - Stock parameters: `alpha[N], B[N,F], sigma[N]`
  - Observed returns: `r[N]`
  - Prior parameters: `mu_z[F], Sigma_z[F,F], nu_z` (optional)
- **Output**: 
  - `mu_q[F]`: Posterior mean
  - `L_q[F,F]`: Cholesky factor of posterior covariance (for reparameterization)
  - `Sigma_q[F,F]`: Full covariance (optional)
  - `prec[F,F]`: Precision matrix (optional)

**Key Features**:
- **Moment matching**: Converts Student-T prior to Normal via variance scaling `nu/(nu-2)`
- **Numerical stability**: fp64 computation, adaptive Cholesky jitter (2x multiplier)
- **Efficient inversion**: Uses Cholesky solve instead of direct matrix inverse
- **Masking support**: Handles missing/invalid stocks via boolean mask

**Usage Context**:
- **Training only**: Used to compute posterior for CIWAE loss
- **Not used in inference**: Inference samples directly from prior p(z)

**Design Notes**:
- Approximates Student-T posterior as Gaussian (tractable for VAE training)
- Jitter starts at 1e-6 and increases by 2x until Cholesky succeeds
- Returns Cholesky factor L_q for efficient sampling via reparameterization trick

---

### 3. Decoder (`src/models/decoder.py`) [DONE]

**Purpose**: Computes likelihood p(r|z) and samples returns given latent factors.

**Model**: Linear factor model with Student-T noise (Paper Section 3.2):
```
r_i ~ Student-T(alpha_i + beta_i^T z, sigma_i, nu_i)
```

**Architecture**:
- **Input**:
  - Stock parameters: `alpha[N], B[N,F], sigma[N], nu[N]`
  - Latent factors: `z[batch,K,F]` (K samples)
- **Output**:
  - Log-likelihood: `log p(r|z)` for each sample
  - Samples: `r[batch,N,K]` or `r[batch,N]` if K=1

**Key Functions**:
1. **log_pdf_r_given_z**: Compute log p(r|z) for likelihood evaluation
2. **sample_r_given_z**: Sample returns conditional on latent factors
3. **marginal_mean**: Compute E[r] = alpha + B @ mu_z analytically
4. **marginal_covariance**: Compute full Cov[r] = diag(σ²) + B Σ_z B^T
5. **marginal_cov_actionable**: Compute portfolio variance w^T Cov[r] w efficiently

**Key Features**:
- **Numerically stable**: Uses log1p for Student-T pdf computation
- **Efficient sampling**: Vectorized Student-T sampling with fallback
- **Closed-form statistics**: No sampling needed for mean/covariance
- **Shape safety**: Rejects ambiguous z shapes (enforces explicit 3D)

**Usage Context**:
- **Training**: Compute log p(r|z) for CIWAE loss
- **Inference**: Sample r|z from prior or compute marginal statistics

**Design Notes**:
- Student-T provides heavy tails for robust modeling of outliers
- Linear factor structure enables classical risk analysis (mean-variance optimization)
- Sampling uses correct Gamma(nu/2, 0.5) for Student-T construction

---

### 4. Prior Distribution (`src/models/prior.py`) [DONE]

**Purpose**: Learnable Student-T prior p(z) over latent factors.

**Model**: Time-homogeneous Student-T (Paper Section 3.1):
```
z ~ Student-T(nu_z, mu_z, sigma_z)
```

**Architecture**:
- **StudentTPrior(nn.Module)**: Learnable prior with reparameterized parameters
- **Parameters**:
  - `mu_z[F]`: Mean of latent factors
  - `log_sigma_z[F]`: Log-scale (ensures positivity)
  - `log_nu_z_minus_4`: Log degrees of freedom minus 4 (ensures nu>4)
- **Output**: Samples z or log p(z) for IWAE loss

**Key Methods**:
1. **get_params()**: Returns constrained (μ_z, σ_z, ν_z) with enforced constraints
2. **sample(batch, K)**: Samples [batch,K,F] from Student-T via reparameterization
3. **log_prob(z)**: Computes log p(z) summing over independent factors
4. **to_normal_params()**: Moment matching for encoder (returns μ, Σ with variance scaled by ν/(ν-2))

**Key Features**:
- **Learnable**: All parameters updated via gradient descent during training
- **Constrained**: Log-parameterization ensures σ>0 and ν>4 (finite fourth moment)
- **Moment matching**: Converts to Normal for encoder analytical posterior
- **Independent factors**: Assumes diagonal covariance (F independent Student-T distributions)

**Usage Context**:
- **Training**: Compute log p(z) for CIWAE prior term, provide Normal params to encoder
- **Inference**: Sample z from prior (no encoder)

**Design Notes**:
- Paper sets μ_z=0 without loss of generality (Section 3.2)
- ν_z>4 constraint ensures finite kurtosis for stability
- Student-T provides heavy tails for robust factor modeling

---

### 5. NeuralFactors Model (`src/models/neuralfactors.py`) [DONE]

**Purpose**: Main VAE integrating all components with train/inference modes.

**Model**: Complete NeuralFactors architecture (Paper Section 3):
```
Training:  q(z|r) -> z -> p(r|z)  [uses encoder]
Inference: p(z) -> z -> p(r|z)    [no encoder]
```

**Architecture**:
- **NeuralFactors(nn.Module)**: Complete VAE with StockEmbedder + Prior + Encoder/Decoder
- **Components**:
  - `embedder`: StockEmbedder generating (α, β, σ, ν)
  - `prior`: StudentTPrior for p(z)
  - Uses encoder function for q(z|r) during training
  - Uses decoder functions for p(r|z) in both modes

**Key Methods**:
1. **encode(S, S_static, r)**: Generate parameters and compute posterior q(z|r)
2. **compute_iwae_loss(...)**: CIWAE loss with K importance samples (Paper Eq. 7)
3. **predict(S, S_static, K)**: Sample from prior and generate return predictions
4. **forward(...)**: Automatic train/inference dispatch based on r presence

**Key Features**:
- **Train mode**: Uses encoder for analytical posterior, computes CIWAE loss
- **Inference mode**: Samples from prior p(z), no encoder needed
- **IWAE loss**: log p(r|z) + log p(z) - log q(z|r) with importance weighting
- **Diagnostics**: Returns log-likelihood, KL divergence, importance weights
- **Marginal statistics**: Computes E[r] and Cov[r] closed-form without sampling

**Usage Context**:
- **Training**: Call with (S, S_static, r) to get loss dictionary
- **Inference**: Call with (S, S_static) to get predictions dictionary
- Integrates with PyTorch Lightning for training loop

**Design Notes**:
- Encoder only invoked during training (q(z|r) approximates intractable posterior)
- Importance sampling with K=20 samples recommended (paper default)
- Returns both sampled predictions and analytical moments for risk analysis

**Planned**:
- Training mode: Uses encoder to compute posterior q(z|r)
- Inference mode: Samples from prior p(z) (no encoder)
- CIWAE loss computation with k=20 importance samples
- Analytical mean/covariance computation

---

## Configuration (`src/utils/config.py`) [DONE]

Centralized hyperparameter configuration with paper defaults:

```python
ModelConfig:
    num_factors: 64        # F (paper Table 3)
    hidden_size: 256       # h (paper Section 3.5)
    lookback: 256          # L (paper Table 3)
    dropout: 0.25          # paper Section 3.5
    nhead: 4               # attention heads
    num_layers: 2          # transformer layers
    d_ts: int              # time-series feature dimension (auto-discovered)
    d_static: int          # static feature dimension (auto-discovered)

TrainingConfig:
    learning_rate: 1e-4    # Adam learning rate
    weight_decay: 1e-6     # L2 regularization
    max_steps: 100000      # total training steps
    val_every_n_steps: 1000  # validation frequency
    num_iwae_samples: 20   # k samples for CIWAE loss
    use_polyak: True       # exponential moving average
    polyak_alpha: 0.999    # EMA coefficient
    polyak_start_step: 50000  # when to start EMA

PriorConfig:
    mu_z_init: 0.0         # initial mean
    sigma_z_init: 1.0      # initial scale
    nu_z_init: 10.0        # initial df

EncoderConfig:
    jitter_multiplier: 2.0  # Cholesky stability
    use_fp64: True         # numerical precision
```

---

## Data Pipeline

### Data Utilities (`src/utils/data_utils.py`) [DONE]

**Purpose**: Data loading, preprocessing, and returns computation for IBX Brazilian stocks.

**Key Functions**:
1. **load_parquets()**: Load time-series features, static features, and prices
   - Handles European decimal format (comma separator)
   - Automatic date parsing with `dayfirst=True`
   - Uses `decimal=','` parameter for proper numeric conversion

2. **compute_returns()**: Calculate log or simple returns from prices
   - Filters infinite values from log(0) using `np.isfinite()`
   - Handles missing data gracefully

3. **compute_returns_std_from_train()**: Compute normalization std from training period
   - Filters training data before computing returns
   - Returns std across all valid (finite) returns

4. **normalize_returns()**: Normalize returns by standard deviation
   - Paper uses std≈0.0267 for S&P 500, IBX data gives std≈0.0627

5. **split_by_date()**: Split data into train/val/test by date ranges
   - Adjusted for IBX data availability (2005-2025)

**Data Format Requirements**:
- **CSV files**: Semicolon-delimited (`;`), comma decimals (`,`)
- **Date column**: `DATES` with format `DD/MM/YYYY`
- **Parquet files**: Long format with `date`, `ticker`, feature columns

### Dataset (`src/utils/dataset.py`) [DONE]

**Purpose**: PyTorch Dataset providing lookback windows for training/validation.

**Architecture**:
- **NeuralFactorsDataset**: Loads data, computes returns, builds lookback tensors
- **collate_fn**: Custom collation flattening batch and stock dimensions

**Data Flow**:
1. Load parquets and prices CSV (using `load_parquets`)
2. Compute log returns (using `compute_returns`)
3. Normalize returns by training std
4. Convert returns from wide to long format (date, ticker, return)
5. Split by date ranges
6. For each date, build lookback windows for all available stocks

**Key Features**:
- Returns one trading day per `__getitem__` call
- Handles variable universe (stocks come/go over time)
- Validates sufficient lookback history (requires L=256 days)
- Automatically merges time-series and static features by (date, ticker)

**Output Shapes**:
- `S[N, L, d_ts]`: Time-series features with lookback
- `S_static[N, d_static]`: Static features for each stock
- `r[N]`: Next-day returns for each stock
- `mask[N]`: Boolean mask indicating valid stocks

### Lightning Module (`src/models/lightning_module.py`) [DONE]

**Purpose**: PyTorch Lightning wrapper for automated training, validation, and model management.

**Key Responsibilities**:
1. **Training Loop**: 
   - Computes CIWAE loss with k=20 importance samples
   - Logs loss, log-likelihood, KL divergence, effective sample size (ESS)
   
2. **Validation Loop**:
   - Uses 100 importance samples for more accurate NLL estimation
   - Tracks validation metrics for model selection

3. **Optimizer Configuration**:
   - Adam optimizer with lr=1e-4, weight_decay=1e-6
   - Automatic learning rate scheduling

4. **Polyak Averaging**:
   - Maintains exponential moving average of model weights
   - Starts at step 50,000 with α=0.999
   - Provides more stable generalization

5. **Checkpointing & Logging**:
   - Saves best models based on validation loss
   - TensorBoard logging for all metrics
   - Config serialization for reproducibility

**Integration**:
- Works seamlessly with PyTorch Lightning `Trainer`
- Handles GPU/CPU device placement automatically
- Manages gradient accumulation and mixed precision

---

## Training Pipeline [DONE]

**Status**: Complete training infrastructure aligned with paper Section 3.5.

**Components**:
1. **TrainingConfig** ([src/utils/config.py](src/utils/config.py)): All training hyperparameters
2. **Data Preprocessing** ([src/utils/data_utils.py](src/utils/data_utils.py)): Returns computation, normalization, splitting
3. **NeuralFactorsDataset** ([src/utils/dataset.py](src/utils/dataset.py)): PyTorch Dataset with lookback windows
4. **NeuralFactorsLightning** ([src/models/lightning_module.py](src/models/lightning_module.py)): Lightning wrapper with Polyak averaging
5. **Training Script** ([scripts/train.py](scripts/train.py)): Main training entry point

**Training Procedure** (Paper Section 3.5):
- **Optimizer**: Adam with lr=1e-4, weight_decay=1e-6
- **Loss**: CIWAE with k=20 importance samples (Paper Equation 3)
- **Batch Size**: 1 (all stocks from one trading day)
- **Total Training**: 100,000 gradient updates
- **Validation**: Every 1,000 steps using NLL_joint metric
- **Polyak Averaging**: Starts at step 50,000 with α=0.999
- **Model Selection**: Checkpoint with lowest validation loss

**Data Splits** (Adjusted for IBX data availability):
- **Training**: 2005-01-01 to 2018-12-31 (14 years, 3458 trading days)
- **Validation**: 2019-01-01 to 2022-12-31 (4 years, 994 trading days)
- **Test**: 2023-01-01 to 2025-11-04 (3 years)
- **Note**: Original paper used S&P 500 (1996-2023), adjusted for Brazilian data range

**Usage**:
```bash
# Install dependencies (including CUDA-enabled PyTorch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning tensorboard pandas pyarrow

# Quick test with fast_dev_run (1 batch only)
python scripts/train.py --fast_dev_run

# Full training with default paper hyperparameters
python scripts/train.py --data_dir data --checkpoint_dir checkpoints

# Custom hyperparameters
python scripts/train.py \
    --num_factors 64 \
    --hidden_size 256 \
    --lookback 256 \
    --num_iwae_samples 20 \
    --max_steps 100000 \
    --learning_rate 1e-4 \
    --gpus 1  # set to 0 for CPU-only training
```

**Output**:
- **Checkpoints**: Saved to `checkpoints/<experiment>/` directory
- **Best Model**: Selected by lowest `val/loss` (NLL_joint)
- **Polyak Model**: Saved separately as `polyak_model.pt`
- **Config**: Training configuration saved as `config.json`
- **Logs**: TensorBoard logs in `logs/<experiment>/`

**Key Features**:
- Automatic feature dimension discovery from parquets
- Returns normalization using training set statistics (std≈0.0627 for IBX)
- Lookback window construction with universe changes handling
- Polyak-averaged weights for stable inference
- GPU acceleration with automatic CPU fallback
- European decimal format support (comma separator, semicolon delimiter)
- Handles missing/infinite values in returns computation

**Known Issues & Fixes**:
- **European CSV format**: Uses `decimal=','` and `sep=';'` for proper parsing
- **GPU detection**: Automatically falls back to CPU if CUDA unavailable
- **Tensor shapes**: Collate function flattens batch and stock dimensions for model compatibility
- **Data format**: Converts wide-format returns to long format (date, ticker, return)
- **Infinite returns from bad prices**: Fixed in `data_utils.py` by replacing `±Inf` with `NaN` after log returns computation
  - Root cause: Some stocks have zero or negative prices in raw data
  - `np.log(0)` or `np.log(negative)` produces `-Inf` which propagates through the model
  - Solution: `returns_df.replace([np.inf, -np.inf], np.nan)` immediately after computing log returns
  - Masked stocks are then properly excluded during training via the dataset's `fillna(0)` and masking

---

## Data Quality & Troubleshooting

### Critical Fix: Infinite Returns Handling

**Problem**: Training crashes with NaN loss around step 12 due to infinite values in return data.

**Root Cause**:
1. Raw price data contains stocks with `price ≤ 0` (data errors, delistings, or bankruptcies)
2. Log returns computation: `np.log(price_t / price_t-1)` produces `-Inf` when price ≤ 0
3. These `-Inf` values propagate through normalization unchanged
4. Dataset loads returns with `-Inf` into tensors
5. Model's encoder computes `inv_sigma * (r - alpha)` → weighted_resid = `-Inf`
6. Matrix operations with `-Inf` produce `NaN` in posterior mean `mu_q`
7. All downstream computations become `NaN`, causing training failure

**Solution** (implemented in [src/utils/data_utils.py](src/utils/data_utils.py)):
```python
# In compute_returns() function
returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
```

This ensures:
- Invalid returns are converted to `NaN` (not `-Inf`)
- Dataset's `fillna(0)` handles `NaN` properly
- Masking system excludes problematic stocks from loss computation
- Model receives clean data without infinite values

**Verification**:
- Training should proceed smoothly past step 12
- No `NaN` in loss values during training
- Monitor logs for stocks with consistent missing data

**Recommendations**:
- Audit raw price data for stocks with `price ≤ 0`
- Consider pre-filtering stocks with suspicious price patterns
- Log which stocks/dates have invalid returns for analysis

### Numerical Stability Best Practices

**Gradient Clipping**:
- Enabled with `gradient_clip_val=1.0, gradient_clip_algorithm='norm'`
- Prevents gradient explosion in prior parameters and embedder
- Particularly important during early training

**Output Clamping**:
- Stock embedder outputs are clamped to prevent extreme values:
  - `alpha`: `[-100, 100]`
  - `beta`: `[-10, 10]`
  - `sigma`: `[eps, 100]`
  - `nu`: `[4.0, 100]` (ensures finite kurtosis)

**Float64 in Encoder**:
- Encoder uses FP64 for Cholesky decomposition
- Prevents numerical instability in precision matrix computation
- Configured via `EncoderConfig.use_fp64=True`

**Adaptive Jitter**:
- Cholesky decomposition uses adaptive jitter starting at `1e-4`
- Multiplies by 2.0 on failure, max jitter `10.0`
- Ensures positive definiteness of covariance matrices

---

## Evaluation [DONE]

**Status**: Complete evaluation infrastructure with article-format metrics.

### Test Script (`scripts/test.py`)

Implements all evaluation metrics from the paper (Tables 4-7):

**Usage**:
```bash
# Evaluate on test set
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --data_dir data \
    --split test \
    --output_dir results_test

# Quick evaluation (fewer samples)
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --split test \
    --num_joint_samples 20 \
    --num_ind_samples 100 \
    --output_dir results_quick
```

**Metrics Implemented**:
- **Table 4**: Negative log-likelihood (joint and individual)
  - NLL_joint: Log-likelihood of joint distribution across universe
  - NLL_ind: Log-likelihood of individual stock distributions
- **Table 5**: Covariance forecasting
  - MSE: Mean squared error of whitened returns
  - Box's M: Statistical test for covariance equality
- **Table 6**: Value-at-Risk (VaR) calibration
  - Calibration error: Quantile prediction accuracy (100 quantiles)
- **Table 7**: Portfolio optimization
  - Sharpe ratio: Risk-adjusted returns
  - Market benchmark comparison

**Output Files**:
- `results_{split}.json`: Summary metrics in JSON format
- `results_table_{split}.txt`: Formatted table matching article style
- `nll_joint_per_day_{split}.csv`: Daily NLL values for temporal analysis

**Documentation**: See [docs/TEST_SCRIPT_USAGE.md](docs/TEST_SCRIPT_USAGE.md) for detailed usage instructions.

---

## Paper Reference

Gopal, A. (2024). NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities. arXiv:2408.01499v1 [q-fin.ST].

**Key Contributions**:
- Novel VAE-based factor learning without predefined factors
- Linear factor exposures enabling classical risk analysis
- Outperforms PPCA and BDG on S&P 500 constituents
- Closed-form mean/covariance computation

---

## Development Status

**Phase 1: Model Architecture** [DONE]
- [DONE] Configuration system
- [DONE] Stock embedder
- [DONE] Encoder (posterior)
- [DONE] Decoder (likelihood)
- [DONE] Prior distribution
- [DONE] Main model integration

**Phase 2: Training Infrastructure** [DONE]
- [DONE] Data preprocessing utilities
- [DONE] Dataset with lookback windows
- [DONE] CIWAE training loop
- [DONE] Validation and checkpointing
- [DONE] Polyak averaging
- [DONE] Training script with CLI

**Phase 3: Evaluation** [DONE]
- [DONE] Metrics implementation (NLL_ind, NLL_joint, covariance forecasting, VaR)
- [DONE] Test script with article-format output
- [DONE] Portfolio optimization metrics (Sharpe ratio)
- [TODO] Baseline comparisons (PPCA, BDG, GARCH)
- [TODO] Ablation studies
