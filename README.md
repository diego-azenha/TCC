# NeuralFactors: Replication Study

Replication and adaptation of "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities" by Achintya Gopal (arXiv:2408.01499v1), applied to Brazilian IBX equities (2005–2025).

**Thesis context**: This repository accompanies the undergraduate thesis submitted to [institution]. It implements NeuralFactors from scratch in PyTorch, trains it on ~100 IBX constituents, and benchmarks it against a Probabilistic PCA (PPCA) baseline across four quantitative metrics: NLL, covariance MSE, VaR calibration, and min-variance portfolio performance.

## Repository Structure

```
TCC/
├── data/
│   ├── _raw_data/          # Raw data from LSEG Refinitiv/FRED
│   ├── cleaned/            # Cleaned and normalized data
│   ├── parquets/           # Parquet format for efficient loading
│   └── processing/         # Data cleaning scripts
├── src/
│   ├── models/             # Core NeuralFactors model components
│   │   ├── stock_embedder.py    # Stock feature encoder
│   │   ├── encoder.py           # Variational posterior q(z|r)
│   │   ├── decoder.py           # Likelihood p(r|z)
│   │   ├── prior.py             # Prior distribution p(z)
│   │   ├── neuralfactors.py     # Main model integration
│   │   └── lightning_module.py  # PyTorch Lightning wrapper
│   ├── utils/
│   │   ├── config.py            # Model & training config
│   │   ├── data_utils.py        # Data loading & preprocessing
│   │   └── dataset.py           # PyTorch Dataset implementation
│   └── analysis/           # NeuralFactors evaluation metrics
├── PPCA/                   # PPCA baseline model
│   ├── model.py            # Closed-form PPCA
│   ├── loader.py           # Data loading and split indexing
│   ├── evaluate.py         # CLI entry point
│   └── analysis/           # NLL, covariance, VaR, portfolio metrics
├── scripts/
│   ├── train.py            # NeuralFactors training script
│   └── test.py             # NeuralFactors evaluation script
├── results/
│   ├── compare.py          # Cross-model comparison script
│   ├── evaluation/         # NeuralFactors evaluation outputs
│   ├── ppca/               # PPCA evaluation outputs
│   └── comparison/         # Side-by-side comparison tables
├── checkpoints/            # Saved model checkpoints
├── logs/                   # TensorBoard training logs
└── requirements.txt
```

---

## Model Architecture

NeuralFactors is a VAE-based generative model for equity returns. Each trading day, the model encodes N stocks into a shared F-dimensional latent factor space and decodes them through a linear factor structure with Student-T noise.

```
Training:  q(z|r) → z → p(r|z)   [encoder + decoder]
Inference: p(z)   → z → p(r|z)   [prior + decoder]
```

### Stock Embedder (`src/models/stock_embedder.py`)

Encodes per-stock time-series and static features into factor-specific parameters using a Transformer encoder followed by a two-layer MLP.

- **Input**: `S[N, L, d_ts]` (lookback window, L=256), `S_static[N, d_static]`
- **Output**: `alpha[N]`, `beta[N, F]`, `sigma[N]`, `nu[N]` — parameters of the per-stock Student-T likelihood

### Encoder (`src/models/encoder.py`)

Computes the analytical variational posterior q(z|r) via closed-form linear regression (Paper Eq. 8):

```
Σ_q = (Σ_z⁻¹ + Bᵀ Σ_x⁻¹ B)⁻¹
μ_q = Σ_q (Σ_z⁻¹ μ_z + Bᵀ Σ_x⁻¹ (r − α))
```

Used only during training. Numerical stability ensured via FP64 computation and adaptive Cholesky jitter.

### Decoder (`src/models/decoder.py`)

Computes log p(r|z) and samples returns under the linear factor model with Student-T noise (Paper Section 3.2):

```
r_i ~ Student-T(α_i + βᵢᵀ z, σ_i, ν_i)
```

Also provides closed-form marginal mean `E[r] = α + B μ_z` and covariance `Cov[r] = diag(σ²) + B Σ_z Bᵀ` for portfolio optimization without sampling.

### Prior (`src/models/prior.py`)

Learnable time-homogeneous Student-T prior p(z) with constrained parameters (σ > 0, ν > 4):

```
z ~ Student-T(ν_z, μ_z, σ_z)
```

All parameters are learned via gradient descent alongside the rest of the model.

### NeuralFactors (`src/models/neuralfactors.py`)

Main module integrating all components. Computes CIWAE loss (Paper Eq. 7) with K=20 importance samples during training; switches to prior sampling during inference.

---

## Configuration (`src/utils/config.py`)

Centralized hyperparameter configuration with paper defaults:

```python
ModelConfig:      num_factors=64, hidden_size=256, lookback=256, dropout=0.25, nhead=4, num_layers=2
TrainingConfig:   learning_rate=1e-4, weight_decay=1e-6, max_steps=100000, num_iwae_samples=20
                  use_polyak=True, polyak_alpha=0.999, polyak_start_step=50000
PriorConfig:      mu_z_init=0.0, sigma_z_init=1.0, nu_z_init=10.0
EncoderConfig:    jitter_multiplier=2.0, use_fp64=True
```

---

## Data Pipeline

Data is loaded from `data/parquets/` (long format: `date`, `ticker`, feature columns) and `data/cleaned/` (semicolon-delimited CSV, comma decimals). Key steps in `src/utils/data_utils.py`:

1. **`load_parquets()`** — loads time-series features, static features, and closing prices
2. **`compute_returns()`** — computes log returns; `±Inf` from zero/negative prices are replaced with `NaN` for proper masking
3. **`compute_returns_std_from_train()`** — computes return normalization std from the training period (≈0.0627 for IBX vs. ≈0.0267 for S&P 500)
4. **`split_by_date()`** — partitions into train/val/test

`src/utils/dataset.py` implements a PyTorch `Dataset` that yields, for each trading day, lookback tensors `S[N, L, d_ts]`, static features `S_static[N, d_static]`, returns `r[N]`, and a validity mask `mask[N]`.

**Data splits** (adjusted for IBX availability):
- Training: 2005-01-01 – 2018-12-31 (≈3,458 trading days)
- Validation: 2019-01-01 – 2022-12-31 (≈994 trading days)
- Test: 2023-01-01 – 2025-11-04 (≈712 trading days)

---

## Training

Training follows Paper Section 3.5 via PyTorch Lightning (`src/models/lightning_module.py`):

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-4, weight_decay=1e-6 |
| Loss | CIWAE, K=20 importance samples |
| Batch size | 1 (all stocks from one trading day) |
| Total steps | 100,000 |
| Validation frequency | Every 1,000 steps (NLL_joint) |
| Polyak averaging | α=0.999, starts at step 50,000 |
| Gradient clipping | norm=1.0 |

```bash
# Full training
python scripts/train.py --data_dir data --checkpoint_dir checkpoints

# Single-batch smoke test
python scripts/train.py --fast_dev_run
```

Checkpoints are saved to `checkpoints/neuralfactors/`, Polyak weights to `polyak_model.pt`, and TensorBoard logs to `logs/neuralfactors/`.

---

## Evaluation

Implemented in `scripts/test.py`. Four metrics match the paper's evaluation protocol:

| Metric | Description |
|---|---|
| **NLL** | Joint and per-stock negative log-likelihood via IWAE |
| **Covariance MSE** | Predicted vs. 20-day empirical rolling covariance |
| **VaR calibration** | Theoretical vs. empirical violation rates at 1%, 5%, 10% |
| **Portfolio backtest** | Min-variance portfolio: return, vol, Sharpe, max drawdown |

```bash
# Debug mode (first 50 dates, ~5 min)
python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode debug

# Full paper evaluation
python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode paper
```

Results are saved to `results/evaluation/neuralfactors/` (metrics CSVs, time-series, plots, and a human-readable `evaluation_summary.txt`).

---

## PPCA Baseline

Probabilistic PCA serves as a closed-form baseline. The model is a linear Gaussian factor model with isotropic noise:

```
x = W z + μ + ε,   z ~ N(0, I_F),   ε ~ N(0, σ² I_N)
=> x ~ N(μ, W Wᵀ + σ² I)
```

Parameters are fit via closed-form MLE (top-F eigendecomposition of the sample covariance). Log-likelihood is computed using the Woodbury identity, avoiding any N×N matrix inversion. A 252-day rolling window is used for time-varying estimation.

```bash
python PPCA/evaluate.py --mode debug   # first 50 test dates (~1 min)
python PPCA/evaluate.py --mode paper   # all 712 test dates
```

Key parameters: `--num_factors 12`, `--window_size 252`. Same train/val/test splits as NeuralFactors.

---

## Cross-Model Comparison

`results/compare.py` aggregates evaluation outputs into side-by-side tables:

```bash
python results/compare.py \
    --results "NeuralFactors:results/evaluation/neuralfactors" \
    --results "PPCA:results/ppca/ppca"
```

Output (in `results/comparison/`):

| File | Contents |
|---|---|
| `comparison_nll.csv` | NLL mean and std per model |
| `comparison_cov.csv` | Covariance MSE mean and std per model |
| `comparison_var.csv` | VaR error and empirical violation rates |
| `comparison_portfolio.csv` | Return, vol, Sharpe, max drawdown |
| `comparison_formatted.csv` | Paper Table 2 style (one row per model) |

---

## Paper Reference

Gopal, A. (2024). *NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities*. arXiv:2408.01499v1 [q-fin.ST].
