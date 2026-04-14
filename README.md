# NeuralFactors (Simplified)

VAE-based latent factor model for Brazilian equities (B3/IBX).  
Adapted from Gopal (2024, arXiv:2408.01499) with significant architectural simplifications to prevent posterior collapse on the Brazilian market (~95 tickers, 2005–2026).

---

## Architecture

| Component | Original (Gopal 2024) | This implementation |
|-----------|----------------------|---------------------|
| Prior | Learnable Student-T(μ,σ,ν) | Fixed N(0, I) |
| Posterior | Analytical Cholesky (FP64) | DeepSet MLP |
| Likelihood | Student-T | Gaussian |
| Training objective | CIWAE (K=20) | ELBO (K=1) + free bits |

### Why the changes?

The original learnable prior collapsed on the Brazilian market: `sigma_z` decayed from 10.0 → 0.066 in the first 100k steps because the prior parameters shared the Adam update with the rest of the network and optimised faster than the encoder could track. The result was that the posterior collapsed to the (near-point-mass) prior — no factors were used — and out-of-sample Sharpe was 0.08 vs 0.55 for PPCA.

Fix: remove the learnable prior entirely. KL is now `0.5 * sum(sigma_q^2 + mu_q^2 - 1 - 2*log_sigma_q)` in closed form, and the free-bits floor `max(0, lambda - KL_f)` per factor prevents trivial KL=0 solutions.

### Model components

```
StockEmbedder
  Transformer(lookback=256, L=d_ts=76) -> h=256
  alpha_head:  Linear(256 -> 1)   # idiosyncratic return alpha ~ N(0,alpha_max)
  B_head:      Linear(256 -> F)   # factor loadings beta ~ unscaled
  sigma_head:  Linear(256 -> 1)   # idiosyncratic vol in (sigma_min, sigma_max)

DeepSetEncoder
  phi:    MLP(4+F -> 64 -> 64, GELU)  # per-stock, no loops
  pool:   masked mean over stocks
  rho:    MLP(64 -> 128 -> 128, GELU)
  mu_q:   Linear(128 -> F)
  logσ_q: Linear(128 -> F)   # init weight_std=0.01, bias=0 -> sigma_q ~ 1.0

Fixed prior: z ~ N(0, I_F)

Gaussian decoder:
  r_i | z ~ N(alpha_i + beta_i' z, sigma_i^2)
```

### ELBO loss

```
ELBO = E_q[sum_i log N(r_i; alpha_i + beta_i'z, sigma_i^2)]
     - KL(q(z|r) || N(0,I))
     + free_bits_penalty

free_bits_penalty = sum_f max(0, lambda - KL_f)   # lambda=0.1 nats
```

---

## Repository layout

```
src/
  models/
    stock_embedder.py     StockEmbedder: Transformer -> (alpha, B, sigma)
    encoder.py            DeepSetEncoder: (r, alpha, B, sigma) -> (mu_q, log_sigma_q)
    decoder.py            Gaussian decoder utilities (log_pdf, sample, marginal_cov)
    neuralfactors.py      NeuralFactors: encode() + compute_elbo_loss() + predict()
    lightning_module.py   PyTorch Lightning wrapper, Polyak averaging
  utils/
    config.py             ModelConfig, EncoderConfig, TrainingConfig
    dataset.py            NeuralFactorsDataset (parquet loader, normalisation)
    data_utils.py         discover_feature_dims, compute_returns_std_from_train
  analysis/
    analyze.py            Post-training analysis: loss curves, factor distributions
    test/
      loader.py           load_model_and_data()
      nll.py              Gaussian ELBO-based NLL metrics
      covariance.py       Predicted vs empirical rolling covariance
      var.py              VaR calibration (Kupiec + Christoffersen)
      portfolio.py        Min-variance portfolio backtest vs Ibovespa

scripts/
  train.py    Train NeuralFactors from scratch
  test.py     Full evaluation suite (NLL, covariance, VaR, portfolio)
  run.py      Pipeline: train -> evaluate in one command

data/
  parquets/
    x_ts.parquet      [date, ticker, feature_1..feature_76]  daily time-series
    x_static.parquet  [ticker, feature_1..feature_11]        static fundamentals
    prices.parquet    [date, ticker, price]

checkpoints/<experiment>/
  config.json        Serialised ModelConfig + TrainingConfig
  polyak_model.pt    Polyak-averaged weights (used for evaluation)
  last.ckpt          Latest Lightning checkpoint

logs/<experiment>/   TensorBoard event files
results/evaluation/<experiment>/
  metrics/           CSV tables (nll_timeseries.csv, covariance.csv, ...)
  plots/             PNG figures
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Training

```bash
# Standard 250k run, F=16 factors
python scripts/train.py \
    --experiment_name simplified_v1 \
    --num_factors 16 \
    --max_steps 250000 \
    --free_bits_lambda 0.1

# Equivalent via pipeline script (train + evaluate in one command)
python scripts/run.py \
    --experiment_name simplified_v1 \
    --num_factors 16 \
    --max_steps 250000 \
    --free_bits_lambda 0.1
```

Key hyperparameters:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_factors` | 16 | Latent factor dimension F |
| `--hidden_size` | 256 | Transformer hidden dim h |
| `--lookback` | 256 | Days of history fed to transformer |
| `--learning_rate` | 1e-4 | Adam LR (single group, all params) |
| `--max_steps` | 250000 | Gradient updates |
| `--free_bits_lambda` | 0.1 | Min KL nats per factor (0 disables) |
| `--polyak_alpha` | 0.999 | EMA decay; Polyak starts at step//2 |

Training on RTX 3060 (12 GB): ~18–20 hours for 250k steps.

### Monitoring training health

Open TensorBoard to watch for collapse:

```bash
tensorboard --logdir logs/simplified_v1
```

Key metrics:
- `train/sigma_q_mean` — should stay near 1.0; collapse if → 0
- `train/kl_min_factor` — should exceed `free_bits_lambda`; 0 = unused factor
- `train/kl_divergence` — total KL; healthy range ~2–15 nats
- `train/log_likelihood` — should decrease steadily

---

## Evaluation

```bash
python scripts/test.py \
    --checkpoint checkpoints/simplified_v1/polyak_model.pt \
    --experiment_name simplified_v1 \
    --mode paper \
    --num_samples 100
```

Metrics produced:
- **NLL** — joint and per-stock Gaussian log-likelihood
- **Covariance** — predicted vs empirical 20-day rolling covariance MSE
- **VaR** — 1%, 5%, 10% quantile calibration (Kupiec + Christoffersen tests)
- **Portfolio** — minimum-variance backtest vs Ibovespa benchmark (Sharpe, Sortino, max drawdown)

---

## Data

Raw parquet files are not committed to this repository.  
Expected location: `data/parquets/{x_ts,x_static,prices}.parquet`.  
See `data/data_documentation.md` for field descriptions.

Data period: 2005-01-03 to 2026  
Train split: up to 2018-12-31  
Validation split: 2019-01-01 to 2022-12-31  
Test split: 2023-01-01 onwards  
Universe: ~95 IBX tickers (variable over time due to index changes)

---

## PPCA baseline

The `PPCA/` directory provides a probabilistic PCA baseline for comparison.  
After training NeuralFactors, run:

```bash
python results/compare.py --experiment_name simplified_v1
```

This produces side-by-side tables for NLL, covariance MSE, VaR calibration, and portfolio metrics.

---

## Reference

Gopal, A. (2024). *NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities*. arXiv:2408.01499.
