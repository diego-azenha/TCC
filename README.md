# NeuralFactors: Replication Study

Replication and adaptation of "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities" by Achintya Gopal (arXiv:2408.01499v1), applied to Brazilian IBX equities (2005–2025).

**Thesis context**: This repository accompanies the undergraduate thesis submitted to [institution]. It implements NeuralFactors from scratch in PyTorch, trains it on ~100 IBX constituents, and benchmarks it against a Probabilistic PCA (PPCA) baseline across four quantitative metrics: NLL, covariance MSE, VaR calibration, and min-variance portfolio performance.

## Repository Structure

```
TCC/
├── data/
│   ├── ibovespa.csv        # Ibovespa benchmark daily returns
│   ├── parquets/           # Parquet format for efficient loading
│   └── data_documentation.md
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

## Resumo dos Dados

### Fontes de Dados Brutos

| Fonte | Conteúdo | Período | Formato |
|---|---|---|---|
| **Economatica** — `fechamento.csv` | Preço de fechamento ajustado por proventos | 2004-12 → 2026-03 | Wide CSV (~1.419 tickers) |
| **Economatica** — `preco_valor_patrimonial.csv` | P/VPA diário | 2004-12 → 2026-03 | Wide CSV |
| **Economatica** — trimestrais (5 arquivos) | ROA, ROE, Margem Bruta, Dívida Bruta/Ativo, Dívida Líq./PL (TTM) | 1986 → 2025 | Wide CSV, grade diária |
| **Bloomberg** — `bloomberg_indices_values.xlsx` | 29 séries de índices de mercado (risco, câmbio, renda fixa, commodities, equity global) | 2005-01-03 → 2026-03-26 | Excel (5 sheets) |
| **Economatica** — `setor_ibovespa.xlsx` | Classificação setorial de ~1.420 tickers (3 níveis; pipeline usa `setor_economico`) | — | Excel |

### Estatísticas do Dataset

| Item | Valor |
|---|---|
| **Tickers únicos** (universo total) | 956 |
| **Tickers no período de treino** | 841 |
| **Dias úteis totais** (`x_ts`) | 5.259 |
| **Observações (ticker × dia)** | ~1,74 milhão |
| **Dimensão temporal** (`d_ts`) | 38 features |
| **Dimensão estática** (`d_static`) | 11 setores (one-hot) |
| **Período coberto** | 2005-01-04 → 2026-03-26 |

### Features Utilizadas

**Série temporal por ativo — `x_ts.parquet`** (`d_ts = 38`):

| Grupo | Features | Qtd. |
|---|---|---|
| Retorno do ativo | Log-return diário normalizado ($r / \sigma_{train}$) | 1 |
| Fundamentais contábeis | ROA, ROE, Margem Bruta, Dívida Bruta/Ativo, Dívida Líq./PL, P/VPA, EV/EBITDA, P/L | 8 |
| Risco & sentimento | VIX Index, MOVE Index, Brazil CDS 5Y | 3 |
| Brasil macro & câmbio | DI Over (BZDIOVRA), USD/BRL | 2 |
| Brasil equity factors | MXBRSC, MXBRLC, MXBR000V, IDIV, MLCXBV, MU702608 | 6 |
| Renda fixa | BZRFIMAB, BZRFIMA, SPUHYBDT | 3 |
| Commodities | BCOMAGTR, BCOMGCTR, BCOMINTR, BCOMNGTR, BCOMSITR, BCOMCOT | 6 |
| MSCI internacionais | MXEF, MXCN, MXJP, MXGB, MXCA, MXEU, MXLA, MXPCJ, MXUS | 9 |

**Features estáticas por ativo — `x_static.parquet`** (`d_static = 11`): one-hot do setor econômico (Bens Industriais, Comunicações, Consumo Cíclico, Consumo Não Cíclico, Financeiro, Materiais Básicos, Outros, Petróleo Gás e Biocombustíveis, Saúde, Tecnologia da Informação, Utilidade Pública).

### Tratamento dos Dados

O pipeline é dividido em quatro camadas sequenciais:

```
Camada 0 — Raw          Camada 1 — Clean        Camada 2 — Features      Camada 3 — Model-Ready
raw/ (CSV, XLSX)  ──▶  cleaned/ (Parquet)  ──▶  features/ (Parquet)  ──▶  parquets/ (Parquet)
```

**Camada 1 — Limpeza (`cleaned/`)**:
- **Preços**: remoção de preços ≤ 0; universo definido por pares `(date, ticker)` com preço válido (sem dependência de composição histórica de índice).
- **Fundamentais trimestrais**: filtragem pelo universo de preços; **winsorização nos percentis 1% e 99%** para conter outliers extremos.
- **Índices Bloomberg**: **interpolação linear** para gaps de até 3 dias úteis; gaps maiores permanecem `NaN`.
- **Setores**: mapeamento de setor `"-"` para `"Outros"`; deduplicação por ticker.

**Camada 2 — Feature Engineering (`features/`)**:
- **Log-returns**: $r_{i,t} = \ln(P_{i,t}/P_{i,t-1})$; `±Inf` (preço zero ou negativo) substituídos por `NaN`.
- **Fundamentais**: `ffill()` por ticker após merge com calendário diário. **Sem `bfill()`** — evita look-ahead bias. Valores trimestrais propagados ~60 dias úteis até o próximo reporte.
- **Retornos de índices Bloomberg**: $r^{idx}_t = \ln(I_t/I_{t-1})$; sufixo `_ret` adicionado a cada série.

**Camada 3 — Normalização (`parquets/`)**:

| Feature | Normalização | Estatística de Referência (treino) |
|---|---|---|
| Retorno do ativo | Divisão por $\sigma_{train}$ (sem subtrair média) | $\sigma_{train} = 0{,}0545$ |
| Fundamentais (8) | Z-score global: $(f - \mu_{f,train})/\sigma_{f,train}$ | Pooled across tickers × datas no treino |
| Índices Bloomberg (29) | Z-score por série: $(r^{idx} - \mu^{idx}_{train})/\sigma^{idx}_{train}$ | Calculado individualmente por série |

> **Regra cardinal**: todas as estatísticas de normalização ($\mu$, $\sigma$) são computadas **exclusivamente no período de treino** — nunca em validação ou teste. `NaN` residuais após normalização são preenchidos com `0.0` (= média na escala normalizada).

### Divisão Temporal

| Split | Período | Dias úteis | Tickers |
|---|---|---|---|
| **Treino** | 2005-01-04 → 2018-12-31 | 3.458 | 841 |
| **Validação** | 2019-01-01 → 2022-12-31 | 994 | 556 |
| **Teste** | 2023-01-01 → 2026-03-26 | 807 | 489 |

> A documentação detalhada do pipeline de dados está em [`data/data_documentation.md`](data/data_documentation.md).

---

## Data Pipeline

Data is loaded from `data/parquets/` (long format: `date`, `ticker`, feature columns). Key steps in `src/utils/data_utils.py`:

1. **`load_parquets()`** — loads time-series features, static features, and closing prices
2. **`compute_returns()`** — computes log returns; `±Inf` from zero/negative prices are replaced with `NaN` for proper masking
3. **`compute_returns_std_from_train()`** — computes return normalization std from the training period (≈0.0545 for IBX)
4. **`split_by_date()`** — partitions into train/val/test

`src/utils/dataset.py` implements a PyTorch `Dataset` that yields, for each trading day, lookback tensors `S[N, L, d_ts]`, static features `S_static[N, d_static]`, returns `r[N]`, and a validity mask `mask[N]`.

**Data splits** (adjusted for IBX availability):
- Training: 2005-01-04 – 2018-12-31 (3,458 trading days)
- Validation: 2019-01-01 – 2022-12-31 (994 trading days)
- Test: 2023-01-01 – 2026-03-26 (807 trading days)

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
| **VaR calibration** | Theoretical vs. empirical violation rates at 1%, 5%, 10%; Kupiec (1995) POF test and Christoffersen (1998) conditional coverage test |
| **Portfolio backtest** | Min-variance portfolio: return (CAGR), vol, Sharpe, Sortino, Calmar, max drawdown, turnover, concentration, net-of-TC Sharpe |

```bash
# Debug mode (first 50 dates, ~5 min)
python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode debug

# Full paper evaluation
python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode paper
```

Results are saved to `results/evaluation/neuralfactors/` (metrics CSVs, time-series, plots, and a human-readable `evaluation_summary.txt`).

### Portfolio Backtest Details

The portfolio backtest computes daily min-variance weights from the model-predicted covariance matrix, then realises returns on the next trading day using only tickers common to both days.

**Metrics reported:**
- **Performance**: Total return, annualised return (geometric CAGR), annualised volatility, Sharpe ratio (Rf=0), Sortino ratio, Calmar ratio, max drawdown
- **Portfolio characteristics**: Average daily turnover, annualised turnover, average max weight, average effective N (1/HHI)
- **After transaction costs**: Net return and Sharpe assuming 10 bps proportional cost per one-way turnover
- **Benchmark comparison**: Ibovespa total return, annualised return, Sharpe, excess return, information ratio

### VaR Statistical Tests

Beyond simple violation-rate error, VaR calibration includes:
- **Kupiec (1995)**: Likelihood-ratio test for whether the empirical violation rate equals the theoretical level
- **Christoffersen (1998)**: Tests that violations are independent (no clustering), important for risk management

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
| `comparison_var.csv` | VaR error, empirical violation rates, Kupiec/Christoffersen p-values |
| `comparison_portfolio.csv` | Return, vol, Sharpe, Sortino, Calmar, turnover, max drawdown, net Sharpe |
| `comparison_formatted.csv` | Paper Table 2 style (one row per model) |

---

## Paper Reference

Gopal, A. (2024). *NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities*. arXiv:2408.01499v1 [q-fin.ST].
