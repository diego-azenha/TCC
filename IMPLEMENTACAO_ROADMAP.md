# Roadmap de Implementação - Script de Testes NeuralFactors

## Resumo Executivo

Este documento apresenta um roteiro prático e sequencial para implementar o script de testes descrito no `PLANO_SCRIPT_TESTES.md`, com foco em entregar valor incremental.

---

## Estratégia de Implementação: Iterativa e Incremental

### Princípios
1. **Começar simples**: Implementar métricas básicas primeiro
2. **Validar cedo**: Testar cada componente com dados reais
3. **Documentar resultados**: Comparar com artigo após cada etapa
4. **Priorizar impacto**: Focar nas métricas mais importantes (NLL, Sharpe)

---

## Fase 0: Preparação (1-2 dias)

### Setup inicial

```bash
# 1. Instalar dependências adicionais
pip install cvxpy scipy statsmodels arch

# 2. Verificar estrutura de dados
python -c "
from src.utils.data_utils import load_parquets
ts, static, prices = load_parquets('data')
print(f'Time-series features: {ts.shape}')
print(f'Static features: {static.shape}')
print(f'Prices: {prices.shape}')
"

# 3. Carregar modelo treinado
python -c "
import torch
from src.models.lightning_module import NeuralFactorsLightning
model = NeuralFactorsLightning.load_from_checkpoint('checkpoints/best.ckpt')
print('Model loaded successfully')
"
```

### Checklist Fase 0
- [ ] Todas dependências instaladas
- [ ] Dados carregam sem erros
- [ ] Modelo carrega corretamente
- [ ] GPU disponível (opcional, mas recomendado)

---

## Fase 1: Métrica Fundamental - NLL (3-5 dias)

### 1.1 Implementar NLLjoint

**Arquivo**: `src/evaluation/metrics.py`

```python
import torch
import numpy as np
from typing import Dict, Tuple

def compute_nll_joint(
    model,
    S: torch.Tensor,          # [N, L, d_ts]
    S_static: torch.Tensor,   # [N, d_static]
    r_true: torch.Tensor,     # [N]
    num_samples: int = 100,
    use_posterior: bool = False
) -> float:
    """
    Computa NLL_joint = -log p(r_t+1 | F_t) / N
    
    Args:
        model: NeuralFactors model
        S: Time-series features
        S_static: Static features  
        r_true: Observed returns
        num_samples: Number of importance samples (100 for evaluation)
        use_posterior: If True, use posterior q(z|r) (training), else prior p(z)
    
    Returns:
        NLL_joint for this day
    """
    model.eval()
    with torch.no_grad():
        if use_posterior:
            # Training/validation: use posterior
            output = model(S, S_static, r_true, num_samples=num_samples)
            # log p(r|F) ≈ log(1/K sum_k w_k) where w_k = p(r,z_k)/q(z_k|r)
            log_likelihood = output['iwae_loss']  # já é negativo
            nll_joint = -log_likelihood
        else:
            # Test: sample from prior
            output = model.predict(S, S_static, num_samples=num_samples)
            # output['samples']: [N, num_samples]
            # Compute p(r_true | z_k) for each sample
            log_probs = []
            for k in range(num_samples):
                z_k = output['z_samples'][k]  # [F]
                log_p_r_given_z = model.decoder.log_pdf_r_given_z(
                    r_true,
                    output['alpha'],
                    output['B'],
                    output['sigma'],
                    output['nu'],
                    z_k.unsqueeze(0)
                )
                log_probs.append(log_p_r_given_z)
            
            # Log-sum-exp trick
            log_probs = torch.stack(log_probs)
            log_p_r = torch.logsumexp(log_probs, dim=0) - np.log(num_samples)
            nll_joint = -log_p_r.mean().item()
    
    return nll_joint


def compute_nll_individual(
    model,
    S: torch.Tensor,
    S_static: torch.Tensor,
    r_true: torch.Tensor,
    num_samples: int = 10000
) -> float:
    """
    Computa NLL_ind = -sum_i log p(r_i,t+1 | F_t) / N
    
    Marginaliza sobre z: p(r_i) = int p(r_i|z) p(z) dz
    Aproxima com muitos samples (10K)
    """
    model.eval()
    N = r_true.shape[0]
    
    with torch.no_grad():
        output = model.predict(S, S_static, num_samples=num_samples)
        
        # Para cada stock i:
        nll_ind_per_stock = []
        for i in range(N):
            log_probs_i = []
            for k in range(num_samples):
                z_k = output['z_samples'][k]
                # p(r_i | z_k)
                log_p_ri_given_zk = model.decoder.log_pdf_r_given_z(
                    r_true[i:i+1],
                    output['alpha'][i:i+1],
                    output['B'][i:i+1],
                    output['sigma'][i:i+1],
                    output['nu'][i:i+1],
                    z_k.unsqueeze(0)
                )
                log_probs_i.append(log_p_ri_given_zk)
            
            # p(r_i) ≈ (1/K) sum_k p(r_i|z_k)
            log_probs_i = torch.stack(log_probs_i)
            log_p_ri = torch.logsumexp(log_probs_i, dim=0) - np.log(num_samples)
            nll_ind_per_stock.append(-log_p_ri.item())
        
        nll_ind = np.mean(nll_ind_per_stock)
    
    return nll_ind
```

### 1.2 Script de teste básico

**Arquivo**: `scripts/test_nll.py`

```python
import torch
import argparse
from tqdm import tqdm
from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset
from src.evaluation.metrics import compute_nll_joint, compute_nll_individual

def evaluate_nll(checkpoint_path, data_dir, split='test'):
    # Load model
    model = NeuralFactorsLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Load dataset
    dataset = NeuralFactorsDataset(
        data_dir=data_dir,
        lookback=256,
        split=split
    )
    
    # Evaluate on each day
    nll_joint_list = []
    nll_ind_list = []
    
    for i in tqdm(range(len(dataset)), desc=f'Evaluating {split}'):
        batch = dataset[i]
        S = batch['S'].unsqueeze(0)
        S_static = batch['S_static'].unsqueeze(0)
        r = batch['r'].unsqueeze(0)
        mask = batch['mask']
        
        # Filter valid stocks
        S = S[:, mask, :, :]
        S_static = S_static[:, mask, :]
        r = r[:, mask]
        
        if S.shape[1] < 10:  # Skip days with too few stocks
            continue
        
        # Compute NLL
        nll_j = compute_nll_joint(model.model, S.squeeze(0), S_static.squeeze(0), r.squeeze(0))
        nll_i = compute_nll_individual(model.model, S.squeeze(0), S_static.squeeze(0), r.squeeze(0))
        
        nll_joint_list.append(nll_j)
        nll_ind_list.append(nll_i)
    
    # Results
    results = {
        'nll_joint': np.mean(nll_joint_list),
        'nll_joint_std': np.std(nll_joint_list),
        'nll_ind': np.mean(nll_ind_list),
        'nll_ind_std': np.std(nll_ind_list),
    }
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data')
    args = parser.parse_args()
    
    print("Evaluating Validation Set...")
    val_results = evaluate_nll(args.checkpoint, args.data_dir, split='val')
    print(f"Validation NLLjoint: {val_results['nll_joint']:.4f}")
    print(f"Validation NLLind: {val_results['nll_ind']:.4f}")
    
    print("\nEvaluating Test Set...")
    test_results = evaluate_nll(args.checkpoint, args.data_dir, split='test')
    print(f"Test NLLjoint: {test_results['nll_joint']:.4f}")
    print(f"Test NLLind: {test_results['nll_ind']:.4f}")
```

### 1.3 Teste e validação

```bash
# Rodar teste
python scripts/test_nll.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --data_dir data

# Esperado (valores aproximados para IBX):
# Validation NLLjoint: 0.35-0.45 (artigo: 0.324 para S&P 500)
# Test NLLjoint: 0.55-0.70 (artigo: 0.556 para S&P 500)
```

### Checklist Fase 1
- [ ] `metrics.py` criado com NLL functions
- [ ] `test_nll.py` rodando sem erros
- [ ] Resultados de Val/Test obtidos
- [ ] Resultados documentados e comparados com artigo

---

## Fase 2: Portfolio Optimization - Sharpe Ratio (3-5 dias)

### 2.1 Implementar otimização de portfólio

**Arquivo**: `src/evaluation/portfolio.py`

```python
import cvxpy as cp
import numpy as np

def mean_variance_optimization(
    mu: np.ndarray,      # [N] expected returns
    Sigma: np.ndarray,   # [N, N] covariance matrix
    strategy: str = 'long_only',
    lambda_risk: float = 1.0
) -> np.ndarray:
    """
    Resolve: argmax_w [mu^T w - (lambda/2) w^T Sigma w]
    
    Strategies:
    - 'long_only': w >= 0, sum(w) = 1
    - 'long_short': w ∈ R, sum(|w|) = 2
    - 'long_only_lev': w >= 0, sum(w) = 2
    - 'long_short_lev': w ∈ R, sum(|w|) = 3
    """
    N = len(mu)
    w = cp.Variable(N)
    
    # Objective
    returns = mu @ w
    risk = cp.quad_form(w, Sigma)
    objective = cp.Maximize(returns - (lambda_risk / 2) * risk)
    
    # Constraints
    constraints = []
    
    if strategy == 'long_only':
        constraints += [w >= 0, cp.sum(w) == 1]
    
    elif strategy == 'long_short':
        constraints += [cp.norm(w, 1) == 2]
    
    elif strategy == 'long_only_lev':
        constraints += [w >= 0, cp.sum(w) == 2]
    
    elif strategy == 'long_short_lev':
        constraints += [cp.norm(w, 1) == 3]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status != 'optimal':
        print(f"Warning: Optimization status = {problem.status}")
        # Fallback to equal weights
        w_opt = np.ones(N) / N
    else:
        w_opt = w.value
    
    return w_opt


def compute_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio anualizado
    
    Sharpe = (mean_return / std_return) * sqrt(periods_per_year)
    """
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0:
        return 0.0
    
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    return sharpe
```

### 2.2 Script de backtesting

**Arquivo**: `scripts/test_portfolio.py`

```python
def backtest_portfolio(model, dataset, strategy='long_only', lambda_risk=1.0):
    """
    Backtest portfolio strategy
    """
    portfolio_returns = []
    
    for i in tqdm(range(len(dataset)-1), desc=f'Backtesting {strategy}'):
        # Get data for day t
        batch_t = dataset[i]
        S_t = batch_t['S']
        S_static_t = batch_t['S_static']
        mask_t = batch_t['mask']
        
        # Filter valid stocks
        S_t = S_t[mask_t]
        S_static_t = S_static_t[mask_t]
        
        if S_t.shape[0] < 10:
            continue
        
        # Predict mean and covariance
        with torch.no_grad():
            output = model.model.encode(
                S_t.unsqueeze(0),
                S_static_t.unsqueeze(0),
                None  # no returns, just predict
            )
            
            # Get mean and covariance analytically
            mu_z, Sigma_z = model.model.prior.to_normal_params()
            
            alpha = output['alpha'].cpu().numpy()
            B = output['B'].cpu().numpy()
            sigma = output['sigma'].cpu().numpy()
            
            # E[r] = alpha + B @ mu_z
            mu_pred = alpha + B @ mu_z.cpu().numpy()
            
            # Cov[r] = diag(sigma^2) + B @ Sigma_z @ B^T
            Sigma_pred = np.diag(sigma**2) + B @ Sigma_z.cpu().numpy() @ B.T
        
        # Optimize portfolio
        w_opt = mean_variance_optimization(mu_pred, Sigma_pred, strategy, lambda_risk)
        
        # Get actual returns at t+1
        batch_t1 = dataset[i+1]
        r_t1 = batch_t1['r'][mask_t].cpu().numpy()
        
        # Compute portfolio return
        r_port = w_opt @ r_t1
        portfolio_returns.append(r_port)
    
    # Compute Sharpe ratio
    sharpe = compute_sharpe_ratio(np.array(portfolio_returns))
    
    return {
        'sharpe': sharpe,
        'mean_return': np.mean(portfolio_returns) * 252,  # annualized
        'volatility': np.std(portfolio_returns) * np.sqrt(252),  # annualized
        'returns': portfolio_returns
    }
```

### 2.3 Rodar para 4 estratégias

```bash
python scripts/test_portfolio.py \
    --checkpoint checkpoints/best.ckpt \
    --strategies long_only,long_short,long_only_lev,long_short_lev

# Output esperado:
# Strategy: long_only
#   Val Sharpe: 1.2-1.8
#   Test Sharpe: 0.8-1.3
# Strategy: long_short
#   Val Sharpe: 2.5-3.5
#   Test Sharpe: 1.8-2.6
# ...
```

### Checklist Fase 2
- [ ] `portfolio.py` criado e testado
- [ ] Backtesting funcionando
- [ ] Sharpe ratios para 4 estratégias (Val + Test)
- [ ] Resultados comparados com artigo (Tabela 7)

---

## Fase 3: Covariance Forecasting (2-3 dias)

### 3.1 Implementar whitening e MSE

**Arquivo**: `src/evaluation/metrics.py` (adicionar)

```python
def compute_covariance_mse(
    model,
    dataset,
    min_days_in_universe: int = 100
) -> Tuple[float, float]:
    """
    Computa MSE e Box's M test para covariance forecasting
    
    Returns:
        (mse, box_m_statistic)
    """
    # 1. Filter stocks that appear consistently
    # (similar to article: 324 stocks in S&P500 during 2014-2023)
    stock_counts = {}  # ticker -> count of days
    for i in range(len(dataset)):
        batch = dataset[i]
        tickers = batch['tickers']  # need to add this to dataset
        for ticker in tickers:
            stock_counts[ticker] = stock_counts.get(ticker, 0) + 1
    
    consistent_stocks = [
        ticker for ticker, count in stock_counts.items()
        if count >= min_days_in_universe
    ]
    
    s = len(consistent_stocks)
    print(f"Using {s} consistent stocks")
    
    # 2. Compute whitened returns for each day
    whitened_returns = []
    
    for i in range(len(dataset)-1):
        batch = dataset[i]
        # ... filter to consistent_stocks only ...
        
        # Predict covariance
        Sigma_pred = ...  # compute as before
        
        # Observe actual returns
        r_t1_actual = ...
        mu_pred = ...
        
        # Whiten: r_rot = Sigma^(-1/2) @ (r - mu)
        L = np.linalg.cholesky(Sigma_pred + 1e-6 * np.eye(s))
        L_inv = np.linalg.inv(L)
        r_rot = L_inv @ (r_t1_actual - mu_pred)
        
        whitened_returns.append(r_rot)
    
    # 3. Compute covariance of whitened returns
    whitened_returns = np.array(whitened_returns)  # [T, s]
    Cov_whitened = np.cov(whitened_returns.T)
    
    # 4. MSE: || Cov_whitened - I ||_F^2 / s^2
    I = np.eye(s)
    mse = np.sum((Cov_whitened - I)**2) / (s**2)
    
    # 5. Box's M test
    box_m = compute_box_m_statistic(Cov_whitened, s, len(whitened_returns))
    
    return mse, box_m


def compute_box_m_statistic(Cov, p, n):
    """
    Box's M test for H0: Cov = I
    
    See: Box (1949)
    """
    # Simplified implementation
    det_Cov = np.linalg.det(Cov)
    trace_Cov = np.trace(Cov)
    
    # M = -2 * log(Lambda) where Lambda is likelihood ratio
    # Approximation: M ≈ n * (log(det(Cov)) + trace(Cov) - p)
    M = n * (np.log(det_Cov) + trace_Cov - p)
    
    return M
```

### Checklist Fase 3
- [ ] Whitening implementado corretamente
- [ ] MSE calculado
- [ ] Box's M implementado (mesmo que aproximado)
- [ ] Comparação com artigo (Tabela 5)

---

## Fase 4: VaR Calibration (2-3 dias)

### 4.1 Implementar calibration error

**Arquivo**: `src/evaluation/metrics.py` (adicionar)

```python
def compute_calibration_error(
    predicted_cdfs: List[callable],  # List of CDF functions
    observed_values: np.ndarray,
    num_quantiles: int = 100
) -> float:
    """
    Calibration error (Kuleshov et al.)
    
    Args:
        predicted_cdfs: List of T CDF functions F_t(r)
        observed_values: [T] observed returns
        num_quantiles: Number of quantiles to evaluate
    
    Returns:
        Calibration error
    """
    quantiles = np.linspace(0.01, 0.99, num_quantiles)
    
    errors = []
    for p_j in quantiles:
        # p_hat_j = fraction of data where F_t(r_t) < p_j
        count = 0
        for i, (F_t, r_t) in enumerate(zip(predicted_cdfs, observed_values)):
            cdf_value = F_t(r_t)
            if cdf_value < p_j:
                count += 1
        
        p_hat_j = count / len(observed_values)
        errors.append((p_j - p_hat_j)**2)
    
    calibration_error = np.mean(errors)
    return calibration_error


def get_cdf_function(model, S, S_static, num_samples=10000):
    """
    Returns a CDF function F(r) for a given day
    
    F(r) = P(r_t+1 <= r | F_t) ≈ (1/K) sum_k I(r_k <= r)
    """
    with torch.no_grad():
        output = model.predict(S, S_static, num_samples=num_samples)
        samples = output['samples']  # [N, K] samples
    
    def cdf(r_value):
        # For each stock i, compute P(r_i <= r_value)
        # Average over stocks for portfolio case
        probs = (samples <= r_value).float().mean(dim=1).mean()
        return probs.item()
    
    return cdf
```

### Checklist Fase 4
- [ ] CDF sampling implementado
- [ ] Calibration error para ações individuais
- [ ] Calibration error para portfólio
- [ ] Comparação com artigo (Tabela 6)

---

## Fase 5: Baselines - PPCA (4-5 dias)

### 5.1 Implementar PPCA

**Arquivo**: `src/evaluation/baselines.py`

```python
from sklearn.decomposition import FactorAnalysis
import scipy.stats as stats

class PPCAModel:
    """
    Probabilistic PCA with Student-T decoder
    """
    def __init__(self, n_factors=12, nu=10.0):
        self.n_factors = n_factors
        self.nu = nu
        self.fa = None
        self.sigma_noise = None
    
    def fit(self, returns):
        """
        Fit PPCA on historical returns
        
        Args:
            returns: [T, N] returns matrix
        """
        # Standard Factor Analysis (Gaussian)
        self.fa = FactorAnalysis(n_components=self.n_factors)
        self.fa.fit(returns)
        
        # Estimate noise variance for Student-T
        # (simplified, assumes i.i.d. noise)
        residuals = returns - self.fa.transform(returns) @ self.fa.components_
        self.sigma_noise = np.std(residuals, axis=0)
    
    def predict(self, lookback_returns, num_samples=100):
        """
        Predict returns given lookback
        
        Returns:
            samples: [N, num_samples]
        """
        # In PPCA, factors are just z ~ N(0, I)
        z_samples = np.random.randn(num_samples, self.n_factors)
        
        # r = B @ z + epsilon
        # B = loadings
        B = self.fa.components_.T  # [N, F]
        
        samples = []
        for k in range(num_samples):
            z_k = z_samples[k]
            mean_k = B @ z_k
            
            # Sample Student-T noise
            epsilon_k = stats.t.rvs(
                df=self.nu,
                loc=0,
                scale=self.sigma_noise,
                size=len(self.sigma_noise)
            )
            
            r_k = mean_k + epsilon_k
            samples.append(r_k)
        
        return np.array(samples).T  # [N, num_samples]
    
    def get_mean_cov(self):
        """
        Get analytical mean and covariance
        """
        B = self.fa.components_.T
        mu = np.zeros(B.shape[0])  # zero mean
        
        # Cov = B @ B^T + diag(sigma^2)
        # Adjust for Student-T: sigma^2 -> sigma^2 * nu/(nu-2)
        scale_factor = self.nu / (self.nu - 2)
        Sigma = B @ B.T + np.diag(self.sigma_noise**2 * scale_factor)
        
        return mu, Sigma
```

### 5.2 Integrar PPCA no teste

```python
# Treinar PPCA nos dados de treinamento
ppca = PPCAModel(n_factors=12)
ppca.fit(train_returns)

# Avaliar nos dados de teste (mesmas métricas)
ppca_nll_joint = evaluate_nll_baseline(ppca, test_dataset)
ppca_sharpe = backtest_portfolio_baseline(ppca, test_dataset)
# etc.
```

### Checklist Fase 5
- [ ] PPCA implementado e treinado
- [ ] NLL computado para PPCA
- [ ] Sharpe ratio para PPCA
- [ ] Comparação NeuralFactors vs PPCA

---

## Fase 6: Baselines - GARCH (2-3 dias)

### 6.1 Integrar biblioteca `arch`

```python
from arch import arch_model

def fit_garch_per_stock(returns_series):
    """
    Fit GARCH(1,1) with Skew Student-T
    """
    model = arch_model(
        returns_series,
        vol='GARCH',
        p=1,
        q=1,
        dist='skewt'
    )
    result = model.fit(disp='off')
    return result

def evaluate_garch(returns_matrix, test_indices):
    """
    Evaluate GARCH on test set (NLLind only)
    """
    nll_ind_list = []
    
    for stock_idx in range(returns_matrix.shape[1]):
        returns_stock = returns_matrix[:, stock_idx]
        
        # Fit on training data
        train_returns = returns_stock[:test_indices[0]]
        garch_model = fit_garch_per_stock(train_returns)
        
        # Forecast on test data
        for t in test_indices:
            forecast = garch_model.forecast(horizon=1, start=t)
            mean_t = forecast.mean.iloc[-1, 0]
            var_t = forecast.variance.iloc[-1, 0]
            
            # Compute NLL
            r_actual = returns_stock[t]
            nll = -stats.t.logpdf(
                r_actual,
                df=garch_model.params['nu'],
                loc=mean_t,
                scale=np.sqrt(var_t)
            )
            nll_ind_list.append(nll)
    
    return np.mean(nll_ind_list)
```

### Checklist Fase 6
- [ ] GARCH integrado com `arch`
- [ ] NLLind computado
- [ ] VaR calibration error computado
- [ ] Comparação com artigo

---

## Fase 7: Relatórios e Visualizações (2-3 dias)

### 7.1 Gerar tabelas formatadas

**Arquivo**: `src/evaluation/reporting.py`

```python
def generate_table_4(results_dict):
    """
    Generate Table 4: NLL comparison
    
    Args:
        results_dict: {
            'neuralfactors': {'val_nll_joint': ..., 'test_nll_joint': ..., ...},
            'ppca': {...},
            'garch': {...}
        }
    """
    table = []
    table.append("| Modelo | Val NLLind | Val NLLjoint | Test NLLind | Test NLLjoint |")
    table.append("|--------|------------|--------------|-------------|---------------|")
    
    for model_name, results in results_dict.items():
        row = f"| {model_name:20} | "
        row += f"{results['val_nll_ind']:6.3f} | "
        row += f"{results['val_nll_joint']:6.3f} | "
        row += f"{results['test_nll_ind']:6.3f} | "
        row += f"{results['test_nll_joint']:6.3f} |"
        table.append(row)
    
    return "\n".join(table)
```

### 7.2 Gerar plots

```python
import matplotlib.pyplot as plt

def plot_sharpe_comparison(results_dict, output_path):
    """
    Bar plot comparing Sharpe ratios
    """
    models = list(results_dict.keys())
    strategies = ['L', 'L/S', 'L Lev.1', 'L/S Lev.1']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation
    for i, model in enumerate(models):
        sharpes_val = [results_dict[model][f'val_sharpe_{s}'] for s in strategies]
        x = np.arange(len(strategies)) + i * 0.2
        ax1.bar(x, sharpes_val, width=0.2, label=model)
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Validation Set')
    ax1.set_xticks(np.arange(len(strategies)) + 0.2)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    
    # Test (similar)
    # ...
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
```

### Checklist Fase 7
- [ ] Tabelas 3-7 geradas em Markdown
- [ ] Plots de comparação criados
- [ ] Relatório completo (`report.md`)
- [ ] JSON com todos resultados

---

## Fase 8 (Opcional): Estudos de Ablação (3-4 semanas)

Esta fase requer retreinar o modelo múltiplas vezes. **Muito intensivo em compute.**

**Recomendação**: Fazer apenas se houver recursos computacionais disponíveis e após validar Fases 1-7.

---

## Cronograma Realista

| Fase | Tempo Estimado | Prioridade | Depende de |
|------|----------------|------------|------------|
| 0. Preparação | 1-2 dias | Alta | - |
| 1. NLL | 3-5 dias | **Crítica** | Fase 0 |
| 2. Portfolio/Sharpe | 3-5 dias | **Crítica** | Fase 0 |
| 3. Covariance | 2-3 dias | Alta | Fase 1 |
| 4. VaR | 2-3 dias | Média | Fase 1 |
| 5. PPCA Baseline | 4-5 dias | Alta | Fases 1-4 |
| 6. GARCH Baseline | 2-3 dias | Média | Fase 1 |
| 7. Relatórios | 2-3 dias | Alta | Fases 1-6 |
| 8. Ablação | 3-4 semanas | Baixa | Todas |

**Tempo total (sem ablação)**: 3-4 semanas  
**Tempo total (com ablação)**: 7-8 semanas

---

## Deliverables Mínimos (MVP)

Para ter um script de testes funcional e útil, **foco mínimo**:

1. ✅ **NLL (Tabela 4)**: NeuralFactors vs PPCA
2. ✅ **Sharpe Ratio (Tabela 7)**: 4 estratégias
3. ✅ **Relatório Markdown**: Resultados formatados

**Tempo estimado para MVP**: 2-3 semanas

---

## Próximos Passos Imediatos

1. ✅ **Revisar este roadmap** com equipe/orientador
2. **Começar Fase 0**: Setup e validação de ambiente
3. **Implementar Fase 1**: NLL (métrica mais importante do artigo)
4. **Validar incrementalmente**: Comparar cada resultado com artigo
5. **Iterar e expandir**: Adicionar métricas conforme necessidade

---

**Documento criado em**: 2026-02-12  
**Status**: 🚀 Pronto para começar implementação  
**Recomendação**: Começar com MVP (Fases 0-2) para validar modelo atual
