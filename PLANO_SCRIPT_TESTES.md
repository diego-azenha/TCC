# Plano de Script de Testes - NeuralFactors
## Baseado no Artigo: "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities"

---

## 1. Visão Geral

Este documento apresenta o planejamento detalhado de um script de testes que irá avaliar o modelo NeuralFactors implementado neste repositório, seguindo exatamente o formato de apresentação de resultados do artigo original (Achintya Gopal, arXiv:2408.01499v1).

### 1.1 Objetivos

- Reproduzir as métricas de avaliação apresentadas nas **Tabelas 3-7** do artigo
- Comparar o desempenho do modelo contra baselines clássicos (PPCA, GARCH)
- Gerar relatórios formatados que permitam comparação direta com os resultados do paper
- Validar a implementação do modelo NeuralFactors com dados do IBX brasileiro

---

## 2. Estrutura de Dados

### 2.1 Conjuntos de Dados (Data Splits)

**Adaptação para dados brasileiros (IBX):**
- **Treinamento**: 2005-01-01 a 2018-12-31 (14 anos)
- **Validação**: 2019-01-01 a 2022-12-31 (4 anos)
- **Teste**: 2023-01-01 a 2025-11-04 (3 anos)

**Diferença do artigo original:**
- Artigo usou S&P 500: Train (1996-2013), Val (2014-2018), Test (2019-2023)
- Nossa implementação adaptada para disponibilidade de dados do IBX

### 2.2 Normalização

- **Retornos normalizados**: Dividir pela std do período de treinamento
- **Artigo**: std ≈ 0.02672 (S&P 500)
- **IBX**: std ≈ 0.0627 (a ser confirmado)
- **Importante**: Documentar a std usada para reprodutibilidade

---

## 3. Métricas de Avaliação (Tabelas do Artigo)

### 3.1 Tabela 3: Estudos de Ablação (Validation Set)

**Métrica principal**: NLLjoint no conjunto de validação

#### Experimentos planejados:

**A. Número de Fatores (F)**
```
- F = 8:   NLLjoint = ?
- F = 16:  NLLjoint = ?
- F = 32:  NLLjoint = ?
- F = 64:  NLLjoint = ? (default, esperado ~0.324)
- F = 128: NLLjoint = ?
```

**B. Features (Características)**
```
- Todas features:                    NLLjoint = ? (baseline)
- Sem options e volume:              NLLjoint = ?
- Apenas retornos, financials, ind:  NLLjoint = ?
- Apenas retornos:                   NLLjoint = ?
```

**C. Arquitetura**
```
- Attention (default):  NLLjoint = ?
- LSTM:                 NLLjoint = ?
- α_i,t = 0:            NLLjoint = ?
- Prior Gaussiano:      NLLjoint = ?
- Decoder Gaussiano:    NLLjoint = ?
```

**D. Loss Function**
```
- IWAE k=20 (default): NLLjoint = ?
- VAE (IWAE k=1):      NLLjoint = ?
```

**E. Lookback Size (L)**
```
- L = 128: NLLjoint = ?
- L = 192: NLLjoint = ?
- L = 256: NLLjoint = ? (default)
```

**F. Anos de Treinamento**
```
- Últimos 18 anos: NLLjoint = ? (full)
- Últimos 15 anos: NLLjoint = ?
- Últimos 10 anos: NLLjoint = ?
- Últimos 5 anos:  NLLjoint = ?
```

---

### 3.2 Tabela 4: Negative Log-Likelihoods (NLL)

**Métricas:**
- **NLLjoint**: log p({r_i,t+1}|F_t) - distribuição conjunta
- **NLLind**: média de log p(r_i,t+1|F_t) - distribuições individuais

**Formato da tabela:**

```
| Modelo                      | Val NLLind | Val NLLjoint | Test NLLind | Test NLLjoint |
|----------------------------|------------|--------------|-------------|---------------|
| NeuralFactors-Attention    | ?          | ?            | ?           | ?             |
| NeuralFactors-LSTM         | ?          | ?            | ?           | ?             |
| PPCA (12 Factors)          | ?          | ?            | ?           | ?             |
| GARCH Skew Student         | ?          | N/A          | ?           | N/A           |
```

**Implementação:**

1. **NLLjoint (Equação 10 do artigo)**:
   ```python
   NLL_joint_t = -log p(r_t+1 | F_t) / N_t+1
   # Aproximar com 100 samples do posterior (treino) ou prior (teste)
   ```

2. **NLLind (Equação 11 do artigo)**:
   ```python
   NLL_ind_t = -sum_i log p(r_i,t+1 | F_t) / N_t+1
   # Aproximar com 10,000 samples do prior
   ```

3. **Procedimento**:
   - Para cada dia t no período de avaliação:
     - Carregar features F_t (lookback de 256 dias)
     - Observar retornos reais r_t+1
     - Calcular NLLjoint_t e NLLind_t
   - Reportar média sobre todos os dias do período

---

### 3.3 Tabela 5: Covariance Forecasting

**Objetivo**: Avaliar qualidade das previsões de covariância

**Métricas:**
1. **MSE (Mean Squared Error)**: Erro quadrático médio da covariância
2. **Box's M Test**: Teste estatístico de igualdade de matrizes de covariância

**Procedimento (Seção 5.2.2 do artigo):**

1. **Filtrar ações consistentes**:
   - Usar apenas ações que estão no IBX durante todo período 2019-2025
   - Número esperado: s ≈ 50-100 ações (vs. 324 no artigo com S&P 500)

2. **Whitening dos retornos**:
   ```python
   # Para cada dia t:
   r_rot_t+1 = Sigma_t^(-1/2) @ (r_t+1 - alpha_t)
   
   # Onde:
   # alpha_t = vetor de médias previstas
   # Sigma_t = matriz de covariância prevista
   ```

3. **Calcular MSE**:
   ```python
   MSE = mean_t(|| Cov(r_rot_t+1) - I ||^2_F)
   # Se modelo perfeito: Cov(r_rot) = Identidade
   ```

4. **Box's M Test**:
   - Estatística de teste para H0: Cov(r_rot) = I
   - Referência: Box (1949) [3]

**Formato da tabela:**

```
| Modelo                  | Val MSE | Val Box's M | Test MSE | Test Box's M |
|------------------------|---------|-------------|----------|--------------|
| NeuralFactors-Attention | ?       | ?           | ?        | ?            |
| NeuralFactors-LSTM      | ?       | ?           | ?        | ?            |
| PPCA (12 Factors)       | ?       | ?           | ?        | ?            |
```

---

### 3.4 Tabela 6: Value at Risk (VaR) Analysis

**Objetivo**: Avaliar calibração das distribuições preditivas

**Métrica Principal**: Calibration Error (Kuleshov et al. [18])

**Fórmula (Equação 12 do artigo):**
```python
# Para cada quantil p_j (j=1,...,100):
p_hat_j = #{y_n | F_x_n(y_n) < p_j} / N

calibration_error = sum_j (p_j - p_hat_j)^2

# Onde:
# F_x_n = CDF prevista dado x_n
# p_hat_j = fração observada abaixo do quantil p_j
# Métrica ideal: calibration_error = 0
```

**Duas variantes**:
1. **Uni.** (Univariate): Média ponderada do erro por ação
2. **Port.** (Portfolio): Erro de um portfólio equiponderado

**Formato da tabela:**

```
| Modelo                      | Val Uni. | Val Port. | Test Uni. | Test Port. |
|----------------------------|----------|-----------|-----------|------------|
| NeuralFactors-Attention    | ?        | ?         | ?         | ?          |
| NeuralFactors-LSTM         | ?        | ?         | ?         | ?          |
| PPCA (12 Factors)          | ?        | ?         | ?         | ?          |
| GARCH Skew Student         | ?        | ?         | ?         | ?          |
```

**Implementação:**

1. **Para cada ação individual**:
   ```python
   # Para cada dia t:
   - Prever CDF: F_t(r) para r ∈ R
   - Observar retorno real: r_t+1
   - Computar quantil observado: p_obs = F_t(r_t+1)
   
   # Agregar:
   - Computar histograma de p_obs (esperado: uniforme [0,1])
   - Calcular calibration error
   ```

2. **Para portfólio equiponderado**:
   ```python
   # Para cada dia t:
   w_t = [1/N_t, 1/N_t, ..., 1/N_t]  # pesos iguais
   r_port_t+1 = w_t @ r_t+1           # retorno observado
   
   # Prever distribuição de r_port via:
   # - Sampling: z ~ p(z), r_port ~ sum_i w_i * r_i|z
   # - Ou analítico: E[r_port] = w @ alpha, Var[r_port] = w @ Sigma @ w
   
   F_port_t(r_port_t+1) = ?  # CDF portfólio
   # Calcular calibration error
   ```

---

### 3.5 Tabela 7: Portfolio Optimization

**Objetivo**: Avaliar desempenho em otimização de portfólio

**Métrica**: Sharpe Ratio (anualizado)

**Estratégias (Equação 13 do artigo):**

```python
# Otimização média-variância:
w* = argmax_w [ E[w @ r_t+1] - (lambda/2) Var[w @ r_t+1] ]
         s.t. ||w||_1 = L

# Onde:
# L = leverage (1 = sem leverage, >1 = com leverage)
# lambda = aversão a risco
```

**Quatro estratégias testadas:**
1. **L (Long-Only)**: w ≥ 0, L = sum(w) = 1
2. **L/S (Long-Short)**: w ∈ R, L = sum(|w|) = 2 (leverage 2x)
3. **L Lev. 1 (Long-Only + Leverage)**: w ≥ 0, L = 2
4. **L/S Lev. 1 (Long-Short + Leverage)**: L = 3

**Formato da tabela:**

```
| Modelo                  | Val L | Val L/S | Val L Lev.1 | Val L/S Lev.1 | Test L | Test L/S | Test L Lev.1 | Test L/S Lev.1 |
|------------------------|-------|---------|-------------|---------------|--------|----------|--------------|----------------|
| NeuralFactors-Attention | ?     | ?       | ?           | ?             | ?      | ?        | ?            | ?              |
| NeuralFactors-LSTM      | ?     | ?       | ?           | ?             | ?      | ?        | ?            | ?              |
| PPCA (12 Factors)       | ?     | ?       | ?           | ?             | ?      | ?        | ?            | ?              |
```

**Implementação:**

1. **Para cada dia t**:
   ```python
   # Prever mean e covariance:
   mu_t = alpha_t + B_t @ mu_z
   Sigma_t = diag(sigma_t^2) + B_t @ Sigma_z @ B_t^T
   
   # Resolver otimização (4 variantes):
   w_t = solve_portfolio_optimization(mu_t, Sigma_t, strategy)
   
   # Observar retorno realizado:
   r_port_t+1 = w_t @ r_t+1
   ```

2. **Calcular Sharpe Ratio** (anualizado):
   ```python
   # Agregar retornos diários do portfólio:
   returns_portfolio = [r_port_1, r_port_2, ..., r_port_T]
   
   # Sharpe anualizado (assumindo ~252 dias úteis):
   sharpe = (mean(returns_portfolio) * 252) / (std(returns_portfolio) * sqrt(252))
   sharpe = mean(returns_portfolio) / std(returns_portfolio) * sqrt(252)
   ```

3. **Detalhe importante**: Artigo usa mean-variance optimization com λ não especificado
   - Testar valores típicos: λ ∈ {1, 2, 5, 10}
   - Ou: λ calibrado para atingir target volatility (~10% anual)

---

## 4. Baselines de Comparação

### 4.1 PPCA (Probabilistic PCA)

**Implementação necessária:**

1. **Modelo**: Factor analysis com prior Gaussiano
   ```python
   # Modelo:
   z ~ N(0, I_F)           # F fatores latentes
   r = B @ z + epsilon     # exposições lineares
   epsilon ~ Student-T(0, sigma, nu) # ruído idiossincrático
   
   # Estimação:
   # - B: via PCA nos retornos históricos
   # - sigma, nu: via MLE
   ```

2. **Hiperparâmetro**: Número de fatores F
   - Artigo usa F=12 (selecionado via validação)
   - Nossa implementação: testar F ∈ {8, 12, 16, 24}

3. **Diferença vs. NeuralFactors**:
   - PPCA: exposições B fixas (função apenas de retornos passados)
   - NeuralFactors: exposições B(F_t) time-varying (função de todas features)

**Bibliotecas sugeridas:**
- `scikit-learn.decomposition.FactorAnalysis`
- Adaptar para Student-T likelihood (não trivial)

---

### 4.2 GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)

**Implementação necessária:**

1. **Modelo**: GARCH(1,1) com Skew Student-T
   ```python
   # Para cada ação i independentemente:
   r_i,t = mu_i + sqrt(h_i,t) * epsilon_i,t
   h_i,t = omega_i + alpha_i * (r_i,t-1 - mu_i)^2 + beta_i * h_i,t-1
   epsilon_i,t ~ Skew-Student-T(nu_i, lambda_i)
   ```

2. **Uso**:
   - Apenas para NLLind (distribuições marginais)
   - Não modela correlações → não aplicável a NLLjoint, covariance, portfolio

3. **Limitação importante (Artigo Seção 5.2.3)**:
   - GARCH melhor calibração de VaR que NeuralFactors
   - Mas GARCH não prevê covariância → não serve para portfolio optimization

**Bibliotecas sugeridas:**
- `arch` (Python): `arch.univariate.GARCH`
- Suporta Student-T e Skew-Student-T

---

## 5. Estrutura de Código Proposta

### 5.1 Módulos de Avaliação

```
src/evaluation/
├── __init__.py
├── metrics.py              # Implementação das métricas
│   ├── nll_joint()
│   ├── nll_individual()
│   ├── covariance_mse()
│   ├── box_m_test()
│   ├── calibration_error()
│   └── sharpe_ratio()
│
├── baselines.py            # Modelos baseline
│   ├── PPCAModel
│   └── GARCHModel
│
├── portfolio.py            # Otimização de portfólio
│   ├── mean_variance_optimization()
│   ├── long_only_strategy()
│   ├── long_short_strategy()
│   └── leveraged_strategy()
│
└── reporting.py            # Geração de tabelas e plots
    ├── generate_table_3()  # Ablation studies
    ├── generate_table_4()  # NLL comparison
    ├── generate_table_5()  # Covariance forecasting
    ├── generate_table_6()  # VaR analysis
    └── generate_table_7()  # Portfolio optimization
```

### 5.2 Script Principal de Teste

```
scripts/test.py
├── Argumentos CLI:
│   ├── --checkpoint: caminho do modelo treinado
│   ├── --data_dir: diretório dos dados
│   ├── --output_dir: onde salvar resultados
│   ├── --metrics: quais métricas calcular (all, nll, cov, var, portfolio)
│   ├── --baselines: quais baselines comparar (ppca, garch, all)
│   └── --ablation: se deve rodar estudos de ablação
│
├── Fluxo:
│   1. Carregar modelo treinado
│   2. Carregar dados de validação/teste
│   3. Calcular métricas principais (Tabela 4)
│   4. Se --baselines: rodar PPCA e GARCH
│   5. Se --ablation: rodar experimentos da Tabela 3
│   6. Calcular métricas adicionais (Tabelas 5-7)
│   7. Gerar relatório formatado (Markdown + LaTeX)
│   8. Salvar resultados em CSV/JSON
│
└── Output:
    ├── results/
    │   ├── test_results.json           # Todas métricas
    │   ├── table_3_ablation.csv
    │   ├── table_4_nll.csv
    │   ├── table_5_covariance.csv
    │   ├── table_6_var.csv
    │   ├── table_7_portfolio.csv
    │   └── report.md                   # Relatório completo
    └── plots/
        ├── nll_comparison.png
        ├── sharpe_comparison.png
        └── calibration_curves.png
```

---

## 6. Formato de Saída dos Resultados

### 6.1 Relatório Principal (Markdown)

```markdown
# NeuralFactors: Resultados de Avaliação
## Modelo: [checkpoint_name]
## Data: [YYYY-MM-DD]

---

## 1. Sumário Executivo

| Métrica               | Validação | Teste | Baseline (PPCA) | Melhoria |
|-----------------------|-----------|-------|-----------------|----------|
| NLLjoint              | X.XXX     | X.XXX | X.XXX           | +X.X%    |
| Covariance MSE        | X.XXX     | X.XXX | X.XXX           | -X.X%    |
| VaR Calibration Error | X.XXX     | X.XXX | X.XXX           | -X.X%    |
| Sharpe Ratio (L)      | X.XX      | X.XX  | X.XX            | +X.X%    |

---

## 2. Negative Log-Likelihood (Tabela 4)

[Tabela formatada]

---

## 3. Covariance Forecasting (Tabela 5)

[Tabela formatada]

---

[etc...]
```

### 6.2 JSON com Resultados Detalhados

```json
{
  "model": {
    "checkpoint": "path/to/model.ckpt",
    "config": {...},
    "training_steps": 100000
  },
  "data": {
    "train_period": "2005-01-01 to 2018-12-31",
    "val_period": "2019-01-01 to 2022-12-31",
    "test_period": "2023-01-01 to 2025-11-04",
    "num_stocks_avg": 85,
    "returns_std": 0.0627
  },
  "metrics": {
    "nll": {
      "validation": {
        "joint": 0.324,
        "individual": 0.747
      },
      "test": {
        "joint": 0.556,
        "individual": 1.029
      }
    },
    "covariance": {
      "validation": {
        "mse": 0.181,
        "box_m": 1.756
      },
      "test": {
        "mse": 0.282,
        "box_m": 2.226
      }
    },
    "var": {...},
    "portfolio": {...}
  },
  "baselines": {
    "ppca": {...},
    "garch": {...}
  },
  "ablation": {...}
}
```

---

## 7. Cronograma de Implementação

### Fase 1: Infraestrutura Básica (1-2 semanas)
- [ ] Criar módulo `src/evaluation/metrics.py`
- [ ] Implementar NLLjoint e NLLind
- [ ] Criar script `scripts/test.py` básico
- [ ] Testar com modelo treinado existente

### Fase 2: Métricas Avançadas (2-3 semanas)
- [ ] Implementar covariance forecasting (MSE, Box's M)
- [ ] Implementar VaR calibration error
- [ ] Implementar portfolio optimization
- [ ] Calcular Sharpe ratios

### Fase 3: Baselines (2-3 semanas)
- [ ] Implementar PPCA (12 fatores)
- [ ] Integrar GARCH (biblioteca `arch`)
- [ ] Comparar resultados

### Fase 4: Estudos de Ablação (2-3 semanas)
- [ ] Treinar modelos com diferentes F
- [ ] Treinar modelos com diferentes features
- [ ] Treinar com diferentes arquiteturas (LSTM)
- [ ] Treinar com diferentes lookback sizes

### Fase 5: Relatórios e Visualizações (1 semana)
- [ ] Gerar tabelas formatadas (Markdown/LaTeX)
- [ ] Criar plots de comparação
- [ ] Documentar resultados

**Tempo total estimado**: 8-12 semanas

---

## 8. Desafios e Considerações

### 8.1 Diferenças com o Artigo Original

1. **Dados diferentes**:
   - Artigo: S&P 500 (1996-2023), ~500 ações
   - Nossa impl: IBX (2005-2025), ~100 ações
   - **Impacto**: Métricas não serão diretamente comparáveis

2. **Features disponíveis**:
   - Artigo: options features (puts/calls open interest)
   - Nossa impl: pode não ter options data
   - **Solução**: documentar quais features foram usadas

3. **Poder computacional**:
   - Artigo: 24 GPU-hours para treinar
   - Ablation studies: multiplicar por ~15-20 experimentos
   - **Total**: ~300-500 GPU-hours estimados

### 8.2 Limitações Conhecidas

1. **PPCA baseline**:
   - Implementação de PPCA com Student-T não é trivial
   - Alternativa: usar PPCA gaussiano padrão (pior desempenho)

2. **Box's M test**:
   - Teste estatístico complexo para matrizes de covariância
   - Pode requerer biblioteca especializada ou implementação custom

3. **Portfolio optimization**:
   - Otimização quadrática com restrições (L1 norm, positividade)
   - Solver recomendado: CVXPY ou scipy.optimize

### 8.3 Validação dos Resultados

**Sanity checks importantes**:

1. **NLL deve diminuir com mais fatores** (até certo ponto)
   - Se F=8 → F=64: NLL deve melhorar
   - Se F=64 → F=128: pode piorar (overfitting)

2. **Modelos com mais features devem ter melhor NLL**
   - "All features" > "Stock returns only"

3. **IWAE (k=20) deve superar VAE (k=1)**
   - Diferença esperada: ~0.002-0.005 em NLL

4. **Sharpe ratio deve ser positivo**
   - Se negativo: modelo não está fazendo previsões úteis

5. **Calibration error deve ser < 1.0**
   - Se > 1.0: distribuições mal calibradas

---

## 9. Exemplo de Uso do Script

```bash
# 1. Teste básico com modelo treinado
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --data_dir data \
    --output_dir results/test_run_1 \
    --metrics all

# 2. Comparação com baselines
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --data_dir data \
    --output_dir results/full_comparison \
    --metrics all \
    --baselines ppca,garch

# 3. Apenas estudos de ablação
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --data_dir data \
    --output_dir results/ablation \
    --ablation \
    --metrics nll  # mais rápido

# 4. Apenas portfolio optimization
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --data_dir data \
    --output_dir results/portfolio_only \
    --metrics portfolio \
    --strategies long_only,long_short

# Output:
# results/
# ├── test_results.json
# ├── table_4_nll.csv
# ├── table_5_covariance.csv
# ├── table_6_var.csv
# ├── table_7_portfolio.csv
# ├── report.md
# └── plots/
#     ├── nll_comparison.png
#     ├── sharpe_comparison.png
#     └── calibration_curves.png
```

---

## 10. Checklist Final de Implementação

### Core Metrics
- [ ] NLLjoint (100 samples posterior/prior)
- [ ] NLLind (10,000 samples prior)
- [ ] Covariance MSE (whitening)
- [ ] Box's M test statistic
- [ ] VaR calibration error (100 quantis)
- [ ] Sharpe ratio (4 estratégias)

### Baselines
- [ ] PPCA (12 factors, Student-T decoder)
- [ ] GARCH (Skew Student-T, univariate)

### Ablation Studies
- [ ] Número de fatores: {8, 16, 32, 64, 128}
- [ ] Features: {todas, sem options/volume, base, apenas retornos}
- [ ] Arquitetura: {Attention, LSTM, α=0, Prior/Decoder Gaussiano}
- [ ] Loss: {IWAE k=20, VAE k=1}
- [ ] Lookback: {128, 192, 256}
- [ ] Training years: {5, 10, 15, 18}

### Reporting
- [ ] Gerar Tabela 3 (ablation)
- [ ] Gerar Tabela 4 (NLL)
- [ ] Gerar Tabela 5 (covariance)
- [ ] Gerar Tabela 6 (VaR)
- [ ] Gerar Tabela 7 (portfolio)
- [ ] Relatório Markdown completo
- [ ] JSON com resultados estruturados
- [ ] Plots de comparação

### Documentation
- [ ] Documentar diferenças com artigo original
- [ ] Registrar configurações e hyperparameters
- [ ] Explicar escolhas de implementação
- [ ] Listar limitações e caveats

---

## 11. Referências

1. **Artigo principal**: Achintya Gopal (2024). "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities". arXiv:2408.01499v1 [q-fin.ST].

2. **Métricas**:
   - [3] Box, G. E. P. (1949). A general distribution theory for a class of likelihood criteria.
   - [18] Kuleshov et al. Accurate Uncertainties for Deep Learning Using Calibrated Regression.
   - [20] Markowitz, H. (1952). Portfolio Selection. Journal of Finance.

3. **Bibliotecas**:
   - PyTorch Lightning: https://lightning.ai/
   - CVXPY (portfolio opt): https://www.cvxpy.org/
   - arch (GARCH): https://arch.readthedocs.io/
   - scikit-learn (PPCA): https://scikit-learn.org/

---

## 12. Próximos Passos

1. **Revisar este plano** com orientador/equipe
2. **Priorizar métricas** (começar com NLL, depois portfolio)
3. **Implementar fase 1** (infraestrutura básica)
4. **Validar com modelo existente** (50 épocas)
5. **Iterar e expandir** para métricas avançadas

---

**Documento criado em**: 2026-02-12  
**Baseado em**: NeuralFactors article (arXiv:2408.01499v1)  
**Repositório**: diego-azenha/TCC  
**Status**: 📋 Planejamento completo - pronto para implementação
