# Pipeline de Limpeza e Tratamento de Dados — NeuralFactors Brasil

Documentação do pipeline implementado. Descreve o que foi construído, os dados utilizados, as decisões tomadas, as limitações conhecidas e possibilidades de melhoria.

---

## 0. Contexto: Como o Modelo Consome os Dados

O `StockEmbedder` processa **um dia de negociação por vez**. Para cada dia $t$, retorna os parâmetros $(\alpha, \beta, \sigma, \nu)$ de todos os $N_t$ ativos presentes no universo naquele dia. Os inputs esperados são:

| Tensor | Shape | Conteúdo |
|---|---|---|
| `S` | `[N, L, d_ts]` | Janela de lookback: para cada ativo, os últimos $L = 256$ dias úteis de features temporais |
| `S_static` | `[N, d_static]` | Features estáticas (setor) de cada ativo |
| `r` (target) | `[N]` | Retorno do dia $t+1$ (log-return normalizado) |
| `mask` | `[N]` | Máscara booleana indicando ativos válidos |

O vetor temporal em cada timestep $u$ para o ativo $i$ é:

$$S[i, u, :] = \Big[\underbrace{r_{i,u}}_{\text{retorno}},\ \underbrace{X^{ts,\text{stock}}_{i,u}}_{\text{fundamentais}},\ \underbrace{X^{ts,\text{market}}_{u}}_{\text{índices Bloomberg}}\Big]$$

Todas as features entram concatenadas na dimensão $d_{ts}$. O modelo não distingue internamente entre stock-specific e market-wide.

---

## 1. Dados Brutos Utilizados

Todos os arquivos brutos estão em `raw/`.

### 1.1 Economatica — Preços (`raw/economatica/diario/`)

| Arquivo | Métrica | Frequência | Período |
|---|---|---|---|
| `fechamento.csv` | Fechamento ajust p/ proventos | Diária | 2004-12 → 2026-03 |
| `preco_valor_patrimonial.csv` | P/VPA (Preço / Valor Patrimonial) | Diária | 2004-12 → 2026-03 |

Formato wide Economatica: `Ativo | Data | ~1.419 colunas de tickers`. Missing codificado como `"-"`. P/VPA tem ~28.7% de células não-nulas (cobertura diária para tickers ativos).

### 1.2 Economatica — Fundamentais Trimestrais (`raw/economatica/trimestral/`)

| Arquivo | Métrica | Natureza | Período |
|---|---|---|---|
| `ROA.csv` | Return on Assets (TTM 12 meses) | Rentabilidade | 1986-01 → 2025-12 |
| `ROE.csv` | Return on Equity (TTM 12 meses) | Rentabilidade | 2004-12 → 2025-12 |
| `margembruta.csv` | Margem Bruta (TTM 12 meses) | Rentabilidade | 2004-12 → ~2025 |
| `dividabruta_ativo.csv` | Dívida Bruta / Ativo | Alavancagem | 2004-12 → ~2025 |
| `dividaliq_pl.csv` | Dívida Líquida / PL | Alavancagem | 2004-12 → ~2025 |

Mesmo formato wide, em grade diária. Valores efetivos existem apenas em ~139–244 datas de fim de trimestre — o restante da grade é todo `"-"`. ROA, ROE e margem bruta já vêm como trailing-twelve-months (TTM) na fonte; as métricas de dívida são ratios de balanço point-in-time.

### 1.3 Bloomberg — Índices de Mercado (`raw/bloomberg_indices_values.xlsx`)

Arquivo Excel com 5 sheets e 29 séries de índices (após deduplicação), período 2005-01-03 → 2026-03-26. Os índices que efetivamente entraram no pipeline, conforme registrado em `normalization_stats.json`:

| Categoria | Índices |
|---|---|
| Risco & Sentimento | VIX Index, MOVE Index, BRAZIL CDS USD SR 5Y D14 Corp |
| Brasil — Macro & Mercado | BZDIOVRA Index (DI Over), USDBRL Curncy |
| Brasil — Equity Factors | MXBRSC Index, MXBRLC Index, MXBR000V Index, IDIV Index, MLCXBV Index, MU702608 Index |
| Renda Fixa | BZRFIMAB Index, BZRFIMA Index, SPUHYBDT Index |
| Commodities | BCOMAGTR, BCOMGCTR, BCOMINTR, BCOMNGTR, BCOMSITR, BCOMCOT |
| Internacional (MSCI) | MXEF, MXCN, MXJP, MXGB, MXCA, MXEU, MXLA, MXPCJ, MXUS |

### 1.4 Economatica — Classificação Setorial (`raw/setor_ibovespa.xlsx`)

Arquivo com ~1.420 tickers da Bovespa (ativos + cancelados), contendo classificação em 3 níveis. O pipeline usa apenas `setor_economico` (análogo a GICS Level 1).

### 1.5 Dados Não Utilizados ou Não Disponíveis

| Dado | Situação |
|---|---|
| `raw/composicao_ibovespa.xlsx` | Presente, **não usado** — universo definido implicitamente por preços |
| Volume diário por ativo | **Não disponível**. `log(volume)` seria feature temporal stock-specific |
| P/E, EV/EBITDA, FCF Yield | **Não disponível**. Features de valuation adicionais |

---

## 2. Arquitetura do Pipeline

O pipeline se divide em 4 camadas, cada uma produzindo artefatos persistentes em Parquet. O fluxo é orquestrado por `processing/run_all.py`, que importa e executa sequencialmente cada script:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CAMADA 0       │     │  CAMADA 1       │     │  CAMADA 2       │     │  CAMADA 3       │
│  Raw Ingestion  │────▶│  Clean & Tidy   │────▶│  Feature Eng.   │────▶│  Model-Ready    │
│                 │     │                 │     │                 │     │                 │
│  raw/           │     │  cleaned/       │     │  features/      │     │  parquets/      │
│  (CSV, XLSX)    │     │  (Parquet long) │     │  (Parquet long) │     │  x_ts.parquet   │
│                 │     │                 │     │                 │     │  x_static.parquet│
│                 │     │                 │     │                 │     │  prices.parquet  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Scripts e Módulos Compartilhados

| Arquivo | Papel |
|---|---|
| `processing/config.py` | Constantes globais: caminhos (`ROOT`, `RAW`, `CLEANED`, `FEATURES`, `PARQUETS`), split temporal (`TRAIN_END = "2018-12-31"`, `VAL_END = "2022-12-31"`, `MIN_DATE = "2005-01-03"`), mapa dos 6 CSVs de fundamentais (`FUNDAMENTAL_FILES` — 5 trimestrais + P/VPA diário) |
| `processing/io_utils.py` | Duas funções de leitura reutilizáveis: `read_economatica_wide()` para CSVs Economatica (wide → long, tratamento de `"-"` e duplicatas de coluna) e `read_bloomberg_indices()` para o Excel Bloomberg (5 sheets → DataFrame wide consolidado, deduplicação de colunas como BCOMINTR) |
| `processing/run_all.py` | Orquestrador sequencial: importa e executa `main()` de cada script 01–10, com timing por etapa |

---

## 3. Camada 1: Limpeza (`cleaned/`)

### 3.1 `01_clean_prices.py` → `cleaned/prices.parquet`

Lê `fechamento.csv` via `read_economatica_wide()`. Remove preços ≤ 0, filtra para datas ≥ 2005-01-03.

**Schema:** `date: datetime64 | ticker: str | close: float64`

Este parquet é a **master key do universo**: qualquer dado de fundamentais ou features só é utilizado para pares `(date, ticker)` que existam aqui. Se um ticker tem preço válido na data, ele está no universo — sem dependência de composição histórica de índice.

**Números da execução:** 956 tickers únicos ao longo de todo o período.

### 3.2 `02_clean_fundamentals.py` → `cleaned/{metric}.parquet` (×6)

Para cada CSV registrado em `FUNDAMENTAL_FILES`, lê via `read_economatica_wide()`, filtra tickers pelo universo de `prices.parquet`, filtra datas ≥ 2005-01-01, e aplica **winsorização nos percentis 1% e 99%** para conter outliers extremos.

**Outputs:** `cleaned/roa.parquet`, `cleaned/roe.parquet`, `cleaned/margem_bruta.parquet`, `cleaned/divida_bruta_ativo.parquet`, `cleaned/divida_liq_pl.parquet`, `cleaned/pvpa.parquet`

**Schema (cada):** `date: datetime64 | ticker: str | {metric}: float64`

Para os 5 trimestrais, cada arquivo contém apenas datas de fim de trimestre com valores efetivos. Para P/VPA (diário), o arquivo contém ~2.25M rows com dados em cada dia de negociação onde o ativo tem valor.

### 3.3 `03_clean_bloomberg.py` → `cleaned/market_indices.parquet`

Lê as 5 sheets do Excel via `read_bloomberg_indices()`. Filtra datas ≥ 2005-01-03 e aplica **interpolação linear** para gaps de até 3 dias por índice. Gaps maiores permanecem NaN.

**Schema:** `date: datetime64 | VIX Index: float64 | MOVE Index: float64 | ... (29 colunas de índices)`

### 3.4 `04_clean_sectors.py` → `cleaned/sectors.parquet`

Lê `setor_ibovespa.xlsx` com `header=3`. Extrai `Código` e `Setor Econômico`. Mapeia setor `"-"` para `"Outros"`. Deduplica por ticker (mantém primeira ocorrência). Atribui `sector_id` inteiro em ordem alfabética determinística.

**Schema:** `ticker: str | setor_economico: str | sector_id: int`

---

## 4. Camada 2: Feature Engineering (`features/`)

### 4.1 `05_feature_returns.py` → `features/returns.parquet`

Calcula log-returns por ticker: $r_{i,t} = \ln(P_{i,t} / P_{i,t-1})$. Substitui ±Inf por NaN e descarta o primeiro dia de cada ticker (retorno indefinido).

**Schema:** `date: datetime64 | ticker: str | return: float64`

### 4.2 `06_feature_fundamentals.py` → `features/fundamentals_ffill.parquet`

Para cada uma das 6 métricas em `FUNDAMENTAL_FILES`, faz merge com o calendário diário completo de `prices.parquet`, e aplica `ffill()` por ticker. **Sem `bfill()`** — evita look-ahead bias.

Para os trimestrais, o ffill propaga o último valor reportado por ~60 dias úteis até o próximo reporte. Para P/VPA (diário), o ffill apenas preenche gaps pontuais de poucos dias. O resultado é um DataFrame com uma linha por `(date, ticker)` com as 6 colunas preenchidas.

Lógica temporal (ilustração):
```
Datas de reporte:   2005-03-31    2005-06-30    2005-09-30
Valor ROA:           0.08          0.09          0.07

Grade diária:
  2005-04-01 → 0.08  (forward-fill do Q1)
  2005-04-02 → 0.08
  ...
  2005-06-30 → 0.09  (novo valor Q2 entra)
  2005-07-01 → 0.09  (forward-fill do Q2)
```

**Schema:** `date: datetime64 | ticker: str | roa | roe | margem_bruta | divida_bruta_ativo | divida_liq_pl | pvpa: float64`

### 4.3 `07_feature_indices.py` → `features/index_returns.parquet`

Calcula log-returns para cada série de índice Bloomberg. Substitui ±Inf por NaN. Descarta a primeira linha (sem retorno). Adiciona sufixo `_ret` a cada coluna.

**Schema:** `date: datetime64 | VIX Index_ret: float64 | MOVE Index_ret: float64 | ... (29 colunas _ret)`

**Decisão sobre VIX:** Apesar de o VIX ser um nível de volatilidade e não um preço, é usado como variação (log-return) para consistência com os demais índices. O nível do VIX poderia ser adicionado como feature separada — não foi feito.

---

## 5. Camada 3: Normalização e Montagem Final (`parquets/`)

### 5.1 Split Temporal

Definido em `config.py`:

| Split | Período | Uso |
|---|---|---|
| Train | 2005-01-04 → 2018-12-31 | Treino + estatísticas de normalização |
| Validation | 2019-01-01 → 2022-12-31 | Early stopping |
| Test | 2023-01-01 → fim dos dados | Avaliação final |

**Regra cardinal:** Todas as estatísticas de normalização ($\mu$, $\sigma$) são computadas **exclusivamente** no período de treino.

### 5.2 `08_assemble_x_ts.py` → `parquets/x_ts.parquet` + `normalization_stats.json` + `parquets/prices.parquet`

**Merge:**
1. `features/returns.parquet` define o universo `(date, ticker)`.
2. Left join com `features/fundamentals_ffill.parquet` em `(date, ticker)`.
3. Left join com `features/index_returns.parquet` em `date` (broadcast: mesmo valor para todos os tickers na mesma data).

**Normalização (usando apenas dados com `date ≤ TRAIN_END`):**

- **Retornos:** Divisão por $\sigma_{train}$ apenas, sem subtrair a média. $\tilde{r}_{i,t} = r_{i,t} / \sigma_{train}$. O $\sigma_{train}$ obtido = **0.0545**.
- **Fundamentais:** Z-score global. $\tilde{f}_{i,t} = (f_{i,t} - \mu_{f,train}) / \sigma_{f,train}$. Uma média e desvio por feature, pooled across tickers and dates no período de treino.
- **Índices:** Z-score por série. $\tilde{r}^{idx}_t = (r^{idx}_t - \mu^{idx}_{train}) / \sigma^{idx}_{train}$.
- **Divisão por zero:** Se $\sigma_{f,train} = 0$ para alguma feature, o valor é fixado em 0.0.

**NaN residuais:** Após todas as transformações, NaN remanescentes são preenchidos com 0.0 (= média na escala normalizada).

**Schema final de `x_ts.parquet`:**

```
date:               datetime64
ticker:             str
return:             float64   ← log-return / σ_train
roa:                float64   ← z-score global (treino)
roe:                float64   ← z-score global (treino)
margem_bruta:       float64   ← z-score global (treino)
divida_bruta_ativo: float64   ← z-score global (treino)
divida_liq_pl:      float64   ← z-score global (treino)
pvpa:               float64   ← z-score global (treino)
VIX Index_ret:      float64   ← z-score por série (treino)
MOVE Index_ret:     float64
... (demais 27 séries de índices)
```

**Resultado:** $d_{ts} = 38$ (1 retorno + 8 fundamentais + 29 índices). 956 tickers no total, 841 no período de treino.

O arquivo `parquets/prices.parquet` é uma cópia de `cleaned/prices.parquet` para consumo direto pelo dataset loader (cálculo do retorno-alvo $r_{i,t+1}$).

O `normalization_stats.json` contém todas as estatísticas usadas, o `feature_order`, e metadados dimensionais — garante reprodutibilidade.

### 5.3 `09_assemble_x_static.py` → `parquets/x_static.parquet`

Carrega `cleaned/sectors.parquet`. Identifica tickers presentes em `x_ts.parquet` mas ausentes em sectors, mapeando-os para `"Outros"`. Gera one-hot de `setor_economico` com `pd.get_dummies(..., dtype=float)`.

**Schema:** `ticker: str | Bens industriais: float64 | Comunicações: float64 | ... (uma coluna por setor presente)`

Cada ticker aparece uma única vez. **Sem dimensão temporal** — o merge no dataset loader é feito apenas por `ticker`.

$d_{static}$ = número de categorias únicas de setor encontradas nos dados.

### 5.4 `10_validate_final.py`

Valida os artefatos finais com 6 verificações:

1. **NaN / Inf:** Verifica ausência em `x_ts`, `x_static`, `prices`
2. **Integridade dimensional:** Checa `d_ts` contra `normalization_stats.json`
3. **Normalização no treino:** Confirma std ≈ 1 para retornos, mean ≈ 0 e std ≈ 1 para fundamentais no período de treino
4. **Consistência de tickers:** Todo ticker em `x_ts` deve existir em `x_static`
5. **Cobertura temporal:** Conta tickers com ≥ 256 dias de histórico por data; reporta datas com < 30 tickers elegíveis
6. **Alinhamento de datas:** Datas de `x_ts` devem ser subset das datas em `prices`

O script retorna exit code 1 se houver falhas críticas.

---

## 6. Estrutura do Repositório

```
TCC Data Cleaning/
├── _claude/
│   └── data_pipeline_plan.md          ← este documento
│
├── raw/                                ← Camada 0: dados brutos
│   ├── economatica/
│   │   ├── diario/
│   │   │   ├── fechamento.csv
│   │   │   └── preco_valor_patrimonial.csv
│   │   └── trimestral/
│   │       ├── ROA.csv
│   │       ├── ROE.csv
│   │       ├── margembruta.csv
│   │       ├── dividabruta_ativo.csv
│   │       └── dividaliq_pl.csv
│   ├── bloomberg_indices_values.xlsx
│   ├── setor_ibovespa.xlsx
│   └── composicao_ibovespa.xlsx        (não usado)
│
├── cleaned/                            ← Camada 1: dados limpos (long format)
│   ├── prices.parquet                  (master key do universo)
│   ├── roa.parquet
│   ├── roe.parquet
│   ├── margem_bruta.parquet
│   ├── divida_bruta_ativo.parquet
│   ├── divida_liq_pl.parquet
│   ├── pvpa.parquet
│   ├── market_indices.parquet
│   └── sectors.parquet
│
├── features/                           ← Camada 2: features engenheiradas
│   ├── returns.parquet
│   ├── fundamentals_ffill.parquet
│   └── index_returns.parquet
│
├── parquets/                           ← Camada 3: model-ready
│   ├── x_ts.parquet                    (date × ticker × 36 features normalizadas)
│   ├── x_static.parquet                (ticker × one-hot setores)
│   ├── prices.parquet                  (preços brutos para retorno-alvo)
│   └── normalization_stats.json        (estatísticas de normalização)
│
└── processing/                         ← Código do pipeline
    ├── config.py                       (caminhos, constantes, split temporal)
    ├── io_utils.py                     (leitores Economatica + Bloomberg)
    ├── run_all.py                      (orquestrador sequencial)
    ├── 01_clean_prices.py
    ├── 02_clean_fundamentals.py
    ├── 03_clean_bloomberg.py
    ├── 04_clean_sectors.py
    ├── 05_feature_returns.py
    ├── 06_feature_fundamentals.py
    ├── 07_feature_indices.py
    ├── 08_assemble_x_ts.py
    ├── 09_assemble_x_static.py
    └── 10_validate_final.py
```

---

## 7. Como o StockEmbedder Deve Ler os Parquets Finais

Esta seção descreve de forma concreta como o código de treinamento deve carregar e transformar os artefatos de `parquets/` nos tensores consumidos pelo modelo.

### 7.1 Carregamento e Pré-Indexação

```python
import pandas as pd
import numpy as np
import json
import torch

# ── Carregar os 4 artefatos ──
x_ts     = pd.read_parquet("parquets/x_ts.parquet")      # (1.7M rows × 40 cols)
x_static = pd.read_parquet("parquets/x_static.parquet")  # (956 rows × 12 cols)
prices   = pd.read_parquet("parquets/prices.parquet")     # (1.7M rows × 3 cols)

with open("parquets/normalization_stats.json") as f:
    stats = json.load(f)

feature_cols = stats["feature_order"]   # lista ordenada de 38 features
d_ts         = stats["d_ts"]            # 38
```

A coluna `return` dentro de `x_ts` já é o retorno normalizado $r_{i,t}/\sigma_{train}$ — ela é usada como **feature de input** no lookback, não como target. O target $r_{i,t+1}$ deve ser calculado a partir de `prices.parquet`.

### 7.2 Pré-Processamento Recomendado (uma vez, no `__init__`)

Para permitir buscas $O(1)$ por ticker/data durante o treinamento:

```python
# Ordenar x_ts por ticker e data para buscas de janela eficientes
x_ts = x_ts.sort_values(["ticker", "date"]).reset_index(drop=True)

# Extrair a matriz de features como numpy (alinhada ao index)
feature_matrix = x_ts[feature_cols].values              # shape: (n_rows, 38)
dates_array    = x_ts["date"].values                     # shape: (n_rows,)
tickers_array  = x_ts["ticker"].values                   # shape: (n_rows,)

# Indexar linhas por ticker para lookback rápido
ticker_groups = x_ts.groupby("ticker").indices            # dict: ticker → array de row indices

# Calendário de datas disponíveis (ordenado)
all_dates = np.sort(x_ts["date"].unique())

# x_static como dict para lookup O(1)
static_cols = [c for c in x_static.columns if c != "ticker"]
static_dict = x_static.set_index("ticker")[static_cols]  # DataFrame indexado por ticker
d_static    = len(static_cols)                            # 11

# Preços para cálculo do retorno-alvo
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
price_lookup = prices.pivot(index="date", columns="ticker", values="close")
```

### 7.3 Construção dos Tensores para uma Data $t$ (`__getitem__`)

Cada batch corresponde a **um dia de negociação** $t$. O modelo recebe todos os ativos presentes naquela data.

```python
L = 256  # tamanho da janela de lookback

def build_sample(date_t):
    # 1. Universo: tickers presentes em x_ts na data t
    mask_date = x_ts["date"] == date_t
    tickers_t = x_ts.loc[mask_date, "ticker"].unique()

    S_list      = []    # lookback windows
    S_stat_list = []    # static features
    r_list      = []    # target returns
    valid_mask  = []    # True se o ticker tem lookback completo + target

    # 2. Data do target: próximo dia útil após t
    t_idx    = np.searchsorted(all_dates, date_t)
    if t_idx + 1 >= len(all_dates):
        return None  # último dia, sem target
    date_t1  = all_dates[t_idx + 1]

    for ticker in tickers_t:
        rows_idx = ticker_groups[ticker]                  # row indices deste ticker
        ticker_dates = dates_array[rows_idx]

        # Posição da data t no histórico deste ticker
        pos = np.searchsorted(ticker_dates, date_t)

        # 3a. Lookback: precisa de L dias de histórico (incluindo t)
        if pos + 1 < L:
            # Histórico insuficiente → mask = False, preencher com zeros
            S_list.append(np.zeros((L, d_ts), dtype=np.float32))
            S_stat_list.append(static_dict.loc[ticker].values.astype(np.float32))
            r_list.append(0.0)
            valid_mask.append(False)
            continue

        # Janela de L timesteps: [t-L+1, ..., t]
        window_rows = rows_idx[pos - L + 1 : pos + 1]
        S_i = feature_matrix[window_rows]                 # (L, d_ts)

        # 3b. Static
        S_static_i = static_dict.loc[ticker].values       # (d_static,)

        # 3c. Target: log-return bruto de t → t+1 (NÃO normalizado)
        p_t  = price_lookup.loc[date_t,  ticker] if date_t  in price_lookup.index else np.nan
        p_t1 = price_lookup.loc[date_t1, ticker] if date_t1 in price_lookup.index else np.nan
        r_target = np.log(p_t1 / p_t) if (not np.isnan(p_t) and not np.isnan(p_t1) and p_t > 0) else np.nan

        has_target = not np.isnan(r_target)

        S_list.append(S_i.astype(np.float32))
        S_stat_list.append(S_static_i.astype(np.float32))
        r_list.append(r_target / stats["returns_std"] if has_target else 0.0)
        valid_mask.append(has_target)

    # 4. Empilhar em tensores
    S        = torch.tensor(np.stack(S_list))              # (N, L, d_ts)
    S_static = torch.tensor(np.stack(S_stat_list))         # (N, d_static)
    r        = torch.tensor(r_list, dtype=torch.float32)   # (N,)
    mask     = torch.tensor(valid_mask, dtype=torch.bool)  # (N,)

    return S, S_static, r, mask
```

### 7.4 Contratos e Garantias dos Dados

O pipeline garante os seguintes invariantes nos arquivos finais:

| Propriedade | Garantia |
|---|---|
| NaN / Inf em `x_ts` | Zero — todos eliminados (NaN residuais → 0.0) |
| NaN / Inf em `x_static` | Zero |
| Consistência de tickers | Todo ticker em `x_ts` existe em `x_static` |
| Alinhamento temporal | Datas em `x_ts` ⊆ datas em `prices` |
| Ordem das features | Idêntica a `stats["feature_order"]` |
| Normalização | Todas as features em `x_ts` já estão normalizadas pelos stats do treino |
| Tipo dos dados | Todas as features são `float64` |

### 7.5 Resumo dos Shapes

| Tensor | Shape | Origem |
|---|---|---|
| `S` | `[N_t, 256, 38]` | `x_ts.parquet`, coluna `feature_order` |
| `S_static` | `[N_t, 11]` | `x_static.parquet`, colunas de setor |
| `r` | `[N_t]` | `prices.parquet`, log-return $t \to t{+}1$ normalizado por $\sigma_{train}$ |
| `mask` | `[N_t]` | `True` se lookback ≥ 256 dias e target disponível |

Onde $N_t$ varia por data (~200–400 ativos em datas típicas).

### 7.6 Split Temporal para Treinamento

O dataset loader deve filtrar as datas disponíveis segundo o split:

```python
train_dates = all_dates[all_dates <= np.datetime64("2018-12-31")]
val_dates   = all_dates[(all_dates > np.datetime64("2018-12-31")) & (all_dates <= np.datetime64("2022-12-31"))]
test_dates  = all_dates[all_dates > np.datetime64("2022-12-31")]
```

A primeira data treinável efetiva não é 2005-01-04, mas ~2006-01 (256 dias úteis depois do início dos dados). O primeiro ano serve apenas para construir o lookback.

### 7.7 Notas Importantes

1. **Features de índices são idênticas para todos os tickers**: As 29 colunas `*_ret` de Bloomberg repetem o mesmo valor para todos os ativos na mesma data. O modelo não precisa tratá-las de forma diferente — elas entram concatenadas no vetor $S[i, u, :]$.

2. **O retorno em `x_ts` é feature, não target**: A coluna `return` em `x_ts` é o retorno do dia $u$ normalizado — serve como input no lookback. O target $r_{i,t+1}$ deve ser calculado a partir de `prices.parquet` (preços brutos).

3. **Nenhum ticker tem identidade explícita**: O modelo não recebe o nome do ticker. A identidade do ativo é implícita nas suas features temporais e no setor estático. Dois ativos do mesmo setor com trajetórias similares produzirão embeddings similares.

4. **Variabilidade de $N_t$**: O número de ativos varia entre datas. O dataset loader deve suportar batch com $N$ variável (padding + mask, ou collate customizado).

---

## 8. Tratamento de Missings e Edge Cases

| Situação | Tratamento Implementado | Script |
|---|---|---|
| Preço = NaN ou ≤ 0 | Excluído de `prices.parquet` | `01_clean_prices.py` |
| Fundamentais com outliers extremos | Winsorização percentis 1/99 | `02_clean_fundamentals.py` |
| Índice Bloomberg com gap ≤ 3 dias | Interpolação linear | `03_clean_bloomberg.py` |
| Índice Bloomberg com gap > 3 dias | NaN mantido → preenchido com 0.0 após z-score | `03` + `08` |
| Ticker sem setor | Mapeado para "Outros" | `04_clean_sectors.py` + `09_assemble_x_static.py` |
| Retorno = ±Inf | Substituído por NaN → descartado | `05_feature_returns.py` |
| Fundamental sem valor (pré-primeiro reporte) | NaN após ffill → preenchido com 0.0 | `06` + `08` |
| Gap entre reportes trimestrais | Forward-fill (sem bfill) | `06_feature_fundamentals.py` |
| NaN residuais em x_ts | Preenchidos com 0.0 (≈ média normalizada) | `08_assemble_x_ts.py` |
| Ticker com < 256 dias de histórico | Excluído via mask no dataset loader | Runtime |

---

## 9. Decisões de Design e Justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Formato intermediário | Parquet long | Compressão nativa, tipagem forte, I/O rápido |
| Universo de ativos | Definido por preço de fechamento válido | Pragmático; evita dependência de composição histórica do IBX |
| Normalização de retornos | Divisão por $\sigma_{train}$ sem subtrair média | Consistente com o paper original; o $\alpha_{i,t}$ do modelo captura a média esperada |
| Normalização de fundamentais | Z-score global com stats do treino | Simples, consistente com lookback, padrão em factor investing acadêmico |
| Normalização de índices | Z-score temporal por série com stats do treino | Cada série na mesma escala |
| Forward-fill fundamentais | `ffill()` sem `bfill()` | `bfill()` introduziria look-ahead bias |
| Winsorização de fundamentais | Clip nos percentis 1%/99% | Contém outliers sem descartar dados |
| Missing final | Preencher com 0.0 | Após z-score, 0.0 ≡ "na média". O masking trata casos extremos |
| Índices como log-returns | Log-return, não nível | Modelo precisa ver variação diária, não tendência de longo prazo |
| Setor como one-hot | `setor_economico` (análogo a GICS Level 1) | Granularidade adequada para ~200 ativos no universo BRB |
| Fallback setorial | Tickers sem setor → "Outros" | Garante cobertura 100% no x_static |

---

## 10. Limitações Conhecidas

### 10.1 Features Ausentes

- **Volume diário por ativo:** Principal feature temporal stock-specific ausente. O paper original usa `log(volume)`. Sem volume, o modelo não tem informação sobre liquidez ou intensidade de negociação.
- **Features de valuation adicionais (P/E, EV/EBITDA, FCF Yield):** P/VPA já está incluído, mas outros múltiplos de valuation ampliariam a dimensão stock-specific temporal.
- **Nível do VIX como feature separada:** Apenas a variação do VIX entra. O nível absoluto carrega informação de regime de mercado (alto VIX = stress) que poderia complementar a variação.

### 10.2 Normalização

- **Z-score global vs. cross-sectional para fundamentais:** A abordagem implementada (z-score global com stats do treino) é simples e consistente, mas não captura a posição relativa de uma empresa versus seus peers em cada data. Uma normalização cross-sectional por data capturaria melhor o ranking relativo, porém complicaria o lookback — o mesmo valor de ROA de uma empresa mudaria dependendo do universo presente naquele dia.
- **Sensibilidade a outliers residuais:** A winsorização 1/99 ajuda, mas persiste sensibilidade a caudas pesadas. Exemplo: Dívida Líquida/PL tem $\sigma_{train} = 151.6$, indicando distribuição altamente leptocúrtica mesmo após winsorização.

### 10.3 Definição do Universo

- O universo é definido por preço de fechamento válido, não por composição real do IBX ou outro critério de investibilidade. Isso inclui tickers de empresas muito pequenas ou ilíquidas que talvez não fossem investíveis.
- Não há filtro de liquidez mínima ou free-float. Ativos com pouquíssimas negociações entram no universo da mesma forma que blue chips. Isso pode introduzir noise no treino.

### 10.4 Qualidade dos Dados

- Os fundamentais Economatica podem conter atrasos de publicação não modelados: o pipeline assume que o valor aparece na data do fim do trimestre fiscal, mas na realidade a empresa publica semanas depois. Isso cria um **look-ahead bias leve** nos fundamentais.
- Não há verificação de stock splits ou corporate actions nos preços — a premissa é que `fechamento.csv` já vem ajustado pela Economatica.

### 10.5 Validação

- O script `10_validate_final.py` não verifica explicitamente a ausência de look-ahead bias nos fundamentais (data de publicação vs. data fiscal).
- Não valida cobertura setorial mínima (percentual de tickers não-"Outros").

---

## 11. Possibilidades de Melhoria

### Curto Prazo (sem novos dados)

1. **Incluir nível do VIX como feature adicional:** Concatenar a série de níveis do VIX (já limpa em `market_indices.parquet`) como coluna extra em `x_ts`, normalizada por z-score.
3. **Filtro de liquidez:** Usar o próprio `prices.parquet` para excluir tickers com menos de $k$ observações de preço por mês, ou com gaps frequentes. Criaria um universo mais realista.
4. **Normalização robusta dos fundamentais:** Substituir z-score por winsorized z-score, ou usar medianas ao invés de médias para computar as estatísticas de treino. Alternativa: rank-transform para features de cauda pesada como Dívida Líq./PL.

### Médio Prazo (novos dados necessários)

5. **Volume diário:** Obter via Economatica ou outra fonte. Processar como `log(volume + 1)`, normalizar por z-score do treino, merge em x_ts. Impacto potencialmente alto — o paper destaca volume como feature relevante.
6. **Features de valuation:** P/E, EV/EBITDA, FCF Yield. Expandiriam significativamente a dimensão stock-specific.
7. **Data de publicação real dos fundamentais:** Se disponível, usar a data de divulgação (ao invés de fim de trimestre) para o ffill, eliminando o look-ahead bias leve.

### Longo Prazo (estrutural)

8. **Normalização cross-sectional adaptativa:** Implementar ranking percentílico por data para fundamentais, mantendo consistência com o lookback via lookup table de ranks.
9. **Features dinâmicas de setor:** Ao invés de one-hot estático, usar a evolução do setor do ativo ao longo do tempo (empresas podem mudar de classificação).
10. **Embedding aprendido para setor:** Substituir one-hot por um embedding treinável de dimensão menor, especialmente se o número de categorias crescer.

---

## 12. Números Produzidos (Referência)

Extraídos de `normalization_stats.json` após a última execução:

| Métrica | Valor |
|---|---|
| Período de treino | 2005-01-04 → 2018-12-31 |
| $\sigma_{train}$ (retornos) | 0.0545 |
| $d_{ts}$ | 38 |
| $d_{static}$ | determinado pelo número de setores únicos |
| Tickers no treino | 841 |
| Tickers total (todos os períodos) | 956 |

**Feature order em `x_ts`:** `return`, `roa`, `roe`, `margem_bruta`, `divida_bruta_ativo`, `divida_liq_pl`, `pvpa`, `ev_ebitda`, `preco_lucro`, seguidos de 29 séries de retornos de índices Bloomberg (sufixo `_ret`).

**Extensibilidade:** Para adicionar uma nova feature fundamental, basta adicionar a entrada em `FUNDAMENTAL_FILES` no `config.py` e reexecutar o pipeline — o `d_ts` é descoberto automaticamente.
