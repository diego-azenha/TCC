# Test Script Usage - NeuralFactors Evaluation

Este documento explica como utilizar o script de teste (`scripts/test.py`) para avaliar o modelo NeuralFactors e produzir resultados no formato do artigo.

## Visão Geral

O script `test.py` implementa todas as métricas de avaliação descritas no artigo "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities" (Gopal, 2024). O script produz resultados formatados de acordo com as Tabelas 4-7 do artigo.

### Métricas Implementadas

#### Tabela 4: Negative Log-Likelihood (NLL)
- **NLL_joint**: Log-verossimilhança negativa da distribuição conjunta
- **NLL_ind**: Log-verossimilhança negativa das distribuições individuais

#### Tabela 5: Previsão de Covariância
- **MSE**: Erro quadrático médio dos retornos whitened
- **Box's M**: Teste estatístico para igualdade de covariância

#### Tabela 6: Calibração de VaR (Value at Risk)
- **Calibration Error**: Medida de quão bem os quantis previstos correspondem aos quantis empíricos

#### Tabela 7: Otimização de Portfólio
- **Sharpe Ratio**: Índice de Sharpe do portfólio otimizado
- **Market Sharpe**: Índice de Sharpe do mercado (equal-weighted)
- **Excess Return**: Retorno anual excedente do portfólio

## Pré-requisitos

### 1. Instalar Dependências

```bash
# Instalar PyTorch (com suporte CUDA se disponível)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar outras dependências
pip install -r requirements.txt

# Dependências específicas do test.py:
pip install scipy tqdm
```

### 2. Modelo Treinado

Você precisa de um checkpoint de modelo treinado. Os checkpoints devem estar no formato PyTorch Lightning (`.ckpt`).

### 3. Dados

O script espera encontrar os dados no seguinte formato:
- `data/parquets/x_ts.parquet`: Features de séries temporais
- `data/parquets/x_static.parquet`: Features estáticas
- `data/cleaned/fechamentos_ibx.csv`: Preços de fechamento

## Uso Básico

### Comando Simples

```bash
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --data_dir data \
    --split test
```

### Parâmetros Disponíveis

```bash
python scripts/test.py --help
```

Parâmetros principais:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--checkpoint` | *obrigatório* | Caminho para o arquivo checkpoint (.ckpt) |
| `--data_dir` | `data` | Diretório contendo os dados |
| `--split` | `test` | Split do dataset: `train`, `val`, ou `test` |
| `--output_dir` | `results_test` | Diretório de saída para resultados |
| `--num_joint_samples` | `100` | Número de amostras para NLL_joint (paper: 100) |
| `--num_ind_samples` | `1000` | Número de amostras para NLL_ind (paper: 10000) |
| `--num_quantiles` | `100` | Número de quantis para calibração VaR |

### Exemplos de Uso

#### 1. Avaliar no Conjunto de Teste

```bash
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --split test \
    --output_dir results_test
```

#### 2. Avaliar no Conjunto de Validação

```bash
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --split val \
    --output_dir results_val
```

#### 3. Avaliação Completa (como no artigo)

```bash
# Usar mais amostras para estimativas mais precisas (mais lento)
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --split test \
    --num_joint_samples 100 \
    --num_ind_samples 10000 \
    --output_dir results_final
```

#### 4. Avaliação Rápida (para debugging)

```bash
# Usar menos amostras para teste rápido
python scripts/test.py \
    --checkpoint checkpoints/neuralfactors/last.ckpt \
    --split test \
    --num_joint_samples 20 \
    --num_ind_samples 100 \
    --output_dir results_quick
```

## Formato de Saída

O script gera os seguintes arquivos no diretório de saída:

### 1. `results_{split}.json`
Arquivo JSON com sumário das métricas:

```json
{
  "split": "test",
  "nll_joint": 0.3240,
  "nll_ind": 0.7470,
  "cov_mse": 0.110000,
  "box_m": 0.092000,
  "var_calibration": 0.009200,
  "sharpe_ratio": 0.8500,
  "market_sharpe": 0.5200,
  "excess_return": 0.0450
}
```

### 2. `results_table_{split}.txt`
Tabela formatada no estilo do artigo:

```
================================================================================
NEURALFACTORS EVALUATION RESULTS - TEST SET
================================================================================

Based on: Gopal, A. (2024). NeuralFactors: A Novel Factor Learning Approach
          to Generative Modeling of Equities. arXiv:2408.01499v1

--------------------------------------------------------------------------------
Table 4: Negative Log-Likelihood (Lower is better)
--------------------------------------------------------------------------------
NLL_joint:    0.3240
NLL_ind:      0.7470

--------------------------------------------------------------------------------
Table 5: Covariance Forecasting (Lower is better)
--------------------------------------------------------------------------------
MSE:          0.110000
Box M:        0.092000

--------------------------------------------------------------------------------
Table 6: VaR Calibration Error (Lower is better)
--------------------------------------------------------------------------------
Calibration:  0.009200

--------------------------------------------------------------------------------
Table 7: Portfolio Performance
--------------------------------------------------------------------------------
Sharpe Ratio: 0.8500
Market Sharpe:0.5200
Excess Return:0.0450
================================================================================
```

### 3. `nll_joint_per_day_{split}.csv`
Valores de NLL_joint para cada dia (útil para análise temporal).

## Interpretação dos Resultados

### Negative Log-Likelihood (NLL)
- **Valores menores são melhores**
- Compara a qualidade do modelo generativo
- NLL_joint mede a distribuição conjunta de todos os ativos
- NLL_ind mede as distribuições marginais individuais

### Covariance Forecasting
- **Valores menores são melhores**
- MSE mede o erro de previsão da matriz de covariância
- Box's M testa se a covariância prevista está correta

### VaR Calibration Error
- **Valores menores são melhores**
- Mede quão bem o modelo estima o risco
- Erro de calibração próximo a zero indica quantis bem calibrados

### Portfolio Performance
- **Valores maiores são melhores para Sharpe Ratio**
- Sharpe Ratio > 1.0 é considerado bom
- Excess Return mede o retorno anualizado acima do mercado

## Comparação com o Artigo

Para comparar seus resultados com o artigo original (Tabelas 4-7):

### Conjunto de Dados
- **Artigo**: S&P 500 constituents (1996-2023)
- **Este projeto**: IBX Brazilian stocks (2005-2025)

### Normalização
- Os retornos são normalizados pelo desvio padrão do período de treino
- **Artigo**: std ≈ 0.0267
- **IBX**: std ≈ 0.0627

### Performance Esperada
Seus resultados podem diferir do artigo devido a:
1. Diferente conjunto de dados (IBX vs S&P 500)
2. Diferente período temporal
3. Características diferentes do mercado brasileiro

## Troubleshooting

### Erro: "No module named 'numpy'"
```bash
pip install numpy pandas scipy torch pytorch-lightning
```

### Erro: "CUDA out of memory"
Use menos amostras:
```bash
python scripts/test.py ... --num_joint_samples 50 --num_ind_samples 500
```

### Erro: "Checkpoint not found"
Verifique se o caminho do checkpoint está correto:
```bash
ls -la checkpoints/neuralfactors/
```

### Aviso: "Covariance computation failed"
Isso pode acontecer se houver instabilidade numérica. Tente:
1. Usar um checkpoint diferente
2. Avaliar em um split diferente
3. Verificar se os dados estão corretos

## Notas de Performance

### Tempo de Execução Estimado

Para o conjunto de teste completo (~740 dias) com um modelo de 64 fatores:

| Métrica | Tempo Estimado | Configuração |
|---------|---------------|--------------|
| NLL_joint (100 samples) | ~10-15 min | GPU |
| NLL_ind (1000 samples) | ~30-45 min | GPU |
| Covariance | ~5-10 min | GPU |
| VaR Calibration | ~15-20 min | GPU |
| Portfolio | ~5-10 min | GPU |
| **Total** | **~1-2 horas** | GPU |

**CPU**: Espere 3-5x mais tempo se executando em CPU.

### Otimizações

Para acelerar a avaliação:
1. Use GPU se disponível
2. Reduza `num_ind_samples` de 10000 para 1000
3. Avalie apenas as métricas principais (comente as outras no código)

## Referências

- Gopal, A. (2024). NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities. arXiv:2408.01499v1 [q-fin.ST]. https://arxiv.org/abs/2408.01499

## Contato e Suporte

Para questões ou problemas com o script de teste, consulte:
- README principal: `README.md`
- Documentação do código: comentários em `scripts/test.py`
- Análise de resultados: `results_analysis/analysis_report.md`
