# Test Script Planning - NeuralFactors Implementation

## 📋 Visão Geral

Este conjunto de documentos apresenta o planejamento completo de um script de testes para avaliar o modelo NeuralFactors implementado neste repositório, seguindo rigorosamente o formato de apresentação de resultados do artigo científico original.

**Artigo de Referência**: "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities" por Achintya Gopal (arXiv:2408.01499v1)

---

## 📚 Documentos Disponíveis

### 1. [PLANO_SCRIPT_TESTES.md](./PLANO_SCRIPT_TESTES.md)
**Planejamento Detalhado e Especificações Técnicas**

Este documento contém:
- ✅ Descrição completa de todas as métricas do artigo (Tabelas 3-7)
- ✅ Fórmulas matemáticas e procedimentos de cálculo
- ✅ Especificações de formato de saída (JSON, Markdown, CSV)
- ✅ Estrutura de código proposta (`src/evaluation/`)
- ✅ Requisitos de baselines (PPCA, GARCH)
- ✅ Considerações sobre diferenças entre dados IBX e S&P 500

**Conteúdo Principal**:
1. Visão geral e objetivos
2. Estrutura de dados (train/val/test splits)
3. Métricas de avaliação detalhadas:
   - **Tabela 3**: Estudos de ablação (número de fatores, features, arquitetura)
   - **Tabela 4**: Negative Log-Likelihood (NLLjoint, NLLind)
   - **Tabela 5**: Covariance Forecasting (MSE, Box's M)
   - **Tabela 6**: Value at Risk - VaR (Calibration Error)
   - **Tabela 7**: Portfolio Optimization (Sharpe Ratio)
4. Baselines (PPCA, GARCH)
5. Estrutura de código proposta
6. Formato de saída dos resultados
7. Cronograma (8-12 semanas)
8. Desafios e considerações
9. Exemplos de uso
10. Checklist de implementação

---

### 2. [IMPLEMENTACAO_ROADMAP.md](./IMPLEMENTACAO_ROADMAP.md)
**Roteiro Prático e Incremental de Implementação**

Este documento fornece:
- ✅ Estratégia de implementação iterativa (começar simples, validar cedo)
- ✅ Código de exemplo para cada métrica
- ✅ Scripts prontos para executar
- ✅ Checklist por fase
- ✅ Cronograma realista (3-4 semanas para MVP)
- ✅ Priorização clara de tarefas

**Fases de Implementação**:

| Fase | Tempo | Prioridade | Descrição |
|------|-------|------------|-----------|
| **Fase 0** | 1-2 dias | Alta | Preparação: setup, dependências, validação |
| **Fase 1** | 3-5 dias | **Crítica** | **NLL metrics** (NLLjoint, NLLind) |
| **Fase 2** | 3-5 dias | **Crítica** | **Portfolio Optimization** (Sharpe Ratio) |
| **Fase 3** | 2-3 dias | Alta | Covariance Forecasting (MSE, Box's M) |
| **Fase 4** | 2-3 dias | Média | VaR Calibration Error |
| **Fase 5** | 4-5 dias | Alta | PPCA Baseline |
| **Fase 6** | 2-3 dias | Média | GARCH Baseline |
| **Fase 7** | 2-3 dias | Alta | Relatórios e Visualizações |
| **Fase 8** | 3-4 semanas | Baixa (opcional) | Estudos de Ablação |

**MVP (Mínimo Viável)**: Fases 0-2 + Fase 7 = **2-3 semanas**
- NLL comparison (Tabela 4)
- Sharpe Ratio (Tabela 7)
- Relatório formatado

---

## 🎯 Objetivos do Teste

### Objetivo Principal
Reproduzir as métricas de avaliação do artigo NeuralFactors para validar a implementação e comparar com baselines clássicos usando dados brasileiros (IBX).

### Métricas Principais (do Artigo)

1. **Negative Log-Likelihood (NLL)**
   - Avalia qualidade das previsões probabilísticas
   - Duas variantes: NLLjoint (distribuição conjunta) e NLLind (distribuições marginais)
   - **Esperado**: NeuralFactors > PPCA > GARCH (na maioria das métricas)

2. **Covariance Forecasting**
   - Avalia acurácia na previsão de correlações entre ações
   - Métricas: MSE e Box's M test
   - **Esperado**: NeuralFactors > BDG > PPCA

3. **Value at Risk (VaR)**
   - Avalia calibração das estimativas de risco
   - Métrica: Calibration Error
   - **Esperado**: GARCH melhor que NeuralFactors (conforme artigo)

4. **Portfolio Optimization**
   - Avalia utilidade prática para construção de portfólios
   - Métrica: Sharpe Ratio (4 estratégias)
   - **Esperado**: NeuralFactors > PPCA significativamente

---

## 🚀 Como Começar

### Passo 1: Revisar os Documentos
```bash
# Ler planejamento detalhado
cat PLANO_SCRIPT_TESTES.md

# Ler roadmap de implementação
cat IMPLEMENTACAO_ROADMAP.md
```

### Passo 2: Setup do Ambiente (Fase 0)
```bash
# Instalar dependências adicionais
pip install cvxpy scipy statsmodels arch

# Verificar que modelo treinado existe
ls checkpoints/neuralfactors_50epochs/

# Verificar que dados estão disponíveis
ls data/parquets/
```

### Passo 3: Implementar MVP (Fases 1-2)
```bash
# Criar estrutura de diretórios
mkdir -p src/evaluation
mkdir -p results
mkdir -p scripts

# Seguir implementação da Fase 1 (NLL) em IMPLEMENTACAO_ROADMAP.md
# Criar: src/evaluation/metrics.py
# Criar: scripts/test_nll.py

# Seguir implementação da Fase 2 (Portfolio) em IMPLEMENTACAO_ROADMAP.md
# Criar: src/evaluation/portfolio.py
# Criar: scripts/test_portfolio.py
```

### Passo 4: Executar Testes
```bash
# Rodar NLL evaluation
python scripts/test_nll.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --data_dir data

# Rodar Portfolio backtesting
python scripts/test_portfolio.py \
    --checkpoint checkpoints/neuralfactors_50epochs/best.ckpt \
    --strategies long_only,long_short
```

---

## 📊 Formato de Resultados Esperado

### Exemplo: Tabela 4 (NLL Comparison)

```markdown
| Modelo                      | Val NLLind | Val NLLjoint | Test NLLind | Test NLLjoint |
|----------------------------|------------|--------------|-------------|---------------|
| NeuralFactors-Attention    | 0.747      | 0.324        | 1.029       | 0.556         |
| PPCA (12 Factors)          | 1.016      | 0.441        | 1.326       | 0.664         |
```

### Exemplo: Tabela 7 (Portfolio Optimization)

```markdown
| Modelo                  | Val Sharpe (L) | Test Sharpe (L) |
|------------------------|----------------|-----------------|
| NeuralFactors-Attention | 1.87          | 1.20            |
| PPCA (12 Factors)       | 0.43          | 0.56            |
```

---

## ⚠️ Considerações Importantes

### Diferenças com o Artigo Original

1. **Dados Diferentes**:
   - **Artigo**: S&P 500 (1996-2023), ~500 ações
   - **Nossa Implementação**: IBX (2005-2025), ~100 ações
   - **Impacto**: Valores absolutos das métricas não serão diretamente comparáveis

2. **Features Disponíveis**:
   - Artigo usa options features (puts/calls open interest)
   - Nossa implementação pode não ter todas as features
   - **Solução**: Documentar claramente quais features foram usadas

3. **Normalização**:
   - Artigo: std ≈ 0.0267 (S&P 500)
   - IBX: std ≈ 0.0627 (a ser confirmado)
   - **Importante**: NLL values devem ser ajustados pela normalização usada

### Validações Essenciais

✅ **Sanity Checks**:
1. NLL deve diminuir com mais fatores (até certo ponto)
2. Modelos com mais features devem ter melhor NLL
3. IWAE (k=20) deve superar VAE (k=1)
4. Sharpe ratio deve ser positivo e razoável (< 5.0)
5. Calibration error deve ser < 1.0

---

## 📈 Cronograma Resumido

### MVP (Minimum Viable Product)
**Tempo**: 2-3 semanas  
**Inclui**:
- NLL metrics (Tabela 4)
- Sharpe Ratio (Tabela 7)
- Relatório básico

### Implementação Completa (sem ablação)
**Tempo**: 3-4 semanas  
**Inclui**: MVP + Covariance + VaR + Baselines + Visualizações

### Implementação Total (com ablação)
**Tempo**: 7-8 semanas  
**Inclui**: Tudo acima + estudos de ablação (Tabela 3)
- Requer retreinar modelo 15-20 vezes
- Muito intensivo em GPU

---

## 🤝 Contribuindo

Este é um planejamento inicial. Contribuições e sugestões são bem-vindas:

1. **Revisar** os documentos e identificar gaps
2. **Priorizar** quais métricas são mais importantes
3. **Implementar** seguindo o roadmap incremental
4. **Validar** cada métrica contra o artigo
5. **Documentar** resultados e diferenças encontradas

---

## 📖 Referências

1. **Artigo Principal**: Achintya Gopal (2024). "NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities". arXiv:2408.01499v1 [q-fin.ST].

2. **Implementação Atual**: Veja [README.md](./README.md) para detalhes da arquitetura implementada.

3. **Análise de Treinamento**: Veja [results_analysis/analysis_report.md](./results_analysis/analysis_report.md) para análise detalhada do modelo treinado (50 épocas).

---

## 📞 Próximos Passos

1. ✅ **Revisar documentação de planejamento**
2. ⬜ **Aprovar roadmap com orientador/equipe**
3. ⬜ **Começar Fase 0**: Setup e validação
4. ⬜ **Implementar Fase 1**: NLL metrics
5. ⬜ **Validar resultados**: Comparar com artigo
6. ⬜ **Iterar**: Adicionar métricas incrementalmente

---

**Status**: 📋 **Planejamento Completo** - Pronto para iniciar implementação  
**Última Atualização**: 2026-02-12  
**Mantido por**: Equipe TCC - NeuralFactors Implementation
