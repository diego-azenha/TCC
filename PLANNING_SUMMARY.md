# 📋 Planejamento de Script de Testes - Resumo Executivo

## Status: ✅ PLANEJAMENTO COMPLETO

**Data**: 2026-02-12  
**Baseado em**: NeuralFactors Article (arXiv:2408.01499v1)  
**Objetivo**: Idealizar script de teste que apresenta resultados no mesmo formato do artigo

---

## 📚 Documentos Criados

| Documento | Propósito | Páginas | Status |
|-----------|-----------|---------|--------|
| **TESTING_README.md** | Visão geral e quick start | ~10 | ✅ |
| **PLANO_SCRIPT_TESTES.md** | Especificação técnica completa | ~60 | ✅ |
| **IMPLEMENTACAO_ROADMAP.md** | Guia de implementação com código | ~40 | ✅ |
| **ARCHITECTURE_DIAGRAM.txt** | Diagramas visuais e fluxo | ~15 | ✅ |

**Total**: ~125 páginas de documentação técnica

---

## 🎯 Métricas do Artigo (Reproduzidas)

### Tabela 3: Ablation Studies
- Número de fatores: F ∈ {8, 16, 32, 64, 128}
- Features: 4 variações (completo, sem options, base, apenas retornos)
- Arquitetura: Attention vs LSTM, com/sem α, Student-T vs Gaussian
- Loss: IWAE (k=20) vs VAE (k=1)
- Lookback: L ∈ {128, 192, 256}
- Training years: {5, 10, 15, 18 anos}

**Métrica**: NLLjoint no conjunto de validação

### Tabela 4: Negative Log-Likelihood
- **NLLjoint**: log p(r_t+1 | F_t) - distribuição conjunta
- **NLLind**: média log p(r_i,t+1 | F_t) - distribuições individuais
- Val + Test sets
- Comparação: NeuralFactors vs PPCA vs GARCH

### Tabela 5: Covariance Forecasting
- **MSE**: Erro quadrático da covariância após whitening
- **Box's M**: Teste estatístico para H0: Cov = I
- Val + Test sets
- Comparação: NeuralFactors vs PPCA

### Tabela 6: Value at Risk (VaR)
- **Calibration Error**: Kuleshov et al. métrica
- Duas variantes: Univariate (por ação) e Portfolio (equiponderado)
- Val + Test sets
- Comparação: NeuralFactors vs PPCA vs GARCH
- **Nota**: GARCH tem melhor calibração (como esperado no artigo)

### Tabela 7: Portfolio Optimization
- **Sharpe Ratio**: 4 estratégias
  1. Long-Only (L=1)
  2. Long-Short (L=2)
  3. Long-Only Leveraged (L=2)
  4. Long-Short Leveraged (L=3)
- Val + Test sets
- Comparação: NeuralFactors vs PPCA

---

## 🏗️ Arquitetura do Sistema

```
Trained Model → Evaluation Modules → Test Scripts → Results & Reports
                ├─ metrics.py
                ├─ baselines.py (PPCA, GARCH)
                ├─ portfolio.py
                └─ reporting.py
```

### Módulos Planejados

1. **src/evaluation/metrics.py**
   - compute_nll_joint()
   - compute_nll_individual()
   - compute_covariance_mse()
   - compute_box_m_test()
   - compute_calibration_error()

2. **src/evaluation/baselines.py**
   - PPCAModel (12 factors, Student-T)
   - GARCHModel (Skew Student-T)

3. **src/evaluation/portfolio.py**
   - mean_variance_optimization()
   - sharpe_ratio()
   - 4 estratégias (long_only, long_short, leveraged)

4. **src/evaluation/reporting.py**
   - generate_table_X() para X=3,4,5,6,7
   - plot_comparisons()
   - generate_report()

### Scripts de Teste

1. **scripts/test_nll.py** (Fase 1 - MVP)
2. **scripts/test_portfolio.py** (Fase 2 - MVP)
3. **scripts/test.py** (Complete - todas métricas)

---

## ⏱️ Timeline de Implementação

### MVP (Minimum Viable Product)
**Tempo**: 2-3 semanas  
**Inclui**:
- ✅ NLL comparison (Tabela 4)
- ✅ Sharpe Ratio (Tabela 7)
- ✅ Relatório básico Markdown

**Fases**:
- Fase 0: Setup (1-2 dias)
- Fase 1: NLL (3-5 dias) ← CRÍTICO
- Fase 2: Portfolio (3-5 dias) ← CRÍTICO
- Fase 7 subset: Report (2 dias)

### Implementação Completa (sem ablação)
**Tempo**: 3-4 semanas  
**Inclui**: MVP + Covariance + VaR + Baselines + Visualizações

**Fases adicionais**:
- Fase 3: Covariance (2-3 dias)
- Fase 4: VaR (2-3 dias)
- Fase 5: PPCA (4-5 dias)
- Fase 6: GARCH (2-3 dias)
- Fase 7 completa: Reports (2-3 dias)

### Implementação Total (com ablação)
**Tempo**: 7-8 semanas  
**Inclui**: Tudo acima + Ablation studies

**Fase adicional**:
- Fase 8: Ablações (3-4 semanas)
  - Requer retreinar modelo 15-20 vezes
  - Intensivo em GPU: ~300-500 GPU-hours estimado

---

## 🎓 Diferenças com Artigo Original

### Dados
| Aspecto | Artigo (S&P 500) | Nossa Impl (IBX) |
|---------|------------------|------------------|
| Período | 1996-2023 (27 anos) | 2005-2025 (20 anos) |
| # Ações | ~500 | ~100 |
| Split Train | 1996-2013 (18 anos) | 2005-2018 (14 anos) |
| Split Val | 2014-2018 (5 anos) | 2019-2022 (4 anos) |
| Split Test | 2019-2023 (5 anos) | 2023-2025 (3 anos) |
| Std retornos | 0.0267 | ~0.0627 |

### Features
- Artigo tem: options data (puts/calls open interest)
- Nossa impl: pode não ter options (verificar disponibilidade)

### Implicações
- ⚠️ Valores absolutos de métricas não diretamente comparáveis
- ✅ Tendências relativas devem ser similares
- ✅ Comparações NeuralFactors vs Baselines devem ser consistentes

---

## ✅ Validações e Sanity Checks

### Esperado (baseado no artigo)

1. **NLL**:
   - NeuralFactors < PPCA < baseline
   - NLLjoint < NLLind (distribuição conjunta tem mais informação)
   - Val NLL < Test NLL (usual em ML)

2. **Covariance**:
   - NeuralFactors MSE < PPCA MSE
   - Box's M: quanto menor, melhor
   - NeuralFactors deve superar PPCA significativamente

3. **VaR**:
   - GARCH melhor que todos (especializado em caudas)
   - NeuralFactors competitivo com BDG
   - PPCA pior (gaussiano demais)

4. **Portfolio**:
   - Sharpe > 0 para todos os modelos (senão modelo inútil)
   - NeuralFactors >> PPCA (esperado: 2-4x melhor)
   - Long-Short > Long-Only (mais graus de liberdade)
   - Sharpe razoável: 0.5-3.0 (acima disso suspeito de overfitting)

### Sanity Checks Automáticos

```python
assert nll_joint < nll_ind + 1.0  # Joint should be better
assert 0 < sharpe < 5.0           # Reasonable Sharpe range
assert calibration_error < 1.0    # Reasonably calibrated
assert covariance_mse < 2.0       # Not completely off
```

---

## 📊 Formato de Output

### JSON Estruturado
```json
{
  "model": {...},
  "data": {...},
  "metrics": {
    "nll": {"val": {...}, "test": {...}},
    "covariance": {...},
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

### Markdown Report
- Sumário executivo
- Tabelas 3-7 formatadas
- Comparação com artigo
- Plots incorporados
- Análise de diferenças

### CSV Files
- table_3_ablation.csv
- table_4_nll.csv
- table_5_covariance.csv
- table_6_var.csv
- table_7_portfolio.csv

### Plots
- nll_comparison.png (bar chart)
- sharpe_comparison.png (grouped bar chart)
- calibration_curves.png (line plots)

---

## 🚀 Próximos Passos

### Imediato (Esta Semana)
- [ ] Apresentar planejamento para orientador/equipe
- [ ] Obter aprovação do escopo (MVP vs Completo)
- [ ] Validar recursos disponíveis (GPU, tempo)

### Curto Prazo (Próximas 2-3 Semanas)
- [ ] Fase 0: Setup e validação
- [ ] Fase 1: Implementar NLL metrics
- [ ] Fase 2: Implementar Portfolio optimization
- [ ] Gerar primeiro relatório (MVP)

### Médio Prazo (1 Mês)
- [ ] Implementar métricas restantes (Covariance, VaR)
- [ ] Integrar baselines (PPCA, GARCH)
- [ ] Gerar relatório completo
- [ ] Comparar resultados com artigo

### Longo Prazo (Opcional, 2 Meses)
- [ ] Ablation studies (se houver GPU disponível)
- [ ] Análise de sensibilidade
- [ ] Paper/relatório final

---

## 💡 Recomendações

### Priorização
1. **Começar com MVP** (NLL + Portfolio)
   - Valida implementação atual
   - Gera resultados úteis rapidamente
   - Risco baixo, valor alto

2. **Expandir incrementalmente**
   - Adicionar Covariance e VaR
   - Integrar baselines um de cada vez
   - Validar cada passo

3. **Ablações por último** (ou nunca)
   - Muito custoso em GPU
   - Valor marginal para validação
   - Útil apenas se buscar otimização

### Recursos Necessários

**Hardware**:
- GPU: Preferível, mas não essencial (CPU funciona)
- RAM: 16GB+ recomendado
- Disk: ~50GB para dados e resultados

**Software**:
- Python 3.8+
- PyTorch, PyTorch Lightning
- cvxpy, scipy, statsmodels, arch
- pandas, numpy, matplotlib

**Tempo**:
- MVP: 2-3 semanas (1 pessoa)
- Completo: 3-4 semanas (1 pessoa)
- Com ablações: 7-8 semanas (1 pessoa) + GPU

---

## 📖 Como Usar Esta Documentação

### Para Começar
1. Ler: **TESTING_README.md**
2. Entender: **ARCHITECTURE_DIAGRAM.txt**
3. Planejar: **Este documento** (PLANNING_SUMMARY.md)

### Para Implementar
1. Detalhe técnico: **PLANO_SCRIPT_TESTES.md**
2. Código passo-a-passo: **IMPLEMENTACAO_ROADMAP.md**
3. Seguir fases sequencialmente

### Para Reportar
1. Usar templates em **PLANO_SCRIPT_TESTES.md** Seção 6
2. Comparar com Tabelas do artigo
3. Documentar diferenças (IBX vs S&P 500)

---

## ✨ Destaques do Planejamento

1. ✅ **Completo**: Todas métricas do artigo especificadas
2. ✅ **Prático**: Código de exemplo para cada fase
3. ✅ **Realista**: Timeline baseado em complexidade real
4. ✅ **Flexível**: MVP permite validação rápida
5. ✅ **Documentado**: 4 documentos cobrindo todos aspectos
6. ✅ **Validável**: Sanity checks e comparações com artigo
7. ✅ **Incremental**: Fases independentes, valor em cada passo

---

## 📞 Contato e Suporte

Este planejamento foi criado com base no:
- Artigo NeuralFactors (Achintya Gopal, 2024)
- README.md existente do repositório
- Análise do código implementado
- Boas práticas de avaliação de ML

Para dúvidas ou sugestões:
- Revisar documentação detalhada (4 docs criados)
- Consultar artigo original para clarificações
- Discutir com orientador/equipe

---

**Status Final**: 📋 **PLANEJAMENTO COMPLETO E PRONTO PARA IMPLEMENTAÇÃO**

**Criado em**: 2026-02-12  
**Repositório**: diego-azenha/TCC  
**Branch**: copilot/plan-testing-script-format
