# =============================================================
# Guia de Desenvolvimento: Validação Quantitativa NeuralFactors
# =============================================================

## O que já foi criado

- Estrutura do script de teste quantitativo (`scripts/test.py`) para o modelo NeuralFactors, com seções para:
  - NLLjoint/NLLind
  - VaR
  - Covariância
  - Backtest de portfólio
- Módulo de backtest (`src/evaluation/test/backtest.py`) para simular a estratégia baseada nos fatores aprendidos, incluindo funções para:
  - Rodar o backtest da estratégia
  - Calcular métricas de performance (retorno, Sharpe, drawdown, etc)
  - Comparar com o Ibovespa
- Integração do backtest ao fluxo do script de teste, permitindo avaliação ponta a ponta.

## Próximos passos

1. **Implementar carregamento real do modelo e dataset**
	- Função `load_model_and_data` em `test.py`.
2. **Implementar cálculo das métricas quantitativas**
	- NLLjoint/NLLind, VaR, covariância, nas funções correspondentes de `test.py`.
3. **Implementar lógica do backtest**
	- Geração de sinais/alocações, rebalanceamento, cálculo dos retornos da estratégia em `run_neuralfactors_backtest`.
4. **Implementar cálculo das métricas de performance**
	- Função `compute_performance_metrics` no módulo de backtest.
5. **Implementar salvamento dos resultados**
	- Função `save_results` em `test.py` (CSV/JSON, gráficos opcionais).
6. **Testar o pipeline completo**
	- Rodar com dados reais, validar outputs e ajustar detalhes.

---
Este guia deve ser seguido para garantir uma validação quantitativa rigorosa e comparável ao artigo, com integração total entre modelo, métricas e avaliação de portfólio.

7. **Reorganizar módulos e resultados**
   - Mover o `analyze.py` para `/src/train` (é um módulo chamado no script de treino, não faz sentido estar em `/scripts`).
   - Criar uma pasta dedicada na raiz do projeto para armazenar gráficos, tabelas e outros resultados (ex: `/visualizations`).
     - Separar essa pasta em subpastas para visualizações do treino e do teste.
   - Mover os gráficos gerados pelo `analyze.py` para essa nova pasta, fora de `/src/evaluation/train`.