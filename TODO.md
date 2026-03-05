## Objetivo
Adicionar dois baselines ao pipeline de avaliação:
- PPCA
- Fama-French (3 fatores)

---

## Tarefas

- Implementar PPCA ajustado no período de treino.
- Implementar Fama-French com regressão rolling por ativo.
- Garantir que ambos usem os mesmos dados de treino/teste do NeuralFactors.
- Gerar previsões fora da amostra de:
  - Covariância
  - Retorno esperado (quando aplicável)
- Calcular:
  - Log-likelihood (NeuralFactors e PPCA)
  - Erro de covariância
  - Métricas de portfólio (Sharpe, volatilidade, drawdown)
- Integrar os dois modelos ao pipeline atual de teste para rodarem como opções alternativas.
- Exportar métricas comparáveis entre os três modelos.