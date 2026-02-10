# Análise Detalhada do Treinamento - NeuralFactors (50 épocas)

## Resumo Executivo

**Problema Principal**: Volatilidade crescente da loss ao longo do treinamento, com instabilidade numérica significativa.

---

## 1. Análise da Loss

### Training Loss (por step)
- **Tendência**: Melhora geral de -23.7 → -138.3 (482% de melhora)
- **Valor final**: -209.84
- **Problema crítico**: Volatilidade aumentou **+65.22%**
  - Primeira metade: std = 216.93
  - Segunda metade: std = 358.40
- **Pico anômalo**: 6432.5 no step 113199 (época ~33)

### Training Loss (por época)
```
Época  0: +29.87  (baseline)
Época 10: -56.86  
Época 20: -90.63  
Época 30: -93.47  
Época 40: -107.24 
Época 49: -127.30 (final)
```

**Observações**:
- Épocas 0-20: Convergência rápida e consistente
- Épocas 20-30: Platô com oscilações (~-90 a -100)
- Épocas 30-49: Oscilações mais severas entre -93 e -136
- **Não há convergência clara nas últimas 20 épocas**

---

## 2. Análise dos Componentes da Loss

### KL Divergence (q(z|r,F) || p(z))
- **Tendência**: Diminuição de 92.95 → 54.64 (-41.22%) ✓
  - Indica que posterior está se aproximando do prior
  - Comportamento esperado em VAEs
- **Problema**: Volatilidade **+97.11%** (dobrou!)
  - Primeira metade: std = 120.08
  - Segunda metade: std = 236.70
- **Pico extremo**: 4623.36 no step 113199

### Log Likelihood (log p(r|F,z))
- **Tendência**: Aumento de 110.69 → 185.78 (+67.84%) ✓
  - Modelo está melhorando em explicar os dados
- **Volatilidade**: +15.03% (aumento moderado)
- **Valor final**: 238.68

**Interpretação**:
- Log likelihood melhorando = modelo aprendendo
- KL diminuindo mas volátil = encoder instável
- **A volatilidade da loss vem principalmente da KL divergence**

---

## 3. Effective Sample Size (ESS)

### Problema Crítico: Degeneração do IWAE

- **Tendência**: Diminuição de 4.06 → 2.95 (-27.35%)
- **Valor inicial**: ~4 de 20 samples (20% efetivo)
- **Valor final**: **1.05 de 20 samples (5% efetivo!)**
- **Pior caso**: ESS = 1.0 em alguns steps (apenas 1 sample dominante)
- **Melhor caso**: ESS = 19.27 no step 158299 (anômalo)

**O que ESS baixo significa**:
1. Dos 20 importance samples, quase todos têm peso ~0
2. Um único sample domina o gradiente
3. Estimativa do gradiente tem alta variância
4. **Causa direta da volatilidade crescente**

**Por que isso acontece**:
- Importance weights: w_k = p(r,z_k) / q(z_k|r,F)
- Se p e q divergem, alguns pesos explodem
- Com treino longo, distribuições se especializam demais
- Resultado: degeneração dos pesos

---

## 4. Possíveis Causas Raiz

### 4.1 Colapso do Posterior (Posterior Collapse)
**Evidência**:
- KL diminuindo para ~40 (esperado seria > 50)
- ESS caindo drasticamente
- Volatilidade crescente

**Mecanismo**:
- Encoder produz posterior q(z|r,F) muito "peaky" (concentrada)
- Poucos samples de q recebem probabilidade significativa sob p
- IWAE degenera

### 4.2 Instabilidade Numérica
**Evidência**:
- Picos absurdos (6432, 4623)
- Ambos no mesmo ponto (~step 113199, época 33)
- Sugerem overflow ou divisão por zero

**Possíveis fontes**:
- Cholesky decomposition no encoder
- Log-sum-exp em importance weights
- Parâmetros sigma/nu muito pequenos ou grandes

### 4.3 Learning Rate Fixo
**Evidência**:
- LR constante em 1e-4 por 172.900 steps
- Nenhum warmup ou decay
- Volatilidade piora após época 30

**Problema**:
- LR muito alto para fine-tuning nas épocas finais
- Modelo "salta" em vez de convergir
- Explica oscilações -93 ↔ -136

### 4.4 Divergência Prior-Posterior
**Evidência**:
- Prior aprende distribuição global dos fatores
- Encoder aprende distribuição condicional específica
- Gap entre ambos aumenta com otimização

**Resultado**:
- Importance weights mal-calibrados
- ESS cai conforme gap cresce

---

## 5. Verificações Adicionais Necessárias

### 5.1 Analisar Parâmetros do Modelo
```python
# Verificar evolução de:
- train/prior_sigma_z_mean, train/prior_sigma_z_std
- train/prior_nu_z
- train/alpha_mean, train/alpha_std
- train/sigma_mean, train/sigma_std
- train/nu_mean
```

### 5.2 Inspecionar Step 113199
- Por que pico extremo?
- Quais foram os valores dos parâmetros?
- Houve gradient explosion?

### 5.3 Comparar com Paper
- Paper: 100k steps com batch size 1
- Nosso: 172.9k steps (10 épocas a mais)
- ESS do paper não reportado, mas provavelmente similar

---

## 6. Possíveis Soluções (em ordem de prioridade)

### 6.1 Learning Rate Scheduler (PRIORITÁRIO)
**Implementar**:
- Warmup: 0 → 1e-4 nos primeiros 10% dos steps
- Cosine decay: 1e-4 → 1e-6 nos últimos 50-70%
- Ou step decay: 1e-4 → 5e-5 → 1e-5

**Expectativa**: Reduzir volatilidade nas fases finais

### 6.2 Gradient Clipping
**Implementar**:
- Clip norm em 1.0 ou 5.0
- Previne gradient explosion
- Especialmente importante para IWAE

### 6.3 Aumentar Número de Samples (k)
**Testar**:
- k = 20 → k = 50 ou k = 100
- Mais samples = ESS mais robusto
- Trade-off: custo computacional

### 6.4 Modificar Beta-VAE ou Free Bits
**Implementar**:
- Beta-VAE: loss = log_lik - β * KL (β < 1)
- Free Bits: min(KL, threshold) para evitar colapso
- Previne posterior collapse

### 6.5 Jitter Adaptativo no Encoder
**Verificar**:
- Jitter atual: 1e-4 → 10.0
- Se Cholesky falha, aumentar agressivamente
- Pode estar relacionado aos picos

---

## 7. Análise dos Parâmetros Aprendidos

### 7.1 Prior Distribution p(z)

**Prior Sigma (scale)**:
- Inicial: 10.0 → Final: 1.32 (-86.75%)
- **Colapso severo**: Prior está ficando muito concentrado
- Media: 5.84, Std: 2.41

**Prior Sigma Std** (heterogeneidade entre fatores):
- Inicial: 0.0 → Final: 0.95
- Indica que diferentes fatores têm escalas diferentes
- Max: 1.92 (step 123599)

**Prior Nu (graus de liberdade)**:
- Inicial: 10.0 → Final: 23.5 (+135%)
- Nu alto (>20) = distribuição quase Gaussiana
- **Contradiz motivação do Student-T** (modelar caudas pesadas!)

**Interpretação**:
- Prior está "esquecendo" a natureza heavy-tailed dos retornos
- Está convergindo para Gaussiana (nu → ∞)
- Colapso do sigma indica que fatores estão sendo subutilizados

### 7.2 Decoder Parameters (α, σ, ν)

**Alpha (location parameter)**:
- Mean: 0.027 → -0.002 (-109%) - **cruzou zero**
- Std: 0.052 → 0.028 (-47%)
- Range: [-0.13, 0.04]
- Comportamento esperado (retornos centrados em ~0)

**Sigma (scale parameter)** - **PROBLEMA CRÍTICO**:
- Mean: 0.708 → 0.312 (-56%)
- **Std: 0.026 → 0.463 (+1676%!)** ← **CAUSA RAIZ**
- Min: 0.0002 (quase zero!)
- Max: 4.29 (extremo!)
- **Heterogeneidade explodiu entre stocks**

**Nu (degrees of freedom)**:
- Mean: 4.64 → 6.74 (+45%)
- Std: 0.66
- Range: [4.03, 10.41]
- Nu moderado = caudas pesadas mantidas no decoder (bom!)

**Interpretação**:
- **Decoder sigma é a fonte da instabilidade**
- Alguns stocks: sigma → 0 (modelo confiante, baixo ruído)
- Outros stocks: sigma → 4+ (modelo incerto, alto ruído)
- Heterogeneidade de 1676% causa:
  1. Instabilidade numérica em log-probabilities
  2. Gradientes desbalanceados
  3. KL divergence volátil

### 7.3 Anomalia no Step 113199 (Época 33)

**Valores normais (step 113099)**:
- Loss: -166.75
- KL: 49.74
- Log-lik: 209.97
- ESS: 1.13

**Pico anômalo (step 113199)**:
- Loss: **+6432.55** (explosão!)
- KL: **+4623.36** (explosão!)
- Log-lik: **-1818.39** (negativo extremo!)
- ESS: 1.08 (já estava baixo)

**Recuperação (step 113299)**:
- Loss: -199.37 (voltou ao normal)
- KL: 46.46 (voltou ao normal)
- Log-lik: 238.24 (voltou ao normal)
- ESS: 1.52 (ainda baixo)

**Diagnóstico**:
- Evento isolado, **não propagou instabilidade**
- Provável batch com outliers severos
- Decoder tentou ajustar com sigma extremo
- KL explodiu: posterior muito diferente do prior
- **Evidência de fragilidade numérica**, não colapso permanente

---

## 8. Diagnóstico Final: Causa Raiz Identificada

### 8.1 Problema Principal: Overfitting por Stock

**Mecanismo**:
1. Cada stock tem seus próprios parâmetros α, σ, ν via decoder
2. Com 50 épocas, decoder overfit em stocks individuais
3. Alguns stocks: σ → 0.0002 (confiança excessiva)
4. Outros stocks: σ → 4.29 (incerteza extrema)
5. **Heterogeneidade de +1676% é insustentável**

**Consequências em cascata**:
```
Decoder σ heterogêneo 
  ↓
Log p(r|F,z) com valores extremos (-∞ a +∞)
  ↓
Posterior q(z|r,F) distorce para compensar
  ↓
KL divergence volátil (+97% volatilidade)
  ↓
Importance weights w_k desbalanceados
  ↓
ESS colapsa para 1.05 (de 20 samples)
  ↓
Gradientes de alta variância
  ↓
Loss volátil (+65% volatilidade)
```

### 8.2 Por Que Não Apareceu em 5 Épocas?

**Treinamento curto (5 épocas)**:
- Decoder não teve tempo de overfit
- Sigma std: ~0.1 (estimativa)
- ESS: provavelmente ~5-10
- Volatilidade: moderada

**Treinamento longo (50 épocas)**:
- Decoder convergiu para soluções stock-específicas
- Sigma std: 0.463 (+1676%)
- ESS: 1.05 (colapso)
- Volatilidade: severa

**Conclusão**: 
- 5 épocas = underfitting controlado
- 50 épocas = overfitting descontrolado
- **Ponto ótimo provavelmente em 15-25 épocas**

### 8.3 Por Que Prior Colapsou?

**Prior learning**:
- Prior aprende distribuição marginal p(z) dos fatores
- Se decoder overfit, encoder gera posteriors muito específicos
- Prior não consegue generalizar, colapsa (σ: 10 → 1.3)
- Nu aumenta (10 → 23.5), perdendo heavy tails

**Interpretação**:
- Prior está tentando "competir" com decoder overfit
- Mas prior é global (todos os stocks), decoder é local (por stock)
- **Conflito estrutural** quando treinamento é muito longo

---

## 9. Soluções Corretivas (Revisadas)

### Prioridade 1: Regularizar Decoder Sigma ⭐⭐⭐

**Problema**: Sigma std +1676% causa toda cascata de instabilidade

**Solução**:
```python
# Adicionar perda de regularização
sigma_reg = torch.std(sigma) / torch.mean(sigma)
loss = iwae_loss + lambda_sigma * sigma_reg

# Ou: limitar range de sigma
sigma = torch.clamp(sigma, min=0.01, max=2.0)

# Ou: usar prior sobre sigma
sigma_prior_loss = -log_normal(sigma, mean=0.5, std=0.5)
```

**Expectativa**: 
- Reduzir heterogeneidade de sigma para ~50-100%
- Estabilizar KL divergence
- Aumentar ESS para >5

### Prioridade 2: Gradient Clipping ⭐⭐⭐

**Problema**: Picos extremos (6432, 4623) indicam gradient explosion

**Solução**:
```python
# No trainer
trainer = pl.Trainer(
    gradient_clip_val=1.0,  # ou 5.0
    gradient_clip_algorithm="norm"
)
```

**Expectativa**: Prevenir anomalias como step 113199

### Prioridade 3: Aumentar K (IWAE samples) ⭐⭐

**Problema**: ESS = 1.05 de 20 samples (colapso severo)

**Solução**:
```python
# Aumentar k de 20 para 50 ou 100
num_iwae_samples = 50  # ou 100
```

**Trade-off**: 2.5x-5x mais cálculo
**Expectativa**: ESS > 5, gradientes mais estáveis

### Prioridade 4: Learning Rate Scheduler ⭐⭐

**Problema**: LR fixo 1e-4 por 172k steps impede fine-tuning

**Solução**:
```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_steps, eta_min=1e-6
)

# Ou: warmup + decay
# 0 → 1e-4 (10% steps)
# 1e-4 → 1e-6 (cosine, 90% steps)
```

**Expectativa**: Convergência mais suave nas épocas finais

### Prioridade 5: Treinar por Menos Épocas ⭐

**Problema**: 50 épocas induz overfitting estrutural

**Solução**: Testar 15-20 épocas
- ~51k-69k steps vs 172k steps
- Melhor balanço bias-variance
- Verificar se sigma std fica controlado

---

## 10. Conclusões e Recomendações

### 10.1 Principais Achados

1. ✅ **Modelo está aprendendo**: Loss melhora, log likelihood aumenta
2. ⚠️ **Overfitting por stock**: Decoder sigma std +1676%
3. ❌ **IWAE degenerando**: ESS cai de ~4 para 1.05
4. ❌ **Prior colapsando**: Sigma -87%, Nu +135% (→ Gaussiana)
5. ⚠️ **Instabilidade numérica**: Picos extremos, mas recuperação rápida
6. ⚠️ **LR fixo inadequado**: Contribui, mas não é causa raiz

### 10.2 Causa Raiz

**Decoder sigma sem regularização** → **Heterogeneidade explosiva** → **Cascata de instabilidade em IWAE**

### 10.3 Recomendações (em ordem)

1. **Implementar regularização de sigma no decoder** (crítico)
2. **Adicionar gradient clipping** (prevenir explosões)
3. **Aumentar k de 20 para 50** (estabilizar ESS)
4. **Testar 15-20 épocas** (evitar overfitting)
5. **Adicionar LR scheduler** (melhorar convergência)
6. **Monitorar sigma std durante treino** (early stopping)

### 10.4 Próximos Experimentos

**Experimento 1: Baseline Corrigido**
- 20 épocas (69k steps)
- k = 50
- Gradient clip = 1.0
- LR scheduler: cosine 1e-4 → 1e-6
- **Verificar se sigma std fica < 0.3**

**Experimento 2: Sigma Regularizado**
- Baseline + sigma regularização
- λ = 0.01 ou 0.1
- Target: sigma std < 0.2

**Experimento 3: Sigma Clamping**
- Baseline + torch.clamp(sigma, 0.02, 1.5)
- Forçar range físico plausível

**Experimento 4: Beta-VAE**
- Baseline + β = 0.5
- Reduzir pressão em KL
- Pode aumentar ESS

### 10.5 Métricas de Sucesso

- [x] Loss convergindo: ✓ (mas volátil)
- [x] Log likelihood aumentando: ✓
- [ ] **ESS > 5** ao final: ✗ (1.05)
- [ ] **Sigma std < 0.3**: ✗ (0.463)
- [ ] **Volatilidade da loss < 200**: ✗ (358)
- [ ] **KL estável**: ✗ (volatilidade +97%)
- [ ] **Sem anomalias**: ✗ (step 113199)

**Status**: 2/7 métricas satisfeitas. Modelo aprende, mas de forma instável.
