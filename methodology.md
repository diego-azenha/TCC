# 1. Formulação do Problema

- **Background** (Modelagem de distribuições condicionais)

- Qual o objetivo do modelo? O que queremos modelar/obter?

---

O ponto central da metodologia é a modelagem de distribuições condicionais:

$$
p(y \mid x)
$$

O objetivo do modelo é aprender:

$$
p(r_{t+1} \mid \mathcal{F}_t)
$$

ou seja, a **distribuição conjunta** de $N_{t+1}$ (quantas ações existem no dia seguinte) retornos $r_{t+1}$ (retornos do dia seguinte), utilizando toda a informação disponível até o momento $t$.

Com fundamentos na teoria fatorial clássica (CAPM, Ross, etc.), o artigo utiliza o seguinte **processo generativo**:

$$
z_{t+1} \sim t_{\nu_z}(\mu_z, \sigma_z)
\tag{5}
$$

$$
r_{i,t+1} \sim p(r_{i,t+1} \mid z_{t+1}, \mathcal{F}_t)
\quad \text{para } i = 1 \text{ até } N_{t+1}
\tag{6}
$$

---

### Interpretação das Equações

**(5)**: $z_{t+1}$ é um vetor de fatores latentes que representam o estado sistemático do mercado no tempo $t+1$.

O vetor $z_{t+1}$ captura toda a dependência (correlações) que observamos nos ativos.

Esse vetor segue uma **distribuição Student-t multivariada** com $\nu_z$ graus de liberdade, média $\mu_z$ e dispersão $\sigma_z$.

**(6)**: O retorno da ação $i$ no tempo $t+1$ segue uma distribuição condicional que depende do vetor de fatores latentes no tempo $t+1$ e das informações históricas disponíveis $\mathcal{F}_t$.

---

Utilizando esse processo generativo, podemos escrever a verossimilhança como:

$$
\log p(r_{t+1} \mid \mathcal{F}_t)
=
\int
\left(
\prod_{i=1}^{N_{t+1}}
p(r_{i,t+1} \mid z_{t+1}, \mathcal{F}_t)
\right)
p(z_{t+1})
\, dz_{t+1}
$$

$p(r_{t+1} \mid \mathcal{F}_t)$ representa a verossimilhança, ou seja:

> “A probabilidade de observar exatamente esse conjunto de retornos no dia $t+1$ dado os dados históricos”.

<br>

# 2. Arquitetura do Modelo

---

## Stock Embedder

Modelo que entrega como output:

- $\alpha_{i,t}$
- $\beta_{i,t}$
- $\sigma_{i,t}$
- $\nu_{i,t}$

para cada ativo $i$ no tempo $t$.

---

### Funcionamento

Considera os últimos $l$ (*lookback*) dias de retornos e outras séries temporais.

1. Esses dados são passados primeiramente a um MLP de uma camada que entrega:

$$
h_{1}
$$

2. Em seguida a saída é passada para um **Sequence Model** (LSTM ou Attention) de duas camadas que entrega:

$$
h_{2}
$$

$h_{2}$ basicamente sintetiza toda a informação das séries temporais em um vetor.

3. $h_{2}$ é concatenado com $X^{static}_t$ e processado por outro MLP de duas camadas que entrega:

$$
h_{3}
$$

4. Por fim, os parâmetros são obtidos através das seguintes fórmulas:

$$
\alpha_{i,t} = w_\alpha^{T} h_{3,i,t}
$$

$$
\beta_{i,t} = W_\beta h_{3,i,t}
$$

$$
\sigma_{i,t} = S(w_\sigma^{T} h_{3,i,t})
$$

$$
\nu_{i,t} = S(w_\nu^{T} h_{3,i,t}) + 4
$$

Sendo $S(\cdot)$ a função **softplus** para garantir positividade nos parâmetros.

---

## Decoder

No NeuralFactors, o decoder é o componente responsável por transformar os fatores latentes de mercado em retornos de ações, implementando explicitamente uma estrutura de modelo fatorial linear.

Ele assume que o retorno de cada ação no próximo período segue uma distribuição Student-t cuja média é dada por um intercepto específico da ação somado a uma combinação linear dos fatores latentes.

Inspirado por modelagem fatorial clássica:

$$
p(r_{i,t+1} \mid z_{t+1}, \mathcal{F}_t)
=
p\Big(
r_{i,t+1}
\mid
t_{\nu_{i,t}}
(
\alpha_{i,t} + \beta_{i,t}^{T} z_{t+1},
\sigma_{i,t}
)
\Big)
$$

Ou seja:

$$
r_{i,t+1}
\sim
\text{Student-T}
\big(
\text{média} = \alpha_{i,t} + \beta_{i,t}^{T} z_{t+1},
\text{escala} = \sigma_{i,t},
\text{graus de liberdade} = \nu_{i,t}
\big)
$$

Note que os retornos esperados de um ativo são função tanto de $\alpha_{i,t}$ quanto de $\mu_z$ (média da distribuição dos fatores).

---

## Encoder

Para entender o que o encoder faz, considere a estrutura do modelo:

1. Primeiro sorteamos os fatores latentes
2. Depois geramos os retornos das ações a partir desses fatores

No treinamento ocorre o contrário:

1. Observamos os retornos históricos
2. Tentamos inferir quais fatores latentes provavelmente geraram esses retornos

A função do **Encoder**, portanto, é inferir os fatores latentes do período seguinte a partir dos dados históricos observados.

---

### Posterior dos Fatores

$$
q(z \mid r, \mathcal{F})
=
p(z \mid r, \mathcal{F})
\quad
\text{(Exact posterior)}
\tag{7}
$$

$$
=
p\left(
z
\mid
\mathcal{N}
\left(
\Sigma_{z|B}
(
\Sigma_z^{-1}\mu_z + B^T \Sigma_x^{-1}(r-\alpha)
),
\Sigma_{z|B}
\right)
\right)
\tag{8}
$$

$$
=
p\left(
z
\mid
\mathcal{N}
(
\mu_{z|B,r},
\Sigma_{z|B}
)
\right)
\tag{9}
$$

---

## Aproximação Normal (Moment Matching)

Embora o modelo use Student-t no prior e no decoder, para derivar o posterior aproximamos essas distribuições por Normais usando **moment matching**.

Motivação:

- Prior usa Student-t
- Likelihood usa Student-t
- Student-t × Student-t não gera forma fechada simples
- Normal × Normal ⇒ Normal (conjugação)

Assim obtemos uma forma fechada para o posterior.

---

## Covariância Posterior

$$
\Sigma_{z|B}
=
(\Sigma_z^{-1} + B^T \Sigma_x^{-1} B)^{-1}
$$

Onde:

- $\Sigma_z$ é a covariância prévia dos fatores
- $\Sigma_z^{-1}$ representa a **precisão** (informação prévia)
- $\Sigma_x$ é diagonal com $\sigma_i^2$
- $B$ é a matriz de exposições fatoriais

Interpretação:

- $\Sigma_z^{-1}$ → informação prévia
- $B^T \Sigma_x^{-1} B$ → informação trazida pelos dados

Informação total = informação prévia + informação dos dados.

---

## Média Posterior

$$
\mu_{z|B,r}
=
\Sigma_{z|B}
\left(
\Sigma_z^{-1}\mu_z
+
B^T \Sigma_x^{-1}(r-\alpha)
\right)
$$

Equilíbrio entre:

1. Força do prior
2. Força dos dados

- $(r-\alpha)$ remove o componente idiossincrático
- $\Sigma_x^{-1}$ pondera pela precisão
- $B^T$ projeta retornos no espaço dos fatores
- $\Sigma_{z|B}$ ajusta pela nova incerteza

---

## Conclusão Estrutural

Equações (8) e (9) descrevem:

Uma regressão linear bayesiana ponderada onde:

- $B$ conecta ações aos fatores
- $\Sigma_x$ controla o peso de cada ação
- $\Sigma_z$ controla a regularização
- O resultado é uma distribuição Normal sobre os fatores

O encoder não é uma rede neural.

Ele é a solução analítica desse sistema.

# 3. Treinamento

---

## 3.1 Features utilizadas (replicação)

### Preços

- Preços de fechamento (para cálculo dos retornos)

---

### Indicadores de Mercado / Valuation

- PE Ratio  
- Price-to-Book Ratio  
- Free Cash Flow Yield  
- Equity Shares Outstanding  

---

### Indicadores Contábeis (Trimestrais)

- Cash Ratio  
- Return on Assets (ROA)  
- Total Debt to Total Assets  

---

### Outras Variáveis Utilizadas

- Volume negociado  

#### Indicadores macroeconômicos:

- **VIX** – volatilidade implícita esperada no mercado acionário dos EUA  
- **MXEF** – retorno do índice de mercados emergentes global  
- **BCOMTR** – retorno de commodities diversificadas  
- **IFIX** – índice de fundos imobiliários brasileiros  
- **SW002766 Curncy** – taxa de câmbio  
- **SPUHYBDT** – high yield (títulos de alto rendimento)  
- **IDEBB3** – índice de crédito doméstico brasileiro  

---

## 3.2 CIWAE Loss

O modelo quer aprender a distribuição conjunta dos retornos do dia seguinte dado o histórico.

Treinar o modelo significa maximizar a log-verossimilhança:

$$
\log p(r)
$$

Porém, o modelo é latente:

1. Primeiro sorteamos fatores:

$$
z_{t+1} \sim p(z)
$$

2. Depois geramos retornos:

$$
r_{t+1} \sim p(r \mid z)
$$

Logo:

$$
p(r) = \int p(r \mid z) p(z) dz
$$

A integral é de alta dimensão e não possui forma fechada.

---

## 3.3 Surge a ideia de usar VAE

O VAE introduz uma distribuição auxiliar:

$$
q(z \mid r)
$$

E usamos a identidade:

$$
\log p(r)
=
\log
\int
q(z \mid r)
\frac{p(r \mid z)p(z)}{q(z \mid r)}
dz
$$

Sabemos que:

$$
\int q(z \mid r) f(z) dz = \mathbb{E}_{q(z \mid r)}[f(z)]
$$

Então:

$$
\log p(r)
=
\log
\mathbb{E}_{q(z \mid r)}
\left[
\frac{p(r \mid z)p(z)}{q(z \mid r)}
\right]
$$

---

## 3.4 Desigualdade de Jensen

Como o log é côncavo:

$$
\log(\mathbb{E}[X]) \ge \mathbb{E}[\log X]
$$

Aplicando Jensen:

$$
\log p(r)
\ge
\mathbb{E}_{q(z \mid r)}
\left[
\log
\frac{p(r \mid z)p(z)}{q(z \mid r)}
\right]
$$

Expandindo:

$$
\mathbb{E}_{q(z \mid r)}
[
\log p(r \mid z)
+
\log p(z)
-
\log q(z \mid r)
]
$$

Isso é o **ELBO** (Evidence Lower Bound).

Maximizar o ELBO aproxima maximizar o log-likelihood.

---

## 3.5 IWAE

Voltamos à identidade original:

$$
\log p(r)
=
\log
\mathbb{E}_{q(z \mid r)}
\left[
\frac{p(r \mid z)p(z)}{q(z \mid r)}
\right]
$$

A ideia do IWAE é usar K amostras:

Amostramos:

$$
z_1, z_2, ..., z_K \sim q(z \mid r)
$$

Aproximamos a esperança por:

$$
\frac{1}{K}
\sum_{k=1}^{K}
\frac{p(r \mid z_k)p(z_k)}{q(z_k \mid r)}
$$

Colocando dentro do log:

$$
\log p(r)
\ge
\mathbb{E}
\left[
\log
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\frac{p(r \mid z_k)p(z_k)}{q(z_k \mid r)}
\right)
\right]
$$

Isso é o **IWAE bound**.

Quando K aumenta, o bound fica mais apertado.

---

## 3.6 CIWAE (Conditional IWAE)

No NeuralFactors modelamos:

$$
p(r_{t+1} \mid \mathcal{F}_t)
$$

Então tudo fica condicionado em $\mathcal{F}_t$:

$$
\log p(r \mid \mathcal{F})
\ge
\mathbb{E}
\left[
\log
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\frac{p(r \mid z_k, \mathcal{F})p(z_k \mid \mathcal{F})}{q(z_k \mid r, \mathcal{F})}
\right)
\right]
$$

É o mesmo IWAE, mas com condicionamento.

---

## 3.7 Hiperparâmetros e Configurações

- Todas as camadas ocultas têm tamanho 256  
- Dropout de 0.25  
- Treinado por 200.000 steps  
- Otimizador Adam  
  - Learning rate: $10^{-4}$  
  - Weight decay (L2): $10^{-6}$  
  - Batch size: 1  
- IWAE loss com $K=20$  
- Polyak Averaging a partir de 50% do treinamento  
- Validation loss computada a cada 1000 steps  
- Seleção do melhor modelo via menor validation loss  
- Implementado em PyTorch com PyTorch Lightning  

---

# 4. Teste e Validação

Para inferência/validação, não precisamos usar o Encoder.  
Simplesmente amostramos da distribuição prior.

---

## Negative Log-Likelihood (NLL)

Mede o quão bem o modelo atribui probabilidade aos retornos observados.

O modelo aprende:

$$
p(r_{t+1} \mid \mathcal{F}_t)
$$

Para cada dia:

$$
\text{Log-Likelihood}
=
\log p(r_{t+1} \mid \mathcal{F}_t)
$$

Como o modelo é latente:

$$
p(r \mid \mathcal{F})
=
\int
p(r \mid z, \mathcal{F}) p(z \mid \mathcal{F}) dz
$$

Essa integral é aproximada via CIWAE.

A métrica reportada é:

$$
\text{NLL} = -\log p(r \mid \mathcal{F})
$$

---

### NLL Joint

No artigo aparece:

$$
\text{NLL}_{joint,t}
=
\frac{1}{N_t}
\log p\big(\{r_{i,t}\}_{i=1}^{N_t} \mid \mathcal{F}_t\big)
$$

Isso significa:

- A probabilidade conjunta de todos os retornos do dia
- Normalizada pelo número de ativos

Mede a qualidade do modelo como modelo multivariado.

---

## Covariance

Erro na matriz de covariância. Mede o quão bem o modelo prevê a matriz de covariância dos retornos.

No NeuralFactors, a covariância pode ser calculada em forma fechada:

$$
\text{Cov}(r)
=
\Sigma_x
+
B \Sigma_z B^{T}
$$

Onde:

- $\Sigma_x$ → risco idiossincrático (diagonal)
- $B \Sigma_z B^T$ → risco sistêmico via fatores

É a decomposição clássica do modelo fatorial.

---

### Covariância Empírica

Calculada via janela móvel:

$$
\hat{\Sigma}_{emp}
=
\frac{1}{T-1}
\sum_{t=1}^{T}
(r_t - \bar{r})(r_t - \bar{r})^T
$$

---

### Métrica Utilizada

Normalmente usa-se MSE:

$$
\text{MSE}
=
\frac{1}{N^2}
\sum_{i,j}
\left(
\Sigma_{model,ij}
-
\Sigma_{emp,ij}
\right)^2
$$

Quanto menor o MSE, melhor o modelo captura as correlações.

Também é um indicativo de que a estrutura fatorial está funcionando.

---

## Calibration Error (VaR Analysis)

Como o modelo calcula VaR?

1. Amostra muitos retornos simulados do modelo  
2. Calcula o quantil $\alpha$ dessas amostras  

Se o modelo é bem calibrado:

- Em 5% dos dias reais  
- O retorno observado deve ser pior que o VaR\_{0.05}

Erro de calibração:

$$
\text{Error}
=
\left|
\text{Probabilidade empírica}
-
\alpha
\right|
$$

Exemplo:

Se $\alpha = 0.05$ e empiricamente deu 0.08:

$$
\text{Erro} = 0.03
$$

Boa calibração significa:

- O modelo estima bem risco extremo
- As caudas estão corretas
- A distribuição está bem especificada

---

## Backtest / Portfolio Optimization

Em cada dia $t$, o modelo produz a matriz de covariância prevista para o dia seguinte:

$$
\Sigma_{t+1}
=
\Sigma_x
+
B_t \Sigma_z B_t^{T}
$$

Isso é a previsão de risco conjunto entre todas as ações.

- $\Sigma_x$ = risco idiossincrático  
- $B_t \Sigma_z B_t^T$ = risco sistêmico via fatores  

Usamos isso para montar a carteira de mínima variância.

Ela minimiza:

$$
\text{Var}(R_p)
=
w^{T}
\Sigma_{t+1}
w
$$

onde:

- $w$ = vetor de pesos  
- $\Sigma_{t+1}$ = matriz de risco prevista  

Sujeito a:

$$
\sum_i w_i = 1
$$

(opcionalmente $w_i \ge 0$ no caso long-only)

---

Analisamos o resultado da estratégia de carteiras de mínima variância diárias e verificamos:

- Sharpe ratio  
- Volatilidade  
- Drawdowns  

Comparando com estratégias alternativas.