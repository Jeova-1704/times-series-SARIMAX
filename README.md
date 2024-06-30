# Previsão de Séries Temporais com SARIMAX

Este projeto demonstra a previsão de séries temporais usando o modelo SARIMAX. Foram escolhidas duas séries temporais diferentes para a análise:

1. **Consumo e produção diária de eletricidade na Romênia** (Previsão da produção de energia hidrelétrica).
2. **Consumo de energia no Hawaii em todos os setores** (Previsão do consumo de energia).

## Requisitos

Para executar este projeto, é necessário instalar as seguintes bibliotecas:
- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `scikit-learn`

```python
pip install pandas numpy matplotlib statsmodels scikit-learn
```

## Importação das Bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from itertools import product
```

## Configuração do Tamanho das Imagens

```python
rcParams['figure.figsize'] = 15, 6
```

## 1. Consumo e Produção Diária de Eletricidade na Romênia

### Conjunto de Dados

- **Contexto:** Consumo e produção de eletricidade diária por tipo na Romênia durante 5,5 anos.
- **Objetivo:** Prever a produção de energia hidrelétrica.

### Carregamento e Preparação dos Dados

```python
path = '/content/electricity_Consumption_Productioction.csv'
dados = pd.read_csv(path)
dados['Date'] = pd.to_datetime(dados['Date'])
dados = dados.set_index('Date')
dados = pd.DataFrame(index=dados.index, data=dados['Consumption'])
dados = dados.resample('W').sum()
dados.plot()
```

### Separação dos Dados em Treino e Teste

```python
train, test = dados.loc['2019':'2022'], dados.loc['2023':]
plt.plot(train, label = 'train')
plt.plot(test, label = 'test')
plt.legend(loc = 'best')
plt.show()
```

### Análise de Autocorrelação e Parcial Autocorrelação

```python
def acf_pacf(x, qtd_lag):
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(221)
    fig = sm.graphics.tsa.plot_acf(x, lags=qtd_lag, ax=ax1)
    ax2 = fig.add_subplot(222)
    fig = sm.graphics.tsa.plot_pacf(x, lags=qtd_lag, ax=ax2)
    plt.show()
acf_pacf(train, 40)
```

### Teste de Estacionaridade

```python
def teste_estacionaria(serie):
    p_value = adfuller(serie)[1]
    print(p_value)
    if p_value < 0.05:
        print("Série estacionária")
    else:
        print("Série não estacionária")
teste_estacionaria(train)
```

### Diferenciação da Série

```python
train.diff(1).dropna().plot()
acf_pacf(train, 36)
```

### Criação do Modelo SARIMAX

```python
model = SARIMAX(train, order=(1, 1, 7)).fit()
prev_train = model.predict()
plt.plot(prev_train, label = 'predict')
plt.plot(train, label = 'target')
plt.legend(loc = 'best')
plt.show()
```

### Previsão e Avaliação do Modelo

```python
MAPE(train, prev_train)

prev_test = model.forecast(len(test))
plt.plot(test, label = 'target')
plt.plot(prev_test, label = 'predict')
plt.legend()
plt.show()

MAPE(test, prev_test)
```

### Ajuste do Modelo com Parâmetros Sazonais

```python
model = SARIMAX(train, order=(1, 1, 7), seasonal_order=(1, 1, 1, 52)).fit()
train_prev = model.predict()
test_prev = model.forecast(len(test))

plt.plot(train, label = 'target')
plt.plot(train_prev, label = 'predict')
plt.legend()
plt.show()

MAPE(train, train_prev)

plt.plot(test, label = 'target')
plt.plot(test_prev, label = 'predict')
plt.legend()
plt.show()

MAPE(test, test_prev)
```

### Análise dos Resíduos

```python
residuals = model.resid
sm.qqplot(residuals, line='s')
plt.title("Gráfico Q-Q Normal dos Resíduos")
plt.show()
```

## 2. Consumo de Energia no Hawaii

### Conjunto de Dados

- **Contexto:** Geração líquida de eletricidade nos EUA, discriminada por estado e setor.
- **Objetivo:** Prever o consumo de energia no Hawaii em todos os setores.

### Carregamento e Preparação dos Dados

```python
path = '/content/electricity_data.csv'
dados = pd.read_csv(path)
dados = dados.rename(columns={'Unnamed: 0': 'Date'})
dados['Date'] = pd.to_datetime(dados['Date'])
dados = dados.set_index('Date')
dados = pd.DataFrame(index=dados.index, data=dados['Hawaii : all sectors'])
dados.plot()
```

### Separação dos Dados em Treino e Teste

```python
train, test = dados.loc[:'2015'], dados.loc['2016':]
plt.plot(train, label = 'train')
plt.plot(test, label = 'test')
plt.legend(loc = 'best')
plt.show()
```

### Análise de Autocorrelação e Parcial Autocorrelação

```python
acf_pacf(train, 40)
```

### Teste de Estacionaridade

```python
teste_estacionaria(train)
```

### Diferenciação da Série

```python
train.diff(1).dropna().plot()
acf_pacf(train, 36)
```

### Criação do Modelo SARIMAX

```python
model = SARIMAX(train, order=(2, 1, 3)).fit()
prev_train = model.predict()
plt.plot(prev_train, label = 'predict')
plt.plot(train, label = 'target')
plt.legend(loc = 'best')
plt.show()
```

### Previsão e Avaliação do Modelo

```python
MAPE(train, prev_train)

prev_test = model.forecast(len(test))
plt.plot(test, label = 'target')
plt.plot(prev_test, label = 'predict')
plt.legend()
plt.show()

MAPE(test, prev_test)
```

### Ajuste do Modelo com Parâmetros Sazonais

```python
model = SARIMAX(train, order=(2, 1, 3), seasonal_order=(1, 0, 2, 12)).fit()
train_prev = model.predict()
test_prev = model.forecast(len(test))

plt.plot(train, label = 'target')
plt.plot(train_prev, label = 'predict')
plt.legend()
plt.show()

MAPE(train, train_prev)

plt.plot(test, label = 'target')
plt.plot(test_prev, label = 'predict')
plt.legend()
plt.show()

MAPE(test, test_prev)
```

### Análise dos Resíduos

```python
residuals = model.resid
sm.qqplot(residuals, line='s')
plt.title("Gráfico Q-Q Normal dos Resíduos")
plt.show()
```

## Seleção dos Melhores Parâmetros

```python
p_values = [0, 1]
d_values = [0]
q_values = [0, 1, 2, 3, 4, 5, 6, 7]
P_values = [0]
D_values = [0, 1]
Q_values = [0]
m_values = [0, 12, 52]

melhor_modelo = None
melhor_MAPE = np.inf

for p, d, q, P, D, Q, m in product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
        results = model.fit()
        forecast = results.forecast(steps=len(train))
        mape = MAPE(train, forecast)
        if mape < melhor_MAPE:
            melhor_MAPE = mape
            melhor_modelo = model
    except:
        continue
```

## Explicações

- **Série Temporal:** Sequência de dados coletados em intervalos regulares ao longo do tempo.
- **SARIMAX:** Modelo autorregressivo integrado de médias móveis sazonais com fatores exógenos. É usado para modelar séries temporais com sazonalidade.
- **Estacionaridade:** Uma série temporal é estacionária se suas propriedades estatísticas, como média e variância, são constantes ao longo do tempo.
- **Diferenciação:** Processo de transformar uma série temporal não estacionária em estacionária subtraindo o valor atual pelo valor anterior.
- **Autocorrelação:** Medida de correlação entre a série temporal e uma versão defasada dela mesma.
- **MAPE (Mean Absolute Percentage Error):** Métrica de avaliação que calcula a precisão da

 previsão medindo a porcentagem de erro absoluto médio.

## Resultados e Conclusões

Os resultados das previsões foram avaliados utilizando o MAPE. Após ajustar os modelos SARIMAX com parâmetros sazonais, os erros de previsão foram reduzidos. Os resíduos dos modelos foram analisados através de gráficos Q-Q para garantir que os resíduos seguissem uma distribuição normal.

Os modelos ajustados mostraram-se eficazes para prever as séries temporais escolhidas, e os parâmetros foram selecionados com base na análise de autocorrelação, parcial autocorrelação e testes de estacionaridade.

---
