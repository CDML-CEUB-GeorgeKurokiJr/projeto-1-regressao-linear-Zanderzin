import pandas as pd

#ler dataset
df = pd.read_csv('data/iris')
print(df.head())

#nomear colunas em português
df.columns = ['comprimento_sepala', 'largura_sepala', 'comprimento_petala', 'largura_petala', 'especie']

#verificar tipos de dados
print(df.dtypes)

#fazer a regressão linear para prever o comprimento da pétala com base no comprimento da sépala
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#separar as variáveis
X = df[['comprimento_sepala']]
y = df['comprimento_petala']

#criar o modelo de regressão linear
model = LinearRegression()

#ajustar o modelo aos dados
model.fit(X, y)

#fazer previsões
y_pred = model.predict(X)

#visualizar os resultados
plt.scatter(X, y, color='blue', label='Dados Reais')
plt.plot(X, y_pred, color='red', label='Linha de Regressão')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Comprimento da Pétala')
plt.title('Regressão Linear: Comprimento da Sépala vs Comprimento da Pétala')
plt.legend()
plt.show()

#ver o coeficiente de determinação (R²)
r2_score = model.score(X, y)
print(f'R² Score: {r2_score:.2f}')

