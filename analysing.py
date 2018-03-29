import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew

# Caminho dos dados.
DTRAIN_PATH = "data/train.csv"
DTEST_PATH = "data/test.csv"

# Lendo dados.
train = pd.read_csv(DTRAIN_PATH)
test = pd.read_csv(DTRAIN_PATH)

# Concatenando dados de treino e teste para facilitar as operações
# de pré processamento.
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# Aplicando Log nos preços de venda (valores em distribuição normal).
train["SalePrice"] = np.log1p(train["SalePrice"])
## Para a visualização dos dados
# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
# prices.hist()
# plt.show()

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Calcula o skewness
# For normally distributed data, the skewness should be about 0.
# A skewness value > 0 means that there is more weight in the left tail of the distribution.
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))

# Seleciona os valores com skewness > 0.75
skewed_feats = skewed_feats[skewed_feats > 0.75].index

# Aplicando log nos valores com skewness maior que 0.75
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Converte dados categorigos em dummy indicators
all_data = pd.get_dummies(all_data)

# Preenche os valores em branco com a média.
all_data = all_data.fillna(all_data.mean())

# Dados para treinamento e teste após pré processamento.
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
y = train.SalePrice

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
	return np.sqrt(-cross_val_score(model, x_train, y,
		scoring="neg_mean_squared_error", cv=10))

alphas = np.arange(9, 11, 0.1)
# alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
			for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()

print(cv_ridge.idxmin(), cv_ridge.min())