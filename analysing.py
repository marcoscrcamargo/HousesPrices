import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Parametro K do K-fold
K_FOLDS = 10


# Caminho dos dados.
DTRAIN_PATH = "data/train.csv"
DTEST_PATH = "data/test.csv"

# Lendo dados.
train = pd.read_csv(DTRAIN_PATH)
test = pd.read_csv(DTEST_PATH)

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

def rmse_cv(model):
	return np.sqrt(-cross_val_score(model, x_train, y,
		scoring="neg_mean_squared_error", cv=K_FOLDS))

# K-Fold Cross Validation to evaluete models.
# Linear Ridge Model
alphas = [1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
			for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()

# Linear Lasso Model
alphas = [1, 0.1, 0.001, 0.0005]
cv_lasso = [rmse_cv(Lasso(alpha=alpha)).mean()
			for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)

# xgboost Model
dtrain = xgb.DMatrix(x_train, label = y)
dtest = xgb.DMatrix(x_test)

params = {"max_depth":2, "eta":0.1}
cv_xgb = xgb.cv(params, dtrain,  num_boost_round=500, nfold=K_FOLDS, early_stopping_rounds=100)
# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# Comparando os resultados do kfold
print(cv_xgb['test-rmse-mean'].idxmin(), cv_xgb['test-rmse-mean'].min(), sep='\t')
print(cv_ridge.idxmin(), cv_ridge.min(), sep='\t')
print(cv_lasso.idxmin(), cv_lasso.min(), sep='\t')

colors = [ 'g', 'yellow', 'k', 'maroon']
data = [cv_ridge.min(), cv_lasso.min(), cv_xgb['test-rmse-mean'].min()]
x = ['Ridge', 'Lasso', 'XGBoost']
plt.bar(x, data, color=colors)
plt.show()


