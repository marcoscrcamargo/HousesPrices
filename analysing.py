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
# Linear Ridge
alphas = [1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
			for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()

# Linear Lasso
alphas = [1, 0.1, 0.001, 0.0005]
cv_lasso = [rmse_cv(Lasso(alpha=alpha)).mean()
			for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)

# XGBoost Model
dtrain = xgb.DMatrix(x_train, label = y)
#dtest = xgb.DMatrix(x_test)

# testar outros parametros pra ver se a acuracia melhora
# (melhores resultados)
params = {"max_depth":2, "eta":0.1}
cv_xgb = xgb.cv(params, dtrain,  num_boost_round=500, nfold=K_FOLDS, early_stopping_rounds=100)
# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# Comparando os resultados do kfold
print(cv_xgb['test-rmse-mean'].idxmin(), cv_xgb['test-rmse-mean'].min(), sep='\t')
print(cv_ridge.idxmin(), cv_ridge.min(), sep='\t')
print(cv_lasso.idxmin(), cv_lasso.min(), sep='\t')


# Imprimindo gráfico com os melhores resultados.
# data = np.array([cv_ridge.min(), cv_lasso.min(), cv_xgb['test-rmse-mean'].min()])
# indices = ['Ridge', 'Lasso', 'XGBoost']
# plt.bar(indices, 1-data, width= 0.2)
# plt.show()


# Criando modelos para a geração do resultado
# Modelo Lasso
	# alpha = melhor resultado obtido acima
model_lasso = Lasso(alpha=0.0005).fit(x_train, y)

# Modelo XBG
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(x_train, y)

# Modelo Ridge
model_ridge = Ridge(alpha=10).fit(x_train, y)

# Realizando as predições
xgb_preds = np.expm1(model_xgb.predict(x_test))
lasso_preds = np.expm1(model_lasso.predict(x_test))
ridge_preds = np.expm1(model_ridge.predict(x_test))

# Visualização dos resultados.
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds, "ridge":ridge_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

# Combinando resultados
preds = 0.7*lasso_preds + 0.3*xgb_preds

# Exportando CSV para submissão.
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge_sol.csv", index = False)

