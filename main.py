# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo  -  9278045
# SCC277							   1/2018
# Kaggle Competition
# House prices advanced regression techniques

import pandas as pd
import numpy as np

from scipy.stats import skew

from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import GridSearchCV


# Caminho dos dados.
DTRAIN_PATH = "data/train.csv"
DTEST_PATH = "data/test.csv"

def reading_data(path):
	return pd.read_csv(path)

def pre_processing_data(train_data, test_data):
	# Concatenando dados de treino e teste para facilitar as operações
	# de pré processamento.
	all_data = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
                      test_data.loc[:,'MSSubClass':'SaleCondition']))
	# Aplicando Log nos preços de venda (valores em distribuição normal).
	train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

	numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

	# Calcula o skewness
	# For normally distributed data, the skewness should be about 0.
	# A skewness value > 0 means that there is more weight in the left tail of the distribution.
	skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna()))

	# Seleciona os valores com skewness > 0.75
	skewed_feats = skewed_feats[skewed_feats > 0.75].index

	# Aplicando log nos valores com skewness maior que 0.75
	all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

	# Converte dados categorigos em dummy indicators
	all_data = pd.get_dummies(all_data)

	# Preenche os valores em branco com a média.
	all_data = all_data.fillna(all_data.mean())

	# Dados para treinamento e teste após pré processamento.
	x_train = all_data[:train_data.shape[0]]
	x_test = all_data[train_data.shape[0]:]
	y_train = train_data.SalePrice

	return (x_train, y_train, x_test)

def generating_csv(models, weights, x_test, ids):
	# Realizando as predições
	preds = np.array([np.expm1(model.predict(x_test)) for model in models])
	# Combinando resultados
	prediction = np.sum(preds.T*weights, axis=1)
	# Exportando CSV para submissão.
	solution = pd.DataFrame({"id":ids, "SalePrice":prediction})
	solution.to_csv("sol.csv", index = False)
	print("CSV file generated.")


def grid_search(model, params, train_x, train_y):
	clf = GridSearchCV(estimator=model, param_grid=params).fit(train_x, train_y)
	return (clf.best_params_, clf.best_score_)

def grid_search_lasso(train_x, train_y):
	params = {'alpha':(1, 0.1, 0.001, 0.0005)}
	best_params = grid_search(Lasso(), params, train_x, train_y)
	print("LASSO:")
	print(best_params)
	return best_params

def grid_search_ridge(train_x, train_y):
	params = {'alpha':(1, 3, 5, 10, 15, 30, 50, 75)}
	best_params = grid_search(Ridge(), params, train_x, train_y)
	print("RIDGE:")
	print(best_params)
	return best_params

def grid_search_xgb():
	pass

def main():
	train_data = reading_data(DTRAIN_PATH)
	test_data = reading_data(DTEST_PATH)
	train_x, train_y, test_x = pre_processing_data(train_data, test_data)

	# Criando modelos para a geração do resultado
	# Modelo Lasso
	model_lasso = Lasso(alpha=0.0005).fit(train_x, train_y)
	# Modelo XBG
	model_xgb = XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1).fit(train_x, train_y)
	# # Modelo Ridge
	# model_ridge = Ridge(alpha=10).fit(train_x, train_y)

	models = [model_lasso, model_xgb]
	weights = [0.7, 0.3]

	generating_csv(models, weights, test_x, test_data.Id)

if __name__ == '__main__':
	main()