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
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Caminho dos dados.
DTRAIN_PATH = "data/train.csv"
DTEST_PATH = "data/test.csv"

def reading_data(path):
	return pd.read_csv(path)

def rmse_cv(model, x_train, y_train, k_folds=10):
	return np.sqrt(-cross_val_score(model, x_train, y_train,
		scoring="neg_mean_squared_error", cv=k_folds))

# Removing columns that the number of nans values is greater than bound=0.3
def remove_nan_columns(df, bound=0.3):
	for column_name in df.columns:
		column = df[column_name]
		nan_percentage = column.isnull().sum()/column.size
		if(nan_percentage > bound):
			df = df.drop(columns=[column_name])
	return df

# Removing rows that the number of nans values is greater than bound=0.1 (more than 10 values null)
def remove_nan_rows(df, bound=0.1):
	row_size = df.shape[1]
	for index, row in df.iterrows():
		rownan_percentage = row.isnull().sum()/row_size
		if (rownan_percentage >= bound ):
			df = df.drop(index)
	return df


def pre_processing_data(train_data, test_data):
	# Removendo as linhas com muitos dados faltantes do conjunto de treino.
	train_data = remove_nan_rows(train_data)

	# Concatenando dados de treino e teste para facilitar as operações
	# de pré processamento.
	all_data = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
						test_data.loc[:,'MSSubClass':'SaleCondition']))

	# Removendo as colunas com muitos dados ausentes.
	# all_data = remove_nan_columns(all_data)

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
	# all_data = all_data.fillna(all_data.median())

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
	# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
	# gridsearch = GridSearchCV(model, param_grid=params, scoring="neg_log_loss", n_jobs=-1, cv=kfold)

	gridsearch = GridSearchCV(model, param_grid=params, n_jobs=-1)
	gridresult = gridsearch.fit(train_x, train_y)

	return gridresult

def print_grid_result(name, grid_result):
	print(name + ":")
	# sumarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print("\n")


def grid_search_lasso(train_x, train_y):
	params = {'alpha':(1, 0.1, 0.001, 0.0005)}

	grid_result = grid_search(Lasso(), params, train_x, train_y)
	print_grid_result("Lasso", grid_result)

def grid_search_ridge(train_x, train_y):
	params = {'alpha':(1, 3, 5, 10, 15, 30, 50, 75)}

	grid_result = grid_search(Ridge(), params, train_x, train_y)
	print_grid_result("Ridge", grid_result)

def grid_search_xgb(train_x, train_y):
	params = {"max_depth":(2, 3, 4), "learning_rate":(0.0001, 0.001, 0.01, 0.1, 0.2, 0.3), "n_estimators":(100, 200, 300, 400, 500)}
	# params = {"max_depth":(2,), "learning_rate":(0.1,), "n_estimators":(300,)}

	grid_result = grid_search(XGBRegressor(), params, train_x, train_y)
	print_grid_result("XGBRegressor", grid_result)


def grid_search_mlp(train_x, train_y):
	# params = {"hidden_layer_sizes":((10), (100), (200), (100,100), (200, 200), (10,10,10)), "activation": ("logistic", "tanh", "identity", "relu"), "solver":("lbfgs", "adam"), "max_iter": (1000,) }
	params = {"hidden_layer_sizes":((200, 200),), "activation": ("logistic",), "solver":("lbfgs",), "max_iter": (1000,) }
	grid_result = grid_search(MLPRegressor(), params, train_x, train_y)
	print_grid_result("MLPRegressor", grid_result)

def get_results(train_x, train_y, test_x, test_data):
	# Criando modelos para a geração do resultado
	# # Modelo Lasso
	# print('Lasso(alpha=0.0005):')
	# print(rmse_cv(Lasso(alpha=0.0005), train_x, train_y).mean())
	# # Modelo XBG
	# print('XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1):')
	# print(rmse_cv(XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1), train_x, train_y).mean())
	# # Modelo Ridge
	# model_ridge = Ridge(alpha=10).fit(train_x, train_y)

	model_lasso = Lasso(alpha=0.0005).fit(train_x, train_y)
	model_xgb = XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1).fit(train_x, train_y)
	models = [model_lasso, model_xgb]
	weights = [0.7, 0.3]

	generating_csv(models, weights, test_x, test_data.Id)

def main():
	train_data = reading_data(DTRAIN_PATH)
	test_data = reading_data(DTEST_PATH)
	train_x, train_y, test_x = pre_processing_data(train_data, test_data)

	# grid_search_ridge(train_x, train_y)
	# grid_search_lasso(train_x, train_y)
	# grid_search_mlp(train_x, train_y)
	# grid_search_xgb(train_x, train_y)

	get_results(train_x, train_y, test_x, test_data)



if __name__ == '__main__':
	main()