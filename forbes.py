import numpy as np
import pandas as pd

from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from scipy.special import boxcox1p

from xgboost import XGBRegressor

# from sklearn.linear_model import Ridge, Lasso
# from sklearn.neural_network import MLPRegressor
# from xgboost import XGBRegressor

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# Defining data paths.
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SOLUTION_PATH = "data/solution.csv"

# Defining model weights.
LASSO_WEIGHT = 0.5
ENET_WEIGHT = 0.35
XGB_WEIGHT = 0.15
# LGB_WEIGHT = 0.15

# Removes outliers from the training set.
def remove_outliers(train):
	# Manually removing outliers.
	return train.drop(train[(train["GrLivArea"] > 4000) & (train['SalePrice'] < 300000)].index, inplace = False)

# Fills missing values to make data more consistent.
def fill_missing_values(data):
	# Filling not applicable categorical attributes with None.
	data["PoolQC"].fillna("None", inplace = True) # No pool.
	data["MiscFeature"].fillna("None", inplace = True) # No misc. feature.
	data["Alley"].fillna("None", inplace = True) # No alley access.
	data["Fence"].fillna("None", inplace = True) # No fence.
	data["FireplaceQu"].fillna("None", inplace = True) # No fireplace.
	data["GarageType"].fillna("None", inplace = True) # No garage.
	data["GarageFinish"].fillna("None", inplace = True) # No garage.
	data["GarageQual"].fillna("None", inplace = True) # No garage.
	data["GarageCond"].fillna("None", inplace = True) # No garage.
	data["BsmtQual"].fillna("None", inplace = True) # No basement.
	data["BsmtCond"].fillna("None", inplace = True) # No basement.
	data["BsmtExposure"].fillna("None", inplace = True) # No basement.
	data["BsmtFinType1"].fillna("None", inplace = True) # No basement.
	data["BsmtFinType2"].fillna("None", inplace = True) # No basement.
	data["MasVnrType"].fillna("None", inplace = True) # No masonry.
	data["MSSubClass"].fillna("None", inplace = True) # Probably no building class.

	# Filling not applicable numerical attributes with 0.
	data["BsmtFinSF1"].fillna(0, inplace = True) # No basement. Numerical None.
	data["BsmtFinSF2"].fillna(0, inplace = True) # No basement. Numerical None.
	data["BsmtUnfSF"].fillna(0, inplace = True) # No basement. Numerical None.
	data["TotalBsmtSF"].fillna(0, inplace = True) # No basement. Numerical None.
	data["BsmtFullBath"].fillna(0, inplace = True) # No basement. Numerical None.
	data["BsmtHalfBath"].fillna(0, inplace = True) # No basement. Numerical None.
	data["GarageYrBlt"].fillna(0, inplace = True) # No garage. Numerical None.
	data["GarageArea"].fillna(0, inplace = True) # No garage, no area.
	data["GarageCars"].fillna(0, inplace = True) # No garage, no cars.
	data["MasVnrArea"].fillna(0, inplace = True) # No masonry. Numerical None.

	# Filling with "RL" which is by far the most common value.
	data["MSZoning"].fillna(data["MSZoning"].mode()[0], inplace = True)

	# Filling with "SBrkr" which is the standard electrical system for most houses.
	data["Electrical"].fillna(data["Electrical"].mode()[0], inplace = True)

	# Filling with "TA" which is the typical quality for most houses.
	data["KitchenQual"].fillna(data["KitchenQual"].mode()[0], inplace = True)

	# Both attributes have only one missing value for the training set, so we fill them in with the most common string. 
	data["Exterior1st"].fillna(data["Exterior1st"].mode()[0], inplace = True)
	data["Exterior2nd"].fillna(data["Exterior2nd"].mode()[0], inplace = True)

	# Filling with "WD" which is the conventional sale type (most common).
	data["SaleType"].fillna(data["SaleType"].mode()[0], inplace = True)

	# Data description says that NA means Typical for this attribute.
	data["Functional"].fillna("Typ", inplace = True)

	# Dropping Utilities attibute since every value is "AllPub", except for 2 NAs and 1 "NoSewa".
	# The former is actually in the training set, so this attribut won't help in predictive modelling.
	data.drop("Utilities", axis = 1, inplace = True)

	# Filling LotFrontage with Neighborhood's median.
	data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

	return data

# Categorizes data that should be categorical instead of numerical.
def categorize(data):
	data["OverallCond"] = data["OverallCond"].astype(str) # Overall Condition.
	data["MSSubClass"] = data["MSSubClass"].astype(str) # Dwelling type.
	data["YrSold"] = data["YrSold"].astype(str) # Year sold.
	data["MoSold"] = data["MoSold"].astype(str) # Month sold.

	return data

# (Maybe should be reworked) Label encodes some attributes to be able to extract useful information from their ordering sets.
def label_encode(data):
	# Defining columns to label encode.
	columns = ["FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "ExterQual", "ExterCond","HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1", "BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope", "LotShape", "PavedDrive", "Street", "Alley", "CentralAir", "MSSubClass", "OverallCond", "YrSold", "MoSold"]

	# For each attribute to label encode.
	for attribute in columns:
		label = LabelEncoder()
		label.fit(list(data[attribute].values))
		data[attribute] = label.transform(list(data[attribute].values))

	return data

# Reduces skewness of attributes.
def reduce_skewness(data, threshold = 0.75):
	# Retrieving numerical attributes.
	numerical_data = data.dtypes[data.dtypes != "object"].index

	# Retrieving skewness of data.
	skewness = pd.DataFrame({"Skew" : data[numerical_data].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)})

	# Retrieving index of skewed attributes.
	skewed_features = skewness[abs(skewness) > threshold].index

	# Defining lambda.
	lam = 0.33

	# For each skewed feature
	for feature in skewed_features:
		# Apply boxcox1p.
		data[feature] = boxcox1p(data[feature], lam)

	return data

# (Maybe remove rows with many missing attributes) Pre-process data.
def pre_process(train, test):
	# Dropping the ID column.
	train = train.drop("Id", axis = 1, inplace = False)
	test = test.drop("Id", axis = 1, inplace = False)

	# Removing outliers.
	train = remove_outliers(train)

	# Applying log(x + 1) to the target attribute to make it's distribution less skewed.
	train["SalePrice"] = np.log1p(train["SalePrice"])

	# Concatenating all data.
	all_data = pd.concat((train, test), axis = 0, sort = True).reset_index(drop = True)
	all_data.drop("SalePrice", axis = 1, inplace = True)

	# Filling missing values to make data consistent.
	all_data = fill_missing_values(all_data)

	# Transforming some numerical attributes to categorical.
	all_data = categorize(all_data)

	# Label encoding categorical attributes that may hold useful information in their order.
	all_data = label_encode(all_data)

	# (Choice) Adding a new feature which represents the total area of the basement + first floor + second floor,
	# since we know that area is very important to determine house prices.
	all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]

	# Reducing skewness of attributes.
	all_data = reduce_skewness(all_data)

	# Getting dummies for categorical features.
	all_data = pd.get_dummies(all_data)

	return (all_data[:train.shape[0]], train["SalePrice"], all_data[train.shape[0]:])

# Validation function.
def evaluate(model, train_x, train_y, k = 5, seed = 1337):
	# Shuffling and retrieving k folds.
    folds = KFold(k, shuffle = True, random_state = seed).get_n_splits(train_x.values)

    # Retrieving and returning the root mean squared error.
    return np.mean(np.sqrt(-cross_val_score(model, train_x, train_y, scoring = "neg_mean_squared_error", cv = folds)))

# Predicts the Sale Price for unseen data.
def predict(models, weights, test_x):
	# Creating prediction results.
	predictions = np.array([np.expm1(model.predict(test_x)) for model in models])

	# Returning the weighted average of the predictions.
	return pd.DataFrame({"SalePrice" : np.dot(weights, predictions)})

# Builds models on training data.
def build(train_x, train_y):
	# Declaring an empty array of models.
	models = []

	# Filling array with models.
	models.append(make_pipeline(RobustScaler(), Lasso(alpha = 0.0005)).fit(train_x, train_y))
	models.append(make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005, l1_ratio = 0.9)).fit(train_x, train_y))
	models.append(XGBRegressor(n_estimators = 360, max_depth = 3, learning_rate = 0.1).fit(train_x, train_y))
	# models.append(lgb.LGBMRegressor(objective = 'regression', num_leaves = 5, learning_rate = 0.05, n_estimators = 720).fit(train_x, train_y))

	return models

def main():
	# Reading data.
	train = pd.read_csv(TRAIN_PATH) # Reading the training set.
	test = pd.read_csv(TEST_PATH) # Reading the test set.

	# Pre-processing data.
	train_x, train_y, test_x = pre_process(train, test)

	# Building models on training data.
	models = build(train_x, train_y)

	# Weights for each model.
	weights = np.array([LASSO_WEIGHT, ENET_WEIGHT, XGB_WEIGHT])

	# Printing evaluation results.
	for i in range(len(models)):
		print("RMSE for model[%d] = %.4f" % (i, evaluate(models[i], train_x, train_y)))

	# Predicting Sale Prices for the test set.
	test_y = predict(models, weights, test_x)

	# Generating CSV with the predictions.
	solution = pd.concat((test["Id"], test_y), axis = 1, sort = True)
	solution.to_csv(SOLUTION_PATH, index = False)

if __name__ == "__main__":
	main()