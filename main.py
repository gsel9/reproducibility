import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import load_data
from train import select_hyperparameters, evaluate_model


def main():

	X, y = load_data("path_to_data")
	# 1) Select model hyperparameters 
	# 2) Train model using training data with optimal parameters 
	# 3) Compute

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	hparam_grid = {"C": 10 ** np.linspace(-2, 2, 10)}

	# Initialise model.
	model = LogisticRegression(random_state=42)

	opt_model = select_hyperparameters(model, X_train, y_train, hparam_grid)

	evaluate_model(model, X_test, y_test)

	# TODO: Make sure to save vectors of y_pred and y_test in case we need to compute 
	# additional performance measures at a later stage"!!!
	# 


if __name__ == "__main__":
	main()
