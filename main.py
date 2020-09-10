import numpy as np 
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import plot_confusion_matrix, plot_validation_scores, load_data
from train import select_hyperparameters, evaluate_model


def main():

	#X, y = load_data("path_to_data")
	from sklearn.datasets import load_breast_cancer
	X, y = load_breast_cancer(return_X_y=True)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	scaler = StandardScaler()
	X_train_std = scaler.fit_transform(X_train)
	X_test_std = scaler.transform(X_test)

	# Define hyperparameter grid.
	hparam_grid = {"C": 10 ** np.linspace(-3, 1, 10)}

	# Initialise model.
	model = LogisticRegression(random_state=42)

	opt_model = select_hyperparameters(model, hparam_grid, X_train_std, y_train, "cvresults.csv")
	
	cv_results = pd.read_csv("cvresults.csv", index_col=0)
	plot_validation_scores(cv_results, "validation.pdf")
	
	evaluate_model(opt_model, X_test_std, y_test, "best_model")

	y_test = np.load("best_model_true.npy")
	y_pred = np.load("best_model_pred.npy")

	plot_confusion_matrix(y_test, y_pred, "confusion.pdf")


if __name__ == "__main__":
	main()
