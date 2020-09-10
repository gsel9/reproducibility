import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#from utils import load_data
from train import select_hyperparameters, evaluate_model


def main():

	X, y = load_data("path_to_data")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	scaler = StandardScaler()
	X_train_std = scaler.fit_transform(X_train)
	X_test_std = scaler.transform(X_test)

	# Define hyperparameter grid.
	hparam_grid = {"C": 10 ** np.linspace(-2, 2, 10)}

	# Initialise model.
	model = LogisticRegression(random_state=42)

	opt_model = select_hyperparameters(model, hparam_grid, X_train_std, y_train, "path_to_cvresults.csv")

	evaluate_model(opt_model, X_test_std, y_test, "best_model")


if __name__ == "__main__":
	main()
