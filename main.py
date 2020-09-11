import numpy as np 
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

from utils import plot_confusion_matrix, plot_validation_scores, load_data
from train import select_hyperparameters, evaluate_model


def experiment():

	X, y = load_data("./data/wdbc.data")
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	scaler = StandardScaler()
	X_train_std = scaler.fit_transform(X_train)
	X_test_std = scaler.transform(X_test)

	hparam_grid = {"C": 10 ** np.linspace(-3, 1, 10)}

	opt_model = select_hyperparameters(LogisticRegression(random_state=42), 
									   hparam_grid, 
									   X_train_std, y_train, 
									   "cvresults.csv")

	evaluate_model(opt_model, X_test_std, y_test, "best_model")
	

def main():

	experiment()
	
	cvresults = pd.read_csv("cvresults.csv", index_col=0)
	opt_score_idx = np.argmax(cvresults.mean_test_score)
	print("Optimal log_10 C:", np.log10(cvresults.param_C.values[opt_score_idx]))
	print("Validation MCC for optimal C:", max(cvresults.mean_test_score))

	plot_validation_scores(cvresults, "validation.pdf")
		
	y_test = np.load("best_model_true.npy")
	y_pred = np.load("best_model_pred.npy")
	#plot_confusion_matrix(y_test, y_pred, "confusion.pdf")
	print("Test MCC:", matthews_corrcoef(y_test, y_pred))


if __name__ == "__main__":
	main()
