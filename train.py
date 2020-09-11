import numpy as np 
import pandas as pd 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer


def select_hyperparameters(model, hparams, X_train, y_train, path_to_cvresults):
	"""Select model hyperparameters"""
	
	# Run cross-validated parameter search.
	grid_search = GridSearchCV(estimator=model, param_grid=hparams, cv=10, refit=True,
							   scoring=make_scorer(matthews_corrcoef))
	grid_search.fit(X_train, y_train)

	df = pd.DataFrame(grid_search.cv_results_)
	df.to_csv(path_to_cvresults)

	return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, path_to_preds):

	# Apply model to test set.
	y_pred = model.predict(X_test)

	np.save(f"{path_to_preds}_pred.npy", y_pred)
	np.save(f"{path_to_preds}_true.npy", y_test)
