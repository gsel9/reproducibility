from sklearn.model_selection import GridSearchCV


def select_hyperparameters(model, hparams, X_train, y_train):
	"""Select model hyperparameters"""
	
	# Run cross-validated parameter search.
	grid_search = GridSearchCV(estimator=model, param_grid=hparams, 	
							   cv=10, refit=True)
	grid_search.fit(X_train, y_train)

	return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):

	# Apply model to test set.
	y_pred = model.predict(X_test) 	

	# Compute some performance statistics.
