import numpy as np
import pandas as pd 

import matplotlib

import seaborn as sns 
import matplotlib.pyplot as plt 


# TODO (Ali): complete this function
def load_data(path_to_data):
	#manually entering columns because the files in the dataset didn't have it
	cols = """id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst"""
	columns=[]
	for c in cols.split(','):
	    if(c.strip()):
	       columns.append(c.strip())

	data = pd.read_csv(path_to_data, names = columns)

	#dropping the ID
	data.drop("id",axis=1,inplace=True)

	#Converting diagnosis into numeric and setting target
	target = data['diagnosis'].map({'M':1,'B':0})

	#Dropping the target from the data
	data.drop("diagnosis",axis=1,inplace=True)

	return data, target


def plot_validation_scores(cvresults, path_to_fig):

	param_grid = np.round(np.log10(cvresults.loc[:, "param_C"].values), 2)
	train_scores_std = cvresults.loc[:, "std_test_score"].values
	train_scores_mean = cvresults.loc[:, "mean_test_score"].values

	plt.figure(figsize=(8, 6))
	plt.title("Validation Performance")

	plt.grid()
	plt.fill_between(param_grid, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1, color="b")
	plt.plot(param_grid, train_scores_mean, 'o-', color="b")

	plt.xticks(param_grid, param_grid)
	
	plt.xlabel(r"$\log_{10} C$")
	plt.ylabel("Mean MCC")
	plt.tight_layout()
	plt.savefig(path_to_fig)


def confusion(y_true, y_pred):

	cm = np.zeros((2, 2), dtype=int)
	for i, y in enumerate(y_true):
		cm[int(y), int(y_pred[i])] += 1

	return cm


def plot_confusion_matrix(y_test, y_pred, path_to_fig):

	cm = confusion(y_test, y_pred)

	vrange = (0, np.max(cm))
	cmap = plt.cm.get_cmap('Blues', vrange[1])
	norm = matplotlib.colors.BoundaryNorm(np.arange(vrange[0], vrange[1]), cmap.N)

	reals = ["Benign", "Malignant"]
	preds = ["Malignant", "Benign"]

	plt.figure(figsize=(6, 6))
	ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=0.5, cmap=cmap, 
					 square=True, cbar=False, norm=norm, linecolor="k")
	ax.set_ylim(0, 2)
	ax.set_ylabel('Ground truth', weight='bold')
	ax.set_yticklabels(np.array(preds)[::-1], ha='right', va='center', rotation=0)
	ax.set_title("Predicted", weight='bold')
	ax.set_xticklabels(reals, ha='center', va='bottom', rotation=0)
	ax.xaxis.set_ticks_position('top')

	plt.tight_layout()
	plt.savefig(path_to_fig)
