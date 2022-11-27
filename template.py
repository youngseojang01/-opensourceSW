#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/youngseojang01/-opensourceSW/blob/main/template.py

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def load_dataset(dataset_path):
	#To-Do: Implement this function
	#load the csv file
	dataset = pd.read_csv(dataset_path)
	
	df = pd.DataFrame(dataset)
	return df

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	n_feats = len(dataset_df)
	n_class0 = dataset_df.count('target' = 0)
	n_class1 = dataset_df.count('target' = 1)
	return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	data = dataset['target' = 0]
	target = dataset['target'= 1]
	x_train, x_test, y_train, y_test = train_test_split(data, target, testset_size)
	
	return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	#train the decision tree model
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(X_train, y_train)
	
	#evaluate the performances
	accuracy = accuracy_score(y_test, dt_cls.predict(x_test))
	precision = precision_score(y_test, dt_cls.predict(x_test))
	recall = recall_score(y_test, dt_cls.predict(x_test))
	
	#return 3 performance metrics
	return accuracy, precision, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	#train the random forest model
	rf_cls = RandomForestClassifer()
	rf_cls.fit(x_train, y_train)
	
	#evlauate the performances
	acc = accuracy_score(y_test, rf_cls.predict(x_test))
	prec = precision_score(y_test, rf_cls.predict(x_test))
	recall = recall_score(y_test, rf_cls.predict(x_test))
	
	#return 3 performance metrics
	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	# *since SVM is very sensitive to the scale of each feature, normalize features
	svm_pipe = make_pipeline(
		StandardScaler()
		SVC()
	)
	svm_pipe.fit(x_train, y_train)
	
	#evlauate the performances
	acc = accuracy_score(y_test, svm_pipe.predict(x_test))
	prec = precision_score(y_test, svm_pipe.predict(x_test))
	recall = recall_score(y_test, svm_pipe.predict(x_test))
	
	#return 3 performance metrics
	return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
