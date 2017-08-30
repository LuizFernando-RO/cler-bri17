from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import csv
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import util as Util
import xgboost as xgb
import numpy as np

FEATURES_FILE='output/features.csv'

def carregar_features():
	X = []
	y = []
	with open(FEATURES_FILE, 'r') as f:
		reader = csv.reader(f, delimiter=';')
		for row in reader:
			X.append([float(x) for x in row[:-2]])
			y.append(row[-1])

	return (X, y)

def neural_network(X_train, X_test, y_train, y_test):
	print("*** Neural Network ***")
	clf = MLPClassifier(hidden_layer_sizes=[100],solver='adam')
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("neural network:", score)

	# Util.plotConfMatrix(y_test, clf.predict(X_test), clf.classes_, "Neural Network")

def naive_bayes(X_train, X_test, y_train, y_test):
	print("*** Naive Bayes ***")
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("naive bayes:", score)

def decision_tree(X_train, X_test, y_train, y_test):
	print("*** Decision tree ***")
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("decision tree:", score)

def xgboost(X_train, X_test, y_train, y_test):
	print("*** XGBoost ***")
	clf = xgb.XGBClassifier(max_depth=10,n_estimators=500,nthread=8,silent=True,seed=0)
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("xgboost:", score)

def execute():
	X, y = carregar_features()
	
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)
	
	c = list(zip(X, y))
	random.shuffle(c)
	X, y = zip(*c)
	X = np.array(list(X))
	y = np.array(list(y))

	total = len(X)
	prop = int(total/10)
	print("Tamanho do dataset:",total)
	print("Tamanho para treinamento:",total-prop)
	print("Tamanho para testes:",prop)

	X_train = X[:(total-prop)]
	X_test = X[(total-prop):]
	y_train = y[:(total-prop)]
	y_test = y[(total-prop):]

	naive_bayes(X_train, X_test, y_train, y_test)
	neural_network(X_train, X_test, y_train, y_test)
	decision_tree(X_train, X_test, y_train, y_test)
	xgboost(X_train, X_test, y_train, y_test)