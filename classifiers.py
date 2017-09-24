from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import csv
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import util as Util
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.ensemble import VotingClassifier
import operator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

def naive_bayes(X_train, X_val, y_train, y_val):
	print("*** Naive Bayes ***")

	scaler = MinMaxScaler()
	scaler.fit(np.append(np.array(X_train), np.array(X_val),axis=0))
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)

	clf = MultinomialNB()
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=3,n_jobs=1)
	print("naive bayes score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	clf.fit(X_train, y_train)
	score = clf.score(X_val, y_val)
	print("\tnaive bayes:", score)

	return clf

def decision_tree(X_train, X_val, y_train, y_val):
	print("*** Decision tree ***")
	clf = tree.DecisionTreeClassifier()

	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=3,n_jobs=1)
	print("decision tree score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	clf.fit(X_train, y_train)
	score = clf.score(X_val, y_val)
	print("\tdecision tree:", score)

	return clf

def xgboost(X_train, X_val, y_train, y_val, **kwargs):
	print("*** XGBoost ***", kwargs)
	clf = xgb.XGBClassifier(**kwargs)

	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=3,n_jobs=1)
	print("xgboost score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	clf.fit(X_train, y_train)
	score = clf.score(X_val, y_val)
	print("\txgboost:", score)

	Util.plotConfMatrix(y_val, clf.predict(X_val), clf.classes_, "XGBoost")

	return clf

def neural_network(X_train, X_val, y_train, y_val, **kwargs):
	print("*** Neural Network ***", kwargs)
	clf = MLPClassifier(**kwargs)
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=3,n_jobs=1)
	print("neural net score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	clf.fit(X_train, y_train)
	score = clf.score(X_val, y_val)
	print("\tneural network:", score)

	Util.plotConfMatrix(y_val, clf.predict(X_val), clf.classes_, "Neural Network")

	return clf

def SVM(X_train, X_val, y_train, y_val, **kwargs):
	print("*** SVM ***", kwargs)

	clf = SVC(**kwargs)
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=3,n_jobs=1)
	print("svm score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	clf.fit(X_train, y_train)
	score = clf.score(X_val, y_val)
	print("\tsvm:", score)

	Util.plotConfMatrix(y_val, clf.predict(X_val), clf.classes_, "SVM")

	return clf

def execute():
	X, y = carregar_features()

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
	X_train = np.array(X_train)
	X_val = np.array(X_val)

	print("Tamanho do dataset de treino:", len(X_train))
	print("Tamanho do dataset de validacao:", len(X_val))

	clf_nb = naive_bayes(X_train, X_val, y_train, y_val)
	clf_dt = decision_tree(X_train, X_val, y_train, y_val)
	clf_nn = neural_network(X_train, X_val, y_train, y_val, hidden_layer_sizes=[1000],activation='relu',solver='adam',max_iter=500)
	clf_xgb = xgboost(X_train, X_val, y_train, y_val,nthread=8,
		objective='multi:softmax', # fnc obj para multiclass
		learning_rate=0.2, # eta # tantar diminuir p/ evitar overfit
		max_depth=3, # profundidade max das arvores
		n_estimators=100 # qtd de arvores
		)
	clf_svm = SVM(X_train, X_val, y_train, y_val,C=1500,kernel='rbf',gamma=5)
