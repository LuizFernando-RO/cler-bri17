from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import csv
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import util as Util
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

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
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("naive bayes score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("naive bayes:", score)

def neural_network(X_train, X_val, y_train, y_val):
	print("*** Neural Network ***")
	clf = MLPClassifier(hidden_layer_sizes=[100]*2,activation='relu',solver='adam')
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("neural net score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("neural network:", score)

	# Util.plotConfMatrix(y_val, clf.predict(X_val), clf.classes_, "Neural Network")

def decision_tree(X_train, X_val, y_train, y_val):
	print("*** Decision tree ***")
	clf = tree.DecisionTreeClassifier()

	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("decision tree score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("decision tree:", score)

def xgboost(X_train, X_val, y_train, y_val):
	print("*** XGBoost ***")
	clf = xgb.XGBClassifier(learning_rate=0.1,objective='multi:softmax',max_depth=12,n_estimators=500,nthread=8,silent=True,seed=0)

	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("xgboost score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("xgboost:", score)

def knn(X_train, X_val, y_train, y_val):
	print("*** KNN ***")

	scaler = MinMaxScaler()
	scaler.fit(np.append(np.array(X_train), np.array(X_val),axis=0))
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)

	clf = KNeighborsClassifier(n_jobs=-1, n_neighbors = 10)
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("KNN score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("naive bayes:", score

def random_forest(X_train, X_val, y_train, y_val):
	print("*** Random Forest ***")

	scaler = MinMaxScaler()
	scaler.fit(np.append(np.array(X_train), np.array(X_val),axis=0))
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)

	clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("random forest score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("naive bayes:", score)

def SVM(X_train, X_val, y_train, y_val):
	print("*** SVM ***")

	scaler = MinMaxScaler()
	scaler.fit(np.append(np.array(X_train), np.array(X_val),axis=0))
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)

	clf = svm.SVC()
	
	score_cv = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5,n_jobs=1)
	print("svm score cv:", np.mean(score_cv), max(score_cv), min(score_cv), np.std(score_cv))

	# clf.fit(X_train, y_train)
	# score = clf.score(X_val, y_val)
	# print("naive bayes:", score)

def execute():
	X, y = carregar_features()

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
	X_val = X[(total-prop):]
	y_train = y[:(total-prop)]
	y_val = y[(total-prop):]

	naive_bayes(X_train, X_val, y_train, y_val)
	neural_network(X_train, X_val, y_train, y_val)
	decision_tree(X_train, X_val, y_train, y_val)
	#xgboost(X_train, X_val, y_train, y_val)
	knn(X_train, X_val, y_train, y_val)
	random_forest(X_train, X_val, y_train, y_val)
	SVM(X_train, X_val, y_train, y_val)