from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import csv
import random
from sklearn.preprocessing import MinMaxScaler

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
	clf = MLPClassifier(hidden_layer_sizes=[100],solver='adam')
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("neural network:", score)

def naive_bayes(X_train, X_test, y_train, y_test):
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("naive bayes:", score)

def execute():
	X, y = carregar_features()
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)
	c = list(zip(X, y))
	random.shuffle(c)
	X, y = zip(*c)
	X_train = X[:900]
	X_test = X[900:]
	y_train = y[:900]
	y_test = y[900:]

	naive_bayes(X_train, X_test, y_train, y_test)
	neural_network(X_train, X_test, y_train, y_test)