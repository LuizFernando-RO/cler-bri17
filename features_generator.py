import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec
from sparsesvd import sparsesvd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import numpy as np
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PP_FILE="output/preprocessed.csv"
FEATURES_FILE="output/features.csv"

def carregar_dados():
	X = []
	y = []

	with open(PP_FILE, 'r') as f:
		reader = csv.reader(f, delimiter=';')
		for row in reader:

			X.append(row[0])
			y.append(row[1])

	texts = [document.split() for document in X]

	return (texts, y)

def vetor_medio(X, word2vec):

	media = pickle.load(open("media.txt","rb"))

	return np.array([
		np.mean([word2vec[w] for w in words if w in word2vec] or [media], axis=0) for words in X])

def salvar_features(features_x, features_y):
	with open(FEATURES_FILE, 'w') as f:
		writer = csv.writer(f, delimiter=';')
		for index, item in enumerate(features_x):
			row = item.tolist()
			row.append(features_y[index])
			writer.writerow(row)

def execute():
	print("\n*** Geração das features ***\n")
	sentenca, categoria = carregar_dados()
	model = pickle.load(open("w2v.txt","rb"))
	features_x = vetor_medio(sentenca, model)
	salvar_features(features_x, categoria)
