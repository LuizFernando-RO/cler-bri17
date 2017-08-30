import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing

from sparsesvd import sparsesvd
from sklearn.feature_extraction.text import TfidfVectorizer


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
	return (X, y)

def gerar_features(textos, labels):
	# vectorizer = CountVectorizer(ngram_range=(1,3),binary=True)
	vectorizer = TfidfVectorizer(ngram_range=(1, 2))

	categorias = []

	X = vectorizer.fit_transform(textos)
	y = []

	print("Shape de X antes do SVD: ", X.shape)

	X = X.tocsc()
	X, Sigma, VT = sparsesvd(X, 100)
	X = X.transpose()

	print("Shape de X depois do SVD: ", X.shape)

	# plt.scatter(range(len(Sigma)), Sigma)
	# plt.show()

	return (X.tolist(), labels)

def salvar_features(features_x, features_y):
	with open(FEATURES_FILE, 'w') as f:
		writer = csv.writer(f, delimiter=';')
		for index, item in enumerate(features_x):
			row = item
			row.append(features_y[index])
			writer.writerow(row)

def execute():
	print("\n*** Geração das features ***\n")
	textos, labels = carregar_dados()
	features_x, features_y = gerar_features(textos, labels)
	salvar_features(features_x, features_y)