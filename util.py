from unicodedata import normalize
import nltk
import re
import time
import string
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

CATEGORIAS_DICT={
	"Redes Sociais": "Redes Sociais",
	"Cultura Geek": "Cultura Geek",
	"Seguranca": "Segurança",
	"Segurança": "Segurança",
	"Internet": "Internet",
	"Negocios": "Mercado",
	"Gestao": "Mercado",
	"Carreiras": "Mercado",
	"Tecnologia": "Mercado",
	"Mercado": "Mercado",
	"Produto": "Mercado",
	"Mobilidade": "Dispositivos Móveis",
	"Telecom": "Dispositivos Móveis",
	"Dispositivos Móveis": "Dispositivos Móveis",
	"Tecnologias_Emergentes": "Ciência",
	"TI_na_Pratica": "Ciência",
	"Aplicacoes": "Ciência",
	"Big_Data": "Ciência",
	"Ciência": "Ciência",
	"Mobilidade Urbana/Smart Cities": "Ciência",
	"Infraestrutura": "Infra",
	"Cloud_Computing": "Infra",
	"Software": "Software"
}

stemmer = nltk.stem.RSLPStemmer()
stopwords = [x.lower() for x in nltk.corpus.stopwords.words('portuguese')]

def normalize_token(token, stemming=False):
	token = normalize('NFKD', token).encode('ASCII','ignore').decode('ASCII')
	token = re.sub('[^A-Za-z]+', '', token)
	if(len(token) == 0): return ''

	if(not re.fullmatch('[' + string.punctuation + ']+', token)):
		if(stemming): return stemmer.stem(token).lower()
		else: return token.lower()
	else: return ''

def tempo(start, task, minutes=False):
	den = 1.0
	units = " segundos"
	if(minutes):
		den = 60.0
		units = " minutos\t"

	print("T =",round((time.time()-start)/den), units, " (", task, ")")

def erro(msg, exiting=True):
	print("[Erro]", msg)
	if(exiting): exit(0)

def plotConfMatrix(trueLabels, predictedLabels, labels, title):

	cm = confusion_matrix(trueLabels, predictedLabels)    

	plt.figure()

	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels, rotation=45)
	plt.yticks(tick_marks, labels)
	plt.ylabel('Classe real')
	plt.xlabel('Classe prevista')
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > (cm.max() / 2.) else "black")

	plt.show()
