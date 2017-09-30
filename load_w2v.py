import gensim
from unicodedata import normalize
import re
import string
import nltk
import pickle
import numpy as np

word2vec = gensim.models.Word2Vec.load('pt/pt.bin')
w2v_refatorado = {}
media = []

stemmer = nltk.stem.RSLPStemmer()
#stopwords = [x.lower() for x in nltk.corpus.stopwords.words('portuguese')]

def normalize_token(token, stemming=False):
	token = normalize('NFKD', token).encode('ASCII','ignore').decode('ASCII')
	token = re.sub('[^A-Za-z]+', '', token)
	if(len(token) == 0): return ''

	if(not re.fullmatch('[' + string.punctuation + ']+', token)):
		if(stemming): return stemmer.stem(token).lower()
		else: return token.lower()
	else: return ''

for word in word2vec.vocab:
	media.append(word2vec[word])
	w2v_refatorado[normalize_token(word)] = word2vec[word]

media = np.mean(media,axis=0)

pickle.dump(media, open("media.txt", "wb"))
pickle.dump(w2v_refatorado, open("w2v.txt", "wb"))