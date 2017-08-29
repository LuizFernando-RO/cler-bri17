from unicodedata import normalize
import nltk
import re
import time
import string

stemmer = nltk.stem.RSLPStemmer()
stopwords = [x.lower() for x in nltk.corpus.stopwords.words('portuguese')]

def normalize_token(token, stemming=True):
	
	#token = re.sub('[^A-Za-z]+', '', token)
	token = normalize('NFKD', token).encode('ASCII','ignore').decode('ASCII')
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