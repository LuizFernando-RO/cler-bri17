import nltk
from nltk.tokenize import word_tokenize
import util as Util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import operator

DATA_FILE='output/preprocessed.csv'
datasets=['dataset/tecmundo.csv']

# carrega os dados de CSVs
def carregar_dataset(files):
	data = {}
	for csv_file in files:
		print("Carregando arquivo", csv_file)
		with open(csv_file, "r") as f:
			for row in f:
				tmp = row.split(";;;")
				# data[csv_file + tmp[0]] = [str(tmp[1] + ". " + tmp[2]), Util.CATEGORIAS_DICT[tmp[3].strip()]]
				data[csv_file + tmp[0]] = [str(tmp[1] + ". " + tmp[2]), tmp[3].strip()]
	return data

# Printa algumas informações sobre os dados
def analise_dados(data):
	print(len(data), "registros no dataset")
	categorias = {}
	for key in data:
		item = data[key]
		if(item[1] not in categorias):
			categorias[item[1]] = 1
		else:
			categorias[item[1]] += 1

	ordenado = sorted(categorias.items(), key=operator.itemgetter(1))
	ordenado.reverse()

	print("* Taxa de ocorrência de categorias:")
	for i, c in enumerate(ordenado):
		print("\t"+str(i)+". "+c[0]+": " + str(c[1]) + " (" + str(round(100*c[1]/len(data), 2)) +"%)")

	return categorias

# Prepara dados como X,y. Usa índice de lista_categorias para gerar y.
def preparar_dados(data):
	X = []
	y = []
	for key in data:
		item = data[key]
		X.append(item[0])
		y.append(item[1])
	return (X, y)

def preprocessamento(data):
	print("Iniciando pré-processamento")
	X = []
	y = []

	for key in data:
		texto = data[key][0]
		categoria = data[key][1]
		tokens = [x.lower() for x in word_tokenize(texto)]
		normalizados = []

		for t in tokens:
			if(t not in Util.stopwords):
				tmp = Util.normalize_token(t)
				if(len(tmp) > 0):
					normalizados.append(tmp)
		X.append(' '.join(normalizados))
		y.append(categoria)

	return (X, y)

def salvar_dados(X, y):
	print("Salvando", len(X), "itens após pré-processamento")
	with open(DATA_FILE, 'w') as f:
		for index, texto in enumerate(X):
			f.write(texto + ";" + y[index] + "\n")

def truncar_classes(data,analise,proporcao=0.1): # 0.1 para tecmundo e 0.09 para cw
	print("*** Truncando classes com proporcao menor que",proporcao,"***")
	final = {}
	total = len(data)

	for key in data:
		if(analise[data[key][1]]/total >= proporcao):
			final[key] = data[key]
	return final

def limitar_classes(data,limite=5000): # 5000 para tecmundo e 1500 para cw
	print("*** Limitando exemplos por classe em",limite,"***")
	final = {}
	contador = {}
	for key in data:
		if(data[key][1] not in contador):
			contador[data[key][1]] = 0

		if(contador[data[key][1]] < limite):
			contador[data[key][1]] += 1
			final[key] = data[key]

	return final


# Executa todo pré-processamento
def execute():
	print("\n*** Pré-processamento ***\n")
	data = carregar_dataset(datasets)
	analise = analise_dados(data)
	data = truncar_classes(data, analise)
	data = limitar_classes(data)
	analise_dados(data)
	X, y = preprocessamento(data)
	salvar_dados(X, y)



