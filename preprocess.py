
datasets=['dataset/tecmundo.csv']

# carrega os dados de CSVs
def carregar_dataset(files):
	data = {}
	for csv_file in files:
		print("Carregando arquivo", csv_file)
		with open(csv_file, "r") as f:
			for row in f:
				tmp = row.split(";;;")
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

	print("* Taxa de ocorrência de categorias:")
	for c in categorias:
		print("\t"+c+": " + str(categorias[c]) + " (" + str(100*categorias[c]/len(data)) +"%)")

	return list(categorias.keys())

# Prepara dados como X,y. Usa índice de lista_categorias para gerar y.
def preparar_dados(data, categorias):
	X = []
	y = []
	for item in data:
		X.append(item[0])
		y.append(categorias.index(item[1]))
	print(X[:10])
	print(y[:10])

# Executa todo pré-processamento
def execute():
	print("\n*** Pré-processamento ***\n")
	data = carregar_dataset(datasets)
	lista_categorias = analise_dados(data)
	X, y = preparar_dados(data, lista_categorias)


