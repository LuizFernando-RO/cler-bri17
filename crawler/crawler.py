import urllib.request, json 

K=50
URL_TECMUNDO="https://api.tecmundo.com.br/api/v2/news/latest-news/noticias?top=1000&host=https://www.tecmundo.com.br&idReference="
CSV_NAME='tecmundo.csv'
def main():
	print("Buscando notícias em TecMundo")
	categorias = {}
	noticias = {}

	id_atual = 0

	for index in range(K):
		with urllib.request.urlopen(URL_TECMUNDO + str(id_atual)) as url:
			data = json.loads(url.read().decode())
			for n in data['data']:
				conteudo = [str(n['Id']).strip(),n['Title'].strip(),n['Chamada'].strip(),n['Tag']['Title'].strip()]
				if conteudo[0] not in noticias:
					noticias[conteudo[0]] = conteudo
					id_atual = conteudo[0]

		print("k =", index, "->", len(noticias), "noticias")

	with open(CSV_NAME, 'w') as f:
		for n in noticias:
			noticia = noticias[n]
			f.write(';;;'.join(noticia) + '\n')
			if(noticia[3] not in categorias):
				categorias[noticia[3]] = 1
			else:
				categorias[noticia[3]] += 1

	print("Resultado salvo em", CSV_NAME)
	i = 0
	for cat in categorias:
		print(cat, categorias[cat])
		i += categorias[cat]
	print("Total", i)

if __name__ == '__main__':
	main()