import urllib.request, json 

URL_TECMUNDO="https://api.tecmundo.com.br/api/v2/news/latest-news/noticias?top=1000&noHighlights=False&idReference=121252&older=True&os=&host=https://www.tecmundo.com.br"
CSV_NAME='tecmundo.csv'
def main():
	print("Buscando notícias em TecMundo")
	categorias = {}
	with open(CSV_NAME, 'w') as f:
		with urllib.request.urlopen(URL_TECMUNDO) as url:
			data = json.loads(url.read().decode())
			for noticia in data['data']:
				conteudo = [str(noticia['Id']),noticia['Title'],noticia['Chamada'],noticia['Tag']['Title']]
				f.write(';;;'.join(conteudo) + "\n")
				if(noticia['Tag']['Title'] not in categorias):
					categorias[noticia['Tag']['Title']] = 1
				else:
					categorias[noticia['Tag']['Title']] += 1

	print("Resultado salvo em", CSV_NAME)
	i = 0
	for cat in categorias:
		print(cat, categorias[cat])
		i += categorias[cat]
	print("Total", i)

if __name__ == '__main__':
	main()