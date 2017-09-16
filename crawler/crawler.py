import urllib.request, json

import urllib
from pyquery import PyQuery as pq
from lxml import etree

import sys

K_TEC=50
URL_TECMUNDO="https://api.tecmundo.com.br/api/v2/news/latest-news/noticias?top=1000&host=https://www.tecmundo.com.br&idReference="
CSV_TEC_NAME='tecmundo.csv'

K_CW=250
URL_COMPUTERWORLD="http://computerworld.com.br/"
CSV_CW_NAME='computerworld.csv'
URL_CAT_CW=['aplicacoes','gestao','seguranca','infraestrutura','tecnologia','tecnologias-emergentes','carreira','internet','cloud-computing','mobilidade',
			'telecom-0','big-data','ti-na-pratica','negocios']
	
def tecmundo():
	print("Buscando notícias em TecMundo")
	categorias = {}
	noticias = {}

	id_atual = 0

	for index in range(K_TEC):
		with urllib.request.urlopen(URL_TECMUNDO + str(id_atual)) as url:
			data = json.loads(url.read().decode())
			for n in data['data']:
				conteudo = [str(n['Id']).strip(),n['Title'].replace('\n', ' ').strip(),n['Chamada'].replace('\n', ' ').strip(),n['Tag']['Title'].replace('\n', ' ').strip()]
				if conteudo[0] not in noticias:
					noticias[conteudo[0]] = conteudo
					id_atual = conteudo[0]

		print("k =", index, "->", len(noticias), "noticias")

	with open(CSV_TEC_NAME, 'w') as f:
		for n in noticias:
			noticia = noticias[n]
			f.write(';;;'.join(noticia) + '\n')
			if(noticia[3] not in categorias):
				categorias[noticia[3]] = 1
			else:
				categorias[noticia[3]] += 1

	print("Resultado salvo em", CSV_TEC_NAME)
	i = 0
	for cat in categorias:
		print(cat, categorias[cat])
		i += categorias[cat]
	print("Total", i)

def computerworld():
	print("Buscando notícias em ComputerWorld")

	noticias = {}
	categorias = {}

	for categoria in URL_CAT_CW:
		print("CW: buscando em", categoria)
		cont = 0
		for page in range(K_CW):
			if(page%50 == 0):
				print(page)

			url_page = URL_COMPUTERWORLD + categoria + "?page=" + str(page)
			
			d = pq("<html></html>")
			d = pq(etree.fromstring("<html></html>"))
			d = pq(url=url_page, opener=lambda url, **kw: urllib.request.urlopen(url).read())
			
			itens = d('article.itemNoticia')
			if(len(itens) == 0):
				break

			for item in itens:
				identificador = item.cssselect('h3 a')[0].get('href').replace('\n', ' ').strip()
				titulo = item.cssselect('h3')[0].text_content().replace('\n', ' ').strip()
				abstract = item.cssselect('p')[0].text_content().replace('\n', ' ').strip()
				
				conteudo = [identificador, titulo, abstract, categoria]
				if(conteudo[0] not in noticias):
					noticias[conteudo[0]] = conteudo
					cont += 1

		print("->", len(noticias), "noticias")
		categorias[categoria] = cont

	i = 0
	for cat in categorias:
		print(cat, categorias[cat])
		i += categorias[cat]
	print("Total", i)

	with open(CSV_CW_NAME, 'w') as f:
		for n in noticias:
			noticia = noticias[n]
			f.write(';;;'.join(noticia) + '\n')

	print("Resultado salvo em", CSV_CW_NAME)

def main():
	if('tec' in sys.argv):
		tecmundo()
	if('cw' in sys.argv):
		computerworld()

if __name__ == '__main__':
	main()