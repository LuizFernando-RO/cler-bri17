# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET

from os import listdir

FEED_DIRECTORY = 'RSS/Computerworld/'
DATASET_DIRECTORY = 'dataset/'
DELIMITER = ';;;'

def main():

	news = dict()

	fullText = ''

	i = 0
    
	for category in listdir(FEED_DIRECTORY):

		for xmlFile in listdir(FEED_DIRECTORY + '/' + category + '/'):

			tree = ET.parse(FEED_DIRECTORY + '/' + category + '/' + xmlFile)
			root = tree.getroot()

			for child in root:

				for item in child.findall('item'):

					title = item.find('title').text

					if title not in news.keys():

						news[title] = i

						description = item.find('description').text

						fullText += str(i) + DELIMITER + title + DELIMITER + description + DELIMITER + category + '\n'

						i += 1

	f = open(DATASET_DIRECTORY + 'computerworld.csv', 'w')
		
	f.write(fullText)
		
	f.close()
    
	return
    
main()
