import sys
import preprocess
import features_generator
import classifiers

def main():
	if('preprocess' in sys.argv):
		preprocess.execute()
	if('features' in sys.argv):
		features_generator.execute()
	if('cls' in sys.argv):
		classifiers.execute()

if __name__ == '__main__':
	main()