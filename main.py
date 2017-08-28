import sys
import preprocess

def main():
	if('preprocess' in sys.argv):
		preprocess.execute()


if __name__ == '__main__':
	main()