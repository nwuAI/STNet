import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from main import main

if __name__ == '__main__':
	main(mode=2)