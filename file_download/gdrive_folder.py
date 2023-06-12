import gdown, sys, os


subdirectory = 'data'

url = sys.argv[1]
if url.split('/')[-1] == '?usp=sharing':
   url= url.replace('?usp=sharing','')
	
gdown.download_folder(url)

os.path.join(subdirectory, gdown.download_folder(url))