# from chainer.dataset import download as dl
from subprocess import call
import tarfile, os, tempfile, shutil
import pandas as pd
import numpy as np

def download():
	param_list = ["kaggle", "competitions", "download", "-c", "challenges-in-representation-learning-facial-expression-recognition-challenge"]
	# call(param_list)
	archive_path = os.path.expanduser("~/.kaggle/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.tar.gz")
	with tarfile.open(archive_path, 'r:gz') as archive:
		names = archive.getnames()

		cache_root = "./temp/" 
		try:
			os.makedirs(cache_root)
		except OSError:
			if not os.path.isdir(cache_root):
				raise
		cache_path = tempfile.mkdtemp(dir=cache_root)

		data, label = [], []

		try:
			name = 'fer2013/fer2013.csv'
			path = cache_path+"/"+name
			archive.extract(name, path=cache_path)
			data = pd.read_csv(path, delimiter=',', dtype='a')
			label = pd.to_numeric(data['emotion']).as_matrix().astype(np.uint8) 

			img = list(map(lambda x:list(map(lambda y:int(y), x.split(" "))), data['pixels'].values.tolist()))
			img = np.array(img, dtype=np.uint8)
			img = img.reshape((len(label), 48, -1))
			img = np.expand_dims(img, axis=1)


		finally:
			shutil.rmtree(cache_root)

	return img, label
