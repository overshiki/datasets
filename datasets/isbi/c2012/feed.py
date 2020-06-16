from dataset.general.utils import imreadTif, mkdir
from dataset.general.serialize import data2pa, pa2np
import numpy as np
import os

error_info = r'''
not found train-labels.tif and train-volume.tif
please go to website http://brainiac2.mit.edu/isbi_challenge/home
and follow the introduction to get the files
'''

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/isbi/2012/"):
	try:
		data, mask = imreadTif("train-volume.tif"), imreadTif("train-labels.tif")
	except:
		raise ValueError(error_info)
	data, mask = np.expand_dims(data, axis=1), np.expand_dims(mask, axis=1)

	mkdir(feed_path)

	data2pa(feed_path+"X.pa", data)
	data2pa(feed_path+"Y.pa", mask)


def load(load_path="/media/nvme0n1/DATA/TRAININGSETS/isbi/2012/"):
	if not os.path.exists(load_path+"X.pa") and os.path.exists(load_path+"Y.pa"):
		raise ValueError("not files in load_path, please run feed() function first")

	data, mask = pa2np(load_path+"X.pa"), pa2np(load_path+"Y.pa")
	return data, mask

if __name__ == '__main__':
	feed()