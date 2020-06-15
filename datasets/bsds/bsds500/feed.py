from dataset.general.utils import mkdir
from dataset.general.serialize import data2pa, pa2np
import numpy as np
import os
from dataset.bsds.bsds500.download import download

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/bsds/500/"):
	data = download()
	mkdir(feed_path)
	data2pa(feed_path+"X.pa", data)

def load(load_path="/media/nvme0n1/DATA/TRAININGSETS/bsds/500/"):
	if not os.path.exists(load_path+"X.pa"):
		raise ValueError("not files in load_path, please run feed() function first")

	data = pa2np(load_path+"X.pa")
	return data

if __name__ == '__main__':
	# feed()
	data = load()
	print(data.shape)