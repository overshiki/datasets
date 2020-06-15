import chainer
import numpy as np
from ..general.utils import groupby
from ..general.serialize import data2pa, pa2np
import os

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/mnist/"):
	train, test = chainer.datasets.get_mnist(ndim=2, scale=255)

	data_list, label_list = [], []
	for data, label in train:
		data_list.append(data.astype(np.uint8))
		label_list.append(label)

	data = np.stack(data_list)
	label = np.stack(label_list)

	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	data2pa(feed_path+"X.pa", data)
	data2pa(feed_path+"Y.pa", label)

def load(load_path="/media/nvme0n1/DATA/TRAININGSETS/mnist/"):
	if not os.path.exists(load_path+"X.pa") and os.path.exists(load_path+"Y.pa"):
		raise ValueError("not files in load_path, please run feed() function first")

	data, mask = pa2np(load_path+"X.pa"), pa2np(load_path+"Y.pa")
	return data, mask