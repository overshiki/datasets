# import chainer
from .chainer_utils import get_cifar10, get_cifar100
from .. import data2pa, groupby
import numpy as np
import os

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/cifar/", dataset_type=10):

	if dataset_type==10:
		train, test = get_cifar10()

	elif dataset_type==100:
		train, test = get_cifar100()

	else:
		raise ValueError("dataset_type should be 10 or 100, not {}".format(dataset_type))

	data_list, label_list = [], []
	for data, label in train:
		data_list.append(data.astype(np.uint8))
		label_list.append(label.astype(np.uint8))

	data = np.stack(data_list)
	label = np.stack(label_list)

	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	if dataset_type==10:
		data2pa(feed_path+"X_10.pa", data)
		data2pa(feed_path+"Y_10.pa", label)
	if dataset_type==100:
		data2pa(feed_path+"X_100.pa", data)
		data2pa(feed_path+"Y_100.pa", label)
