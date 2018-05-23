from .. import data2pa, groupby
import numpy as np
import os
from .download import download

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/coil20/", dataset_type='unprocessed'):

	data, label = download(dataset_type)

	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	data2pa(feed_path+"X_{}.pa".format(dataset_type), data)
	data2pa(feed_path+"Y_{}.pa".format(dataset_type), label)


