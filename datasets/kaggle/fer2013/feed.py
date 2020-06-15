from .. import data2pa, groupby
import numpy as np
import os
from .download import download

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/fer2013/"):

	data, label = download()

	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	data2pa(feed_path+"X.pa", data)
	data2pa(feed_path+"Y.pa", label)