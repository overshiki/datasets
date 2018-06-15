from subprocess import call
import zipfile, tempfile
import os, shutil, re
import pandas as pd
import numpy as np
from dataset.general.utils import imreadBGR, data2pa, groupby
import numpy as np
import os

def download(dataset_type=1):
	param_list = ["kaggle", "datasets", "download", "-d", "paultimothymooney/blood-cells"]
	if not os.path.exists(os.path.expanduser("~/.kaggle/datasets/paultimothymooney/blood-cells/dataset-master.zip")):
		call(param_list)

	if dataset_type==1:
		archive_path = os.path.expanduser("~/.kaggle/datasets/paultimothymooney/blood-cells/dataset-master.zip")
	fileOb = zipfile.ZipFile(archive_path, mode='r')
	names = fileOb.namelist()

	names = list(filter(lambda x:re.search("dataset-master/JPEGImages/BloodImage", x), names))

	cache_root = "./temp/" 
	try:
		os.makedirs(cache_root)
	except OSError:
		if not os.path.isdir(cache_root):
			raise
	cache_path = tempfile.mkdtemp(dir=cache_root)

	data = []

	try:
		for name in names:
			path = cache_path+"/"+name
			img = imreadBGR(fileOb.extract(name, path=path))
			data.append(img)
	# # 				label.append(int(name.split("__")[0].split("/obj")[1]))
	finally:
		shutil.rmtree(cache_root)

	# final_dict = {}
	# final_dict['labels'] = labels 
	# final_dict['data'] = data 
	# final_dict['key_value'] = id_dict
	data = np.stack(data, axis=0).transpose([0, 3, 1, 2])
	return data



def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/fer2013/"):

	data, label = download()

	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	data2pa(feed_path+"X.pa", data)
	data2pa(feed_path+"Y.pa", label)

if __name__ == '__main__':
	data = download()
	print(data.shape)