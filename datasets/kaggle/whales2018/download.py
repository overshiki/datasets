from subprocess import call
import os, shutil
import zipfile, tempfile
import pandas as pd
import numpy as np
from .. import imreadBGR

def download():
	param_list = ["kaggle", "competitions", "download", "-c", "whale-categorization-playground"]
	# call(param_list)


	archive_path = os.path.expanduser("~/.kaggle/competitions/whale-categorization-playground/train.csv")
	data = pd.read_csv(archive_path, delimiter=',', dtype='a')

	images, ids = data['Image'].as_matrix(), data['Id'].as_matrix()

	id_set = set(ids)
	id_dict = {}
	for index, _id in enumerate(id_set):
		id_dict[_id] = index

	labels = np.array(list(map(lambda x:id_dict[x], ids)))

	archive_path = os.path.expanduser("~/.kaggle/competitions/whale-categorization-playground/train.zip")
	fileOb = zipfile.ZipFile(archive_path, mode='r')
	names = fileOb.namelist()


	cache_root = "./temp/" 
	try:
		os.makedirs(cache_root)
	except OSError:
		if not os.path.isdir(cache_root):
			raise
	cache_path = tempfile.mkdtemp(dir=cache_root)

	data = []

	try:
		for name in images:
				path = cache_path+"/train/"+name
				img = imreadBGR(fileOb.extract("train/"+name, path=path))
				data.append(img)
	# 				label.append(int(name.split("__")[0].split("/obj")[1]))
	finally:
		shutil.rmtree(cache_root)

	final_dict = {}
	final_dict['labels'] = labels 
	final_dict['data'] = data 
	final_dict['key_value'] = id_dict

	return final_dict
