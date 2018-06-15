from subprocess import call
import zipfile, tempfile
import os, shutil, re
import pandas as pd
import numpy as np

def download():
	param_list = ["kaggle", "competitions", "download", "-c", "data-science-bowl-2018"]
	# call(param_list)

	archive_path = os.path.expanduser("~/.kaggle/competitions/data-science-bowl-2018/stage1_train.zip")
	fileOb = zipfile.ZipFile(archive_path, mode='r')
	names = fileOb.namelist()

	for name in names:
		# if bool(re.search("images", name)) and bool(re.search("png", name)):
		# 	if name.replace("images", "masks") in names:

		if bool(re.search("ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48", name)):
			print(name)



	# cache_root = "./temp/" 
	# try:
	# 	os.makedirs(cache_root)
	# except OSError:
	# 	if not os.path.isdir(cache_root):
	# 		raise
	# cache_path = tempfile.mkdtemp(dir=cache_root)


	# archive_path = os.path.expanduser("~/.kaggle/competitions/data-science-bowl-2018/stage1_train_labels.csv.zip")
	# fileOb = zipfile.ZipFile(archive_path, mode='r')
	# names = fileOb.namelist()

	# try:
	# 	for name in names:
	# 		path = cache_path+"/"+name

	# 		df = pd.read_csv(fileOb.extract(name, path=path))
	# 		print(df)

	# finally:
	# 	shutil.rmtree(cache_root)

if __name__ == '__main__':
	download()