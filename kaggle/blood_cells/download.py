from subprocess import call
import zipfile, tempfile
import os, shutil, re
import pandas as pd
import numpy as np
from dataset.general.utils import imreadBGR, groupby
from dataset.general.serialize import data2pa
import numpy as np
import os
import xml.etree.ElementTree as ET

r'''
for more info, please go to:
https://www.kaggle.com/paultimothymooney/identify-blood-cell-subtypes-from-images
'''
def download(dataset_type=1):
	param_list = ["kaggle", "datasets", "download", "-d", "paultimothymooney/blood-cells"]
	if not os.path.exists(os.path.expanduser("~/.kaggle/datasets/paultimothymooney/blood-cells/dataset-master.zip")):
		call(param_list)

	if dataset_type==1:
		archive_path = os.path.expanduser("~/.kaggle/datasets/paultimothymooney/blood-cells/dataset-master.zip")
	fileOb = zipfile.ZipFile(archive_path, mode='r')
	names = fileOb.namelist()

	# for name in names:
	# 	if not bool(re.search("__MACOSX/dataset-master", name)):
	# 		# if bool(re.search(name, "Annotations")):
	# 		print(name)

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
			print(name)
			path = cache_path #+"/"+name
			xml_path = path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")

			print(xml_path)
			print(fileOb.extract(name, path=path))

			img = imreadBGR(fileOb.extract(name, path=path))
			data.append(img)

			xml = ET.parse(fileOb.extract(name.replace("JPEGImages", "Annotations").replace(".jpg", ".xml"), path=xml_path))
			for elem in xml.iter():
				print(elem)


			raise ValueError()
	finally:
		shutil.rmtree(cache_root)

	# data = np.stack(data, axis=0).transpose([0, 3, 1, 2])





	# tree = ET.parse('country_data.xml')
	# root = tree.getroot()


	# return data



def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/fer2013/"):

	data, label = download()

	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	data2pa(feed_path+"X.pa", data)
	data2pa(feed_path+"Y.pa", label)

if __name__ == '__main__':
	download()
	# data = download()
	# print(data.shape)