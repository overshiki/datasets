from .. import save_obj
from .download import download
import os

def feed(feed_path="/media/nvme0n1/DATA/TRAININGSETS/whales2018/"):
	if not os.path.exists(feed_path):
		os.mkdir(feed_path)

	final_dict = download()
	save_obj(feed_path+"dict.pkl", final_dict)