from chainer.dataset import download as dl
import tarfile, os, re, tempfile, shutil, pickle
from dataset.general.utils import imreadBGR
import numpy as np

r'''
TODO: currently only raw image for trainingset, the region mask is in .mat format
'''
def download():
	url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
	archive_path = dl.cached_download(url)
	with tarfile.open(archive_path, 'r:gz') as archive:
		names = archive.getnames()

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
				if bool(re.search("BSR/BSDS500/data/images/train/", name)) and bool(re.search("jpg", name)):
					path = cache_path
					archive.extract(name, path)
					img = imreadBGR(path+"/"+name)
					lead = img.shape[0]
					#for bsds500, imgs are of shape 481*321*3
					if lead!=481:
						img = img.transpose([1,0,2])

					lead = img.shape[0]
					if lead!=481:
						raise ValueError("img shape not to be 481*321*3, but: {}".format(img.shape))

					data.append(img)
		finally:
			shutil.rmtree(cache_root)

		data = np.stack(data, axis=0)
		data = data.transpose([0,3,1,2])
		return data

if __name__ == '__main__':
	download()
