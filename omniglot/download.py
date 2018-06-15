from chainer.dataset import download as dl
import tarfile, os, re

def download(dataset_type='normal'):
	if dataset_type=='normal':
		url = 'https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip'
	elif dataset_type=='small1':
		url = 'https://github.com/brendenlake/omniglot/blob/master/python/images_background_small1.zip'
	elif dataset_type=='small2':
		url = 'https://github.com/brendenlake/omniglot/blob/master/python/images_background_small2.zip'
	elif dataset_type=='evaluation':
		url = 'https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip'


	archive_path = dl.cached_download(url)
	with tarfile.open(archive_path, 'r:gz') as archive:
		names = archive.getnames()
		for name in names:
			print(name)


if __name__ == '__main__':
	download()