from chainer.dataset import download as dl
import tarfile, os, re

def download():
	url = "http://people.csail.mit.edu/rgrosse/intrinsic/intrinsic-data.tar.gz"
	archive_path = dl.cached_download(url)
	with tarfile.open(archive_path, 'r:gz') as archive:
		names = archive.getnames()
		key_set = set()
		object_set = set()
		for path in names:
			if bool(re.search("png", path)):
				key = os.path.split(path)[1].split(".")[0]
				key_set.add(key)

				object_name = os.path.split(os.path.split(path)[0])[1]
				# print(object_name)
				object_set.add(object_name)

		print(key_set, object_set)



if __name__ == '__main__':
	download()