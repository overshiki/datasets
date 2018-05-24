from dataset.cifar.feed import feed as feed_cifar
from dataset.coil20.feed import feed as feed_coil20
from dataset.fer2013.feed import feed as feed_fer2013
def feed():
	# feed_cifar(dataset_type=10)
	# feed_cifar(dataset_type=100)

	# feed_coil20(dataset_type='unprocessed')
	# feed_coil20(dataset_type='processed')

	feed_fer2013()


from loader import loader
def check_shape():
	root_path = "/media/nvme0n1/DATA/TRAININGSETS/"

	savePath = root_path+"mnist/"
	ld = loader(savePath+"X.pa", savePath+"Y.pa")
	X, Y = next(ld.get())
	print("[mnist]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		

	savePath = root_path+"cifar/"
	ld = loader(savePath+"X_10.pa", savePath+"Y_10.pa")
	X, Y = next(ld.get())
	print("[cifar10]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		

	savePath = root_path+"cifar/"
	ld = loader(savePath+"X_100.pa", savePath+"Y_100.pa")
	X, Y = next(ld.get())
	print("[cifar100]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		

	savePath = root_path+"coil20/"
	ld = loader(savePath+"X_unprocessed.pa", savePath+"Y_unprocessed.pa")
	X, Y = next(ld.get())
	print("[coil20-unprocessed]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		

	savePath = root_path+"coil20/"
	ld = loader(savePath+"X_processed.pa", savePath+"Y_processed.pa")
	X, Y = next(ld.get())
	print("[coil20-processed]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		

	savePath = root_path+"fer2013/"
	ld = loader(savePath+"X.pa", savePath+"Y.pa")
	X, Y = next(ld.get())
	print("[fer2013]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		

from loader import data_checker
def check_imgs():
	root_path = "/media/nvme0n1/DATA/TRAININGSETS/"

	savePath = root_path+"mnist/"
	ld = loader(savePath+"X.pa", savePath+"Y.pa")
	data_checker(ld, "./data/mnist.png")

	savePath = root_path+"cifar/"
	ld = loader(savePath+"X_10.pa", savePath+"Y_10.pa")
	data_checker(ld, "./data/cifar_10.png")

	savePath = root_path+"cifar/"
	ld = loader(savePath+"X_100.pa", savePath+"Y_100.pa")
	data_checker(ld, "./data/cifar_100.png")

	savePath = root_path+"coil20/"
	ld = loader(savePath+"X_unprocessed.pa", savePath+"Y_unprocessed.pa")
	data_checker(ld, "./data/coil20-unprocessed.png")

	savePath = root_path+"coil20/"
	ld = loader(savePath+"X_processed.pa", savePath+"Y_processed.pa")
	data_checker(ld, "./data/coil20-processed.png")

	savePath = root_path+"fer2013/"
	ld = loader(savePath+"X.pa", savePath+"Y.pa")
	data_checker(ld, "./data/fer2013.png")

if __name__ == '__main__':
	# feed()

	# check_shape()

	from general.utils import mkdir
	mkdir("./data")
	check_imgs()




