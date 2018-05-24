from dataset.cifar.feed import feed as feed_cifar
from dataset.coil20.feed import feed as feed_coil20
def feed():
	feed_cifar(dataset_type=10)
	feed_cifar(dataset_type=100)

	feed_coil20(dataset_type='unprocessed')
	feed_coil20(dataset_type='processed')


from loader import loader
def check_shape():
	savePath = "/media/nvme0n1/DATA/TRAININGSETS/mnist/"
	ld = loader(savePath+"X.pa", savePath+"Y.pa")
	for X, Y in ld.get():
		print("[mnist]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		break

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/cifar/"
	ld = loader(savePath+"X_10.pa", savePath+"Y_10.pa")
	for X, Y in ld.get():
		print("[cifar10]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		break

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/cifar/"
	ld = loader(savePath+"X_100.pa", savePath+"Y_100.pa")
	for X, Y in ld.get():
		print("[cifar100]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		break

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/coil20/"
	ld = loader(savePath+"X_unprocessed.pa", savePath+"Y_unprocessed.pa")
	for X, Y in ld.get():
		print("[coil20-unprocessed]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		break

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/coil20/"
	ld = loader(savePath+"X_processed.pa", savePath+"Y_processed.pa")
	for X, Y in ld.get():
		print("[coil20-processed]:", X.shape, X.dtype, X.max(), Y.shape, Y.dtype, Y.max())
		break

from loader import data_checker
def check_imgs():
	savePath = "/media/nvme0n1/DATA/TRAININGSETS/mnist/"
	ld = loader(savePath+"X.pa", savePath+"Y.pa")
	data_checker(ld, "./data/mnist.png")

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/cifar/"
	ld = loader(savePath+"X_10.pa", savePath+"Y_10.pa")
	data_checker(ld, "./data/cifar_10.png")

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/cifar/"
	ld = loader(savePath+"X_100.pa", savePath+"Y_100.pa")
	data_checker(ld, "./data/cifar_100.png")

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/coil20/"
	ld = loader(savePath+"X_unprocessed.pa", savePath+"Y_unprocessed.pa")
	data_checker(ld, "./data/coil20-unprocessed.png")

	savePath = "/media/nvme0n1/DATA/TRAININGSETS/coil20/"
	ld = loader(savePath+"X_processed.pa", savePath+"Y_processed.pa")
	data_checker(ld, "./data/coil20-processed.png")

if __name__ == '__main__':
	from general.utils import mkdir
	mkdir("./data")
	check_imgs()




