import numpy as np
from timeit import default_timer as timer
from general.serialize import pa2np
from general.utils import groupby, imwrite, rescale_intensity


class loader:
	def __init__(self, X_path, Y_path, minibatch=1000):
		end = timer()
		self.X = pa2np(X_path)
		self.Y = pa2np(Y_path).astype('int32')
		if self.X.ndim==3:
			self.X = np.expand_dims(self.X, 1)
		print("loading time: ", timer()-end)
		self.minibatch = minibatch

	def get(self):
		indices = np.random.permutation(len(self.X)).tolist()
		groups = groupby(indices, self.minibatch, key='mini')
		for index, group in enumerate(groups):
			yield self.X[group], self.Y[group]


from general.utils import imwrite, rescale_intensity, patch_interface
def data_checker_mask(loader):
	X, Y = next(loader.get())
	X, Y = X[:100].squeeze(), Y[:100]

	X, Y = X.transpose([1, 2, 0]), Y.transpose([1, 2, 0])


	stride = 60
	pos_x = np.arange(0, 600, stride)
	pos_y = np.arange(0, 600, stride)
	vx, vy = np.meshgrid(pos_x, pos_y)
	pos = np.stack([vx, vy]).reshape((2, -1)).transpose([1,0])+stride//2

	real, binary = np.zeros((600, 600)), np.zeros((600, 600))

	real[patch_interface(pos[:,0], pos[:,1], stride//2)] = X
	binary[patch_interface(pos[:,0], pos[:,1], stride//2)] = Y

	binary = binary.astype('bool')

	# print(real.max(), binary.max(), binary.sum(), binary.size)

	img = np.stack([real]*3, axis=2)+500
	img[:,:, 0] = img[:,:, 0]*binary 
	img[:,:, 1] = img[:,:, 1]*np.invert(binary)
	img[:,:, 2] = img[:,:, 2]*np.invert(binary)
	imwrite(rescale_intensity(img.astype('uint8')), "./save/data_check.png")

def data_checker(loader, path):
	X, _ = next(loader.get())
	X = X[:100]

	channels = X.shape[1]

	X = X.transpose([2, 3, 0, 1])

	size_x, size_y = X.shape[0], X.shape[1]

	pos_x = np.arange(0, size_x*10, size_x)
	pos_y = np.arange(0, size_y*10, size_y)
	vx, vy = np.meshgrid(pos_x, pos_y)
	pos = np.stack([vx, vy]).reshape((2, -1)).transpose([1,0])
	pos[:,0] = pos[:,0]+size_x//2
	pos[:,1] = pos[:,1]+size_y//2

	real = np.zeros((size_x*10, size_y*10, channels))

	real[patch_interface(pos[:,0], pos[:,1], size_x//2, size_y//2)] = X

	if real.shape[2]==1:
		real = np.concatenate([real]*3, axis=2)
	print(real.shape)

	imwrite(real, path)