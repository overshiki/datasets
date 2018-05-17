import chainer
import numpy as np
from ..general.utils import groupby
from ..general.serialize import data2pa

train, test = chainer.datasets.get_mnist(ndim=2, scale=255)

data_list, label_list = [], []
for data, label in train:
	data_list.append(data.astype(np.uint8))
	label_list.append(label)

data = np.stack(data_list)
label = np.stack(label_list)

data2pa("/media/nvme0n1/DATA/TRAININGSETS/mnist/X.pa", data)
data2pa("/media/nvme0n1/DATA/TRAININGSETS/mnist/Y.pa", label)