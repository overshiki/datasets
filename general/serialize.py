import pyarrow as pa
import numpy as np 
from timeit import default_timer

def serialize(data):
	buf = pa.serialize(data).to_buffer()
	return buf

def save(name, buf):
	with open(name, 'wb') as f:
		f.write(buf)

def readBuf(name):
	mmap = pa.memory_map(name)
	buf = mmap.read_buffer()
	return buf

def deserialize(buf):
	data = pa.deserialize(buf)
	return data

def npy2pa(path):
	data = np.load(path)
	buf = serialize(data)
	save(path.replace("npy", "pa"), buf)

def pa2np(path):
	buf = readBuf(path)
	data = deserialize(buf)
	return data

def data2pa(name, data):
	buf = serialize(data)
	save(name, buf)

if __name__ == '__main__':
	path = "/media/nvme0n1/DATA/TRAININGSETS/vinn/X.pa"
	data = pa2np(path)
	print(data.shape)

	path = "/media/nvme0n1/DATA/TRAININGSETS/vinn/Y.pa"
	data = pa2np(path)
	print(data.shape)