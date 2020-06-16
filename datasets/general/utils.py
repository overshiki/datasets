import math
import numpy as np
import cv2
import os, re, pickle
from skimage.exposure import rescale_intensity
import tifffile

def save_obj(name, obj):
	if(not re.search('.pkl', name)):
		name = name+'.pkl'
	with open(name, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	if(not re.search('.pkl', name)):
		name = name+'.pkl'
	with open(name, 'rb') as f:
		return pickle.load(f)

def imreadTif(path):
	return tifffile.imread(path)

def imreadBGR(path):
	'''use opencv to load jpeg, png image as numpy arrays, the speed is triple compared with skimage
	'''
	return cv2.imread(path,3)

def mkdir(_dir):
	r'''
	recursively make dir
	'''

	def _mkdir(feed_path, sub_path):
		if not os.path.exists(feed_path):
			_mkdir(os.path.split(feed_path)[0], os.path.split(feed_path)[1])
		else:
			if not os.path.exists(feed_path+"/"+sub_path):
				os.mkdir(feed_path+"/"+sub_path)
				_mkdir(_dir, "/")

	_mkdir(_dir, "/")

def patch_interface(pos_x, pos_y, half_size_x, half_size_y):
	pos_x_left = pos_x - half_size_x
	pos_x_left = np.expand_dims(pos_x_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size_x), 1)
	pos_x = pos_x_left + adding

	pos_y_left = pos_y - half_size_y
	pos_y_left = np.expand_dims(pos_y_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size_y), 1)
	pos_y = pos_y_left+adding

	return pos_x[:, np.newaxis, :], pos_y


def groupby(seq, minibatch=10, key='mini'):
	_len = len(seq)
	if(key=='mini'):
		_num = math.ceil(_len*1./minibatch)
	elif(key=='num'):
		_num = minibatch
		minibatch = math.ceil(_len*1./_num)
	_list = []
	for i in range(_num):
		_start = i*minibatch
		if((i+1)*minibatch<_len):
			_end = (i+1)*minibatch
		else:
			_end = _len
		_list.append(seq[_start:_end])
	return _list

def imwrite(img,path):
	'''write image to path,usage:
	para1: path to be saved
	para2: img
	much much faster than skimage
	'''
	if(len(img.shape)==3):
		for i in range(3):
			slide = img[:,:,i]
			if(slide.max()==1):
				if(slide.dtype=='uint8'):
					img[:,:,i] = img[:,:,i]*255
				elif(slide.dtype=='uint16'):
					img[:,:,i] = img[:,:,i]*65535
	cv2.imwrite(path,img)


def patch_interface(pos_x, pos_y, half_size_x, half_size_y, extra_add=0):
	pos_x_left = pos_x - half_size_x
	pos_x_left = np.expand_dims(pos_x_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size_x+extra_add), 1)
	pos_x = pos_x_left + adding

	pos_y_left = pos_y - half_size_y
	pos_y_left = np.expand_dims(pos_y_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size_y+extra_add), 1)
	pos_y = pos_y_left+adding

	# x * y * N
	# _x = img[pos_x[:, np.newaxis, :], pos_y]
	return pos_x[:, np.newaxis, :], pos_y

def random_crop(data, crop_times=10, crop_size=(15, 15)):
	size_x, size_y = data.shape[2], data.shape[3]
	window_x = (crop_size[0], size_x-crop_size[0])
	window_y = (crop_size[1], size_y-crop_size[1])

	data_stack = []
	for i in range(crop_times):
		centers_x = np.random.randint(window_x[0], window_x[1], len(data))
		centers_y = np.random.randint(window_y[0], window_y[1], len(data))
		
		index_x, index_y = patch_interface(centers_x, centers_y, crop_size[0]//2, crop_size[1]//2)

		# print(index_x.shape, index_y.shape)
		temp = data[:, :, index_x, index_y]
		# print(temp.shape)
		data_stack.append(temp)


	data_stack = np.concatenate(data_stack, axis=-1)

	shape = data_stack.shape
	pivot = [x for x in range(len(shape))]
	lead = pivot.pop(0)
	pivot.append(lead)
	data_stack = data_stack.transpose(pivot)

	shape = list(data_stack.shape)
	end = shape.pop(-1)
	end = end*shape.pop(-1)
	shape.append(end)
	data_stack = data_stack.reshape(*shape)

	shape = data_stack.shape
	pivot = [x for x in range(len(shape))]
	lead = pivot.pop(-1)
	pivot.insert(0, lead)
	data_stack = data_stack.transpose(pivot)

	return data_stack