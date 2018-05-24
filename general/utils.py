import math
import numpy as np
import cv2, os
from skimage.exposure import rescale_intensity

def mkdir(_dir):
	if not os.path.exists(_dir):
		os.makedirs(_dir)

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

