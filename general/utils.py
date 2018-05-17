
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