
if __name__ == '__main__':
	# from dataset.cifar.feed import feed 
	# feed(dataset_type=10)
	# feed(dataset_type=100)

	from dataset.coil20.feed import feed
	feed(dataset_type='unprocessed')
	feed(dataset_type='processed')