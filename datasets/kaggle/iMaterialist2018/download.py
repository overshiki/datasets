from subprocess import call

def download():
	param_list = ["kaggle", "competitions", "download", "-c", "imaterialist-challenge-fashion-2018"]
	call(param_list)


if __name__ == '__main__':
	download()
