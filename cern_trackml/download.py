from subprocess import call
def download():
	param_list = ["kaggle", "competitions", "download", "-c", "trackml-particle-identification"]
	call(param_list)

if __name__ == '__main__':
	download()