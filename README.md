## handle a variety of dataset, the basic idea is just to get the dataset in a convinient way and, most importantly, without any data transformation

## minist:
Get from chainer's dataset module, the resulting images is 0-255 uint8 numpy matrix, saved into pyarrow zero-copy data type
usage:
```
from dataset.minist.feed import feed
path = "./save/"
feed(feed_path=path)
```
then training data will be stored in ./save/X.pa, the training label will be stored in ./save/Y.pa

to load the data into numpy, try:
```
from dataset import pa2np
X, Y = pa2np("./save/X.pa"), pa2np("./save/Y.pa")
```

## cifar
Get from chainer's dataset moduel, the resulting image is 0-255 uint8 numpy matrix, saved into pyarrow zero-copy data type
usage:

for cifar-10
```
from dataset.cifar.feed import feed
path = "./save/"
feed(feed_path=path, dataset_type=10)
```
then training data will be stored in ./save/X_10.pa, the training label will be stored in ./save/Y_10.pa

for cifar-100
```
from dataset.cifar.feed import feed
path = "./save/"
feed(feed_path=path, dataset_type=100)
```
then training data will be stored in ./save/X_100.pa, the training label will be stored in ./save/Y_100.pa

To load the data, see that in mnist above