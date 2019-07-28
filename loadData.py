import idx2numpy as i2n
import numpy as np 
from keras.utils import np_utils

def load():
	X_train = i2n.convert_from_file("Dataset/train_images")
	y_train = i2n.convert_from_file("Dataset/train_labels")
	X_test = i2n.convert_from_file("Dataset/test_images")
	y_test = i2n.convert_from_file("Dataset/test_labels")
	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
	X_test  = X_test.reshape(X_test.shape[0], 1, 28, 28)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_test = np_utils.to_categorical(y_test, 10)
	return X_train, y_train, X_test, y_test