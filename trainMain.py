from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
from loadData import load

X_train, y_train, X_test, y_test = load()
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
for i in range(6):
	model.add(Dense(200, activation = "relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
	if(i % 3 == 0):
		model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax', input_shape=(60000, 10)))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=6)

score = model.predict(X_test)

model.save('my_model.h5')