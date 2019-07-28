from keras.models import load_model
from keras.utils import np_utils
from loadData import load

X_train, y_train, X_test, y_test = load()

model = load_model("my_model.h5")

score = model.predict(X_test)

c = 0
for i in range(10000):
	pos = 0
	maxEle = -1.00
	for j in range(len(score[i])):
		if(score[i][j] > maxEle):
			maxEle = score[i][j]
			pos = j
	if(pos == y_test[i]):
		c += 1
print((c / 10000) * 100)