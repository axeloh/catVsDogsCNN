import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
import numpy as np
import pickle
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l1, l2
import time

trainX = np.load('trainXAugmented.npy')
trainY = np.load('trainYAugmented.npy')

print(trainX.shape)
print(trainY.shape)
# When testing many combinations
#convLayers = [1, 2, 3]
#layerSizes = [32, 64, 128]
#denseLayers = [0, 1, 2]

# When tweaking best models
convLayers = [1]
layerSizes = [32, 128]
denseLayers = [1]
droprates = [0, 0.2]

for denseLayer in denseLayers:
	for layerSize in layerSizes:
		for convLayer in convLayers:
			for droprate in droprates:
				name = "({}-conv_{}-units)-({}-dense_{}-units)-({}-droprate)-{}".format(convLayer, layerSize, denseLayer, 512, droprate, int(time.time()))
				tensorboard = TensorBoard(log_dir='modelsTraindataAugmented/{}'.format(name))
				
				model = Sequential()
				
				model.add(Conv2D(layerSize, (3,3), input_shape=trainX.shape[1:]))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				for l in range(convLayer-1):
					model.add(Conv2D(layerSize, (3,3)))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Flatten())

				for l in range(denseLayer):
					model.add(Dense(512))
					model.add(Activation('relu'))
					model.add(Dropout(droprate))

				model.add(Dense(1))
				model.add(Activation('sigmoid'))


				model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
				model.fit(trainX, trainY, 
					batch_size=32, 
					epochs=10, 
					validation_split=0.1, 
					callbacks=[tensorboard])


'''
Best models:


3 conv 128
1 dense 512
0.2 dropout
7 epoker
0.4253 val loss

3 conv 128
1 dense 512
0 dropout
5 epoker
0.413 val loss

3 conv 64 
1 dense 512
0 dropout
6 epoker
0.4165 val loss


'''




# Load data
'''
trainX = pickle.load(open("trainX.pickle", "rb"))
trainY = pickle.load(open("trainY.pickle", "rb"))
testX = pickle.load(open("testX.pickle", "rb"))
testY = pickle.load(open("testY.pickle", "rb"))


trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')
#testX = np.load('testX.npy')


print(trainX.shape)
print(trainY.shape)
#print(testX.shape)







name = "Cat_vs_dog_Conv:64x2-Dense:64x1_{}".format(int(time.time()))


earlystopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))



model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=trainX.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(trainX, trainY, 
	batch_size=32, 
	epochs=10, 
	validation_split=0.1, 
	callbacks=[earlystopping, tensorboard])
'''

