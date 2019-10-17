

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

trainPath = '/Users/axeloh/Koding/machine_learning/datasets/train/'
testPath = '/Users/axeloh/Koding/machine_learning/datasets/test/'



i = 0
trainX = []
trainY = []
lens = []
for file in os.listdir(trainPath):
	filename = os.fsdecode(file)
	#image = Image.open(trainPath + filename)
	image = load_img(trainPath + filename, target_size=(128,128))
	array = np.asarray(image)
	array = array/255.0 # Normalize
	
	classNum = 1 #1 for dog
	if img[:1] == 'c':
		classNum = 0 #0 for cat

	trainX.append(array)
	lens.append(len(array))

	#if i == 5000:
	#	break
	#i += 1


# plot cat photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread


'''
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	# load image pixels
	# plot raw pixel data
	image = trainX[i]
	pyplot.title(trainY[i])
	pyplot.imshow(image)
# show the figure
pyplot.show()
'''
print("Done with trainset.")




'''
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

print("Saving data..")

np.save('./trainX.npy', trainX)
np.save('./trainY.npy', trainY)

np.save('./testX.npy', testX)
np.save('./testY.npy', testY)
'''


print("Done.")
