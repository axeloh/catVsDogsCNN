import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2 # For image operations
import random
import pickle

trainPath = '/Users/axeloh/Koding/machine_learning/datasets/train/'
#testPath = '/Users/axeloh/Koding/machine_learning/datasets/test/'


'''
for img in os.listdir(trainPath):
	imgArray = cv2.imread(trainPath+img, cv2.IMREAD_GRAYSCALE)
	print(img)
	plt.imshow(imgArray, cmap="gray")
	plt.show()
	break

print(imgArray)


IMG_SIZE = 50
newArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))
# After resize
plt.imshow(newArray, cmap="gray")
plt.show()
'''

IMG_SIZE = 50

def preProcess(path):
	data = []
	for img in os.listdir(path):
		try:
			imgArray = cv2.imread(path+img, cv2.IMREAD_GRAYSCALE)
			newArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))
			classNum = 1 #1 for dog
			if img[:1] == 'c':
				classNum = 0 #0 for cat
			data.append([newArray, classNum])
		except Exception as e:
			print("Problem reading image.")
			pass
	return data


# ------ MAIN --------
trainingData = preProcess(trainPath)
#testData = preProcess(testPath)
print(len(trainingData))
#print(len(testData))

random.shuffle(trainingData)
#random.shuffle(testData)

trainX = []
trainY = []

#testX = []
#testY = []

for features, label in trainingData:
	trainX.append(features)
	trainY.append(label)


trainX = np.array(trainX).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#testX = np.array(testX).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


trainX = np.array(trainX)
trainY = np.array(trainY)
#testX = np.array(testX)


trainX = trainX/255.0
#testX = testX/255.0

print(trainX.shape)
print(trainY.shape)


print("Saving data..")

np.save('./trainX.npy', trainX)
np.save('./trainY.npy', trainY)

#np.save('./testX.npy', testX)


'''
pickleOut = open("trainX.pickle", "wb")
pickle.dump(trainX, pickleOut)
pickleOut.close()

pickleOut = open("trainY.pickle", "wb")
pickle.dump(trainY, pickleOut)
pickleOut.close()

pickleOut = open("testX.pickle", "wb")
pickle.dump(testX, pickleOut)
pickleOut.close()

pickleOut = open("testY.pickle", "wb")
pickle.dump(testY, pickleOut)
pickleOut.close()



pickleIn = open("trainX.pickle", "rb")
trainX = pickle.load(pickleIn)
print(trainX[0])
'''


