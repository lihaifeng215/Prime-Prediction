import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Conv2D,Conv2DTranspose,Activation
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import time

start = time.time()
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
tf.Session(config=tfconfig)

# super parameter:
learningRate = 0.1
batchSize = 500
epochs = 50
activation = "linear"
loss = "categorical_crossentropy"

print("learningRate:",learningRate)
print("batchSize   :",batchSize)
print("epochs      :",epochs)
print("activation  :",activation)
print("loss        :",loss)
model = Sequential()
model.add(Dense(5,activation=activation,input_dim=1))
# model.add(BatchNormalization())
model.add(Dense(3,activation=activation))
model.add(Dense(2,activation='sigmoid'))
# model.add(Activation("sigmoid"))


model.compile(optimizer=keras.optimizers.Adam(lr=learningRate),loss=loss)
print("model compiled!")

data = np.loadtxt("file/BalancePrimeData_1_million.txt",delimiter=',',dtype=int)
trainData_X = data[:-10000,0]
trainData_Y = data[:-10000,1]
testData_X = data[-10000:,0]
testData_Y = data[-10000:,1]
trainData_Y = to_categorical(trainData_Y)
testData_Y = to_categorical(testData_Y)

model.fit(trainData_X,trainData_Y,batch_size=batchSize,epochs=epochs,verbose=2)
output = model.predict(testData_X)
print("the output is:",output)
print("the result:",np.mean(np.abs(output-testData_Y)))
print("use time:%.2fmins" %((time.time()-start)/60))
































# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense,BatchNormalization,Conv2D,Conv2DTranspose
# from keras.utils.np_utils import to_categorical
# import tensorflow as tf
# import time
#
# start = time.time()
# tfconfig = tf.ConfigProto(allow_soft_placement=True)
# tfconfig.gpu_options.allow_growth = True
# tf.Session(config=tfconfig)
#
# model = Sequential()
# model.add(Dense(20,activation='relu',input_dim=1))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(5,activation='relu'))
# # model.add(BatchNormalization())
# model.add(Dense(2,activation='relu'))
#
# model.compile(optimizer=keras.optimizers.Adam(),loss='categorical_crossentropy')
# print("model compiled!")
#
# data = np.loadtxt("file/BalancePrimeData_1_million.txt",delimiter=',',dtype=int)
# trainData_X = data[:-10000,0]
# trainData_Y = data[:-10000,1]
# testData_X = data[-10000:,0]
# testData_Y = data[-10000:,1]
# trainData_Y = to_categorical(trainData_Y)
# print("trainData_Y:",trainData_Y)
# testData_Y = to_categorical(testData_Y)
# print("testData_Y:",testData_Y)
#
# model.fit(trainData_X,trainData_Y,batch_size=1000,epochs=500,validation_split=0.2,verbose=2)
# output = model.predict(testData_X)
# print("the output is:",output)
# print("the result:",np.mean(np.abs(output-testData_Y)))
# print("use time:%.2fmins" %((time.time()-start)%60))
