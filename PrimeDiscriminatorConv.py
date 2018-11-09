import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Conv2D,Conv2DTranspose,Activation,Reshape,Flatten
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import time

start = time.time()
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
tf.Session(config=tfconfig)

# super parameter:
learningRate = [0.1,0.01,0.001,0.0001,0.00001,0.000001]
batchSize = 5000
epochs = 5000
activation = "elu"
loss = "categorical_crossentropy"

print("learningRate:",learningRate)
print("batchSize   :",batchSize)
print("epochs      :",epochs)
print("activation  :",activation)
print("loss        :",loss)

# load data
data = np.loadtxt("file/BalancePrimeData_1_million.txt",delimiter=',',dtype=int)
trainData_X = data[:-10000,0]
trainData_Y = data[:-10000,1]
testData_X = data[-10000:,0]
testData_Y = data[-10000:,1]
trainData_Y = to_categorical(trainData_Y)
testData_Y = to_categorical(testData_Y)

# 建立模型
model = Sequential()
model.add(Reshape((1,1,1),input_shape=(1,)))
# model.add(BatchNormalization())
model.add(Conv2DTranspose(5,(3,3),activation=activation))
model.add(BatchNormalization())
model.add(Conv2DTranspose(5,(3,3),activation=activation))
model.add(BatchNormalization())
model.add(Conv2D(5,(3,3),activation=activation))
model.add(BatchNormalization())
model.add(Conv2D(5,(3,3),activation=activation))
model.add(BatchNormalization())
model.add(Conv2D(2,(1,1),activation=activation))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Activation("softmax"))

for i in learningRate:
    print('learning rate for latitude is :', i)
    # 编译模型
    model.compile(optimizer=keras.optimizers.Adam(lr=i), loss=loss)
    if i != 0.1:
        # load parameter
        model.load_weights('logConv/PrimeDiscriminatorConv.hdf5')
        print("model compiled!")

    # save model
    tensorboard = keras.callbacks.TensorBoard(log_dir='logConv', write_images=True, histogram_freq=0)
    logger = keras.callbacks.CSVLogger('logConv/log.csv', separator=',', append=False)
    earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')
    model_saver = keras.callbacks.ModelCheckpoint('logConv/PrimeDiscriminatorConv.hdf5', monitor='loss', verbose=2,
                                                      save_best_only=True,
                                                      save_weights_only=True, mode='auto', period=1)

    # training
    model.fit(trainData_X,trainData_Y,batch_size=batchSize,epochs=epochs,verbose=2,validation_data=[testData_X,testData_Y],callbacks=[tensorboard,logger,model_saver])

    # testing
    output = model.predict(testData_X)
    print("the output is:\n",output)
    print("the result:",np.mean(np.abs(output-testData_Y)))

print("use time:%.2fmins" %((time.time()-start)/60))















# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense,BatchNormalization,Conv2D,Conv2DTranspose,Activation,Reshape,Flatten
# from keras.utils.np_utils import to_categorical
# import tensorflow as tf
# import time
#
# start = time.time()
# tfconfig = tf.ConfigProto(allow_soft_placement=True)
# tfconfig.gpu_options.allow_growth = True
# tf.Session(config=tfconfig)
#
# # super parameter:
# learningRate = 0.1
# batchSize = 5000
# epochs = 500
# activation = "elu"
# loss = "categorical_crossentropy"
#
# print("learningRate:",learningRate)
# print("batchSize   :",batchSize)
# print("epochs      :",epochs)
# print("activation  :",activation)
# print("loss        :",loss)
#
# # 建立模型
# model = Sequential()
# model.add(Reshape((1,1,1),input_shape=(1,)))
# model.add(BatchNormalization())
# model.add(Conv2DTranspose(5,(3,3),activation=activation))
# model.add(BatchNormalization())
# model.add(Conv2DTranspose(5,(3,3),activation=activation))
# model.add(BatchNormalization())
# model.add(Conv2D(5,(3,3),activation=activation))
# model.add(BatchNormalization())
# model.add(Conv2D(5,(3,3),activation=activation))
# model.add(BatchNormalization())
# model.add(Conv2D(2,(1,1),activation=activation))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Activation("sigmoid"))
#
# # 编译模型
# model.compile(optimizer=keras.optimizers.Adam(lr=learningRate),loss=loss)
# print("model compiled!")
#
# # save model
# tensorboard = keras.callbacks.TensorBoard(log_dir='logConv', write_images=True, histogram_freq=0)
# logger = keras.callbacks.CSVLogger('logConv/log.csv', separator=',', append=False)
# earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')
# model_saver = keras.callbacks.ModelCheckpoint('logConv/PrimeDiscriminatorConv.hdf5', monitor='loss', verbose=2,
#                                                   save_best_only=True,
#                                                   save_weights_only=True, mode='auto', period=1)
#
#
# data = np.loadtxt("file/BalancePrimeData_1_million.txt",delimiter=',',dtype=int)
# trainData_X = data[:-10000,0]
# trainData_Y = data[:-10000,1]
# testData_X = data[-10000:,0]
# testData_Y = data[-10000:,1]
# trainData_Y = to_categorical(trainData_Y)
# testData_Y = to_categorical(testData_Y)
#
# model.fit(trainData_X,trainData_Y,batch_size=batchSize,epochs=epochs,verbose=2,callbacks=[tensorboard,logger,model_saver])
# output = model.predict(testData_X)
# print("the output is:",output)
# print("the result:",np.mean(np.abs(output-testData_Y)))
# print("use time:%.2fmins" %((time.time()-start)/60))