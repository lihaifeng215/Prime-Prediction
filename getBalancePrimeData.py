import numpy as np
positive = np.loadtxt("file/PositivePrimeData_1_million.txt",delimiter=',',dtype=int)
negative = np.loadtxt("file/NegativePrimeData_1_million.txt",delimiter=',',dtype=int)
print("positive.shape:",positive.shape)
print("negative.shape:",negative.shape)

sampleNegative = []
for i in range(0,negative.shape[0],12):
    sampleNegative.append(negative[i])

sampleNegative = np.array(sampleNegative)
print("sampleNegative.shape:",sampleNegative.shape)

BalancePrimeData_1_million = np.row_stack((sampleNegative,positive))
print("BalancePrimeData_1_million.shape:",BalancePrimeData_1_million.shape)

np.random.shuffle(BalancePrimeData_1_million)
print(BalancePrimeData_1_million)
print("BalancePrimeData_1_million.shape:",BalancePrimeData_1_million.shape)
np.savetxt("file/BalancePrimeData_1_million.txt",BalancePrimeData_1_million,fmt='%d',delimiter=',')



















# import numpy as np
#
# # 太慢了，等跑完就哭了。这样不行!
#
#
# data = np.loadtxt("file/primes_in_10_million.txt",delimiter=' ',dtype=int)
# print("prime number:",data.shape)
# print(data)
# # 生成一千万内的数字
# numberTenMillion = np.array([i for i in np.arange(2,10000000)],dtype=int)
# # 给数字加上类标（质数为1，合数为0）
# zeros = np.zeros(9999998,dtype=int)
# trainingData = np.column_stack((numberTenMillion.T,zeros))
# print("trainingData.shape:",trainingData.shape)
# print(trainingData)
#
# # # 将质数后的label改为1
# # for i in range(trainingData.shape[0]):
# #     print(i)
# #     if trainingData[i][0] in set(data):
# #         trainingData[i][1] = 1
#
# # 检查修改结果
# print(trainingData[trainingData[:,1]==1,:].shape)
#
# # 保存数据集
# np.savetxt("file/trainingPrimeData_10_million.txt",trainingData,fmt='%d',delimiter=",")
#
#
