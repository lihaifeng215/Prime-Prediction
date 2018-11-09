import numpy as np
data = np.loadtxt("file/PrimeData_10_million.txt",delimiter=',',dtype=int)
# 将样本正例和负例分开
positiveData = data[data[:,1]==1,:]
print(positiveData)
print("positive data number:",positiveData.shape[0])
negativeData = data[data[:,1]==0,:]
print("positive data number:",negativeData.shape[0])

np.savetxt("file/PositivePrimeData_10_million.txt",positiveData,delimiter=',',fmt='%d')
np.savetxt("file/NegativePrimeData_10_million.txt",negativeData,delimiter=',',fmt='%d')