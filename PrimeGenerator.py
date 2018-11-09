import numpy as np
import time
start = time.time()

# 判断是否为质数：
def isPrime(n):
    for i in range(2, int(np.sqrt(n))+1):
        if n % i == 0:
            return 0
    return 1

primeList=[]
for i in range(2,1000000):
    if isPrime(i):
        primeList.append([i,1])
        print("prime:",i)
    else:
        primeList.append([i,0])

# 得到所有质数及其类标（质数为1，合数为0）
primeList = np.array(primeList)
# print(primeList)

# 检查结果
print("prime number:",primeList[primeList[:,1]==1,:].shape[0])

# 保存数据集
np.savetxt("file/PrimeData_1_million.txt",primeList,fmt='%d',delimiter=",")

print("use time:%.2fmins" %((time.time()-start)/60))
