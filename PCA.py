from DataGen import Get_Data_Class
import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, data, k=4):
        self.k = k
        self.data = data
        pass

    def run(self):
        # 获取数据
        # shape: 1024 * 8
        m, n = self.data.shape
        # 进行数据的居中
        MEAN = self.mean(self.data)
        # shape: 1024 * 8
        data = self.data - MEAN
        # 计算居中矩阵的协方差矩阵
        # shape: 8*8
        covX = np.cov(data.T)
        featValue, featVec = np.linalg.eig(covX)
        index = np.argsort(-featValue)
        finalData = []
        if self.k > n:
            print("K 应该小于 n")
            return
        # shape: k*8
        selectVec = np.matrix(featVec.T[index[:self.k]])
        # 1024*8 * 8*k -> shape: 1024 * k
        finalData = data * selectVec.T
        # 1024*k * k*8 -> shape 1024 * 8
        reconData = (finalData * selectVec) + MEAN
        return finalData, reconData


    def mean(self, data):
        return np.mean(data)


data = Get_Data_Class()
pca = PCA(data=data)
print(data)

for i in range(1, 9):
    print("K = {0}".format(i))
    # shape: 1024 * k, 1024 * 8
    pca.k = i
    finalData, reconData = pca.run()
    np.savetxt("PCA_k_{}.txt".format(i), finalData)
    # 计算差异
    print(np.sum(np.square(reconData-data)))
