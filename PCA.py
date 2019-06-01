from DataGen import Get_Data_Class
import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, k=4):
        self.k = k
        pass

    def run(self):
        # 获取数据
        data = Get_Data_Class()
        m, n = data.shape
        # 进行数据的居中
        MEAN = self.mean(data)
        data = data - MEAN
        # 计算居中矩阵的协方差矩阵
        covX = np.cov(data)
        featValue, featVec = np.linalg.eig(covX.T)
        index = np.argsort(-featValue)
        finalData = []
        if self.k > n:
            print("K 应该小于 n")
            return
        selectVec = np.matrix(featVec.T[index[:self.k]])
        finalData = data.T * selectVec.T
        reconData = (finalData * selectVec) + MEAN
        return finalData, reconData


    def mean(self, data):
        return np.mean(data)


pca = PCA()
finalData, reconData = pca.run()
# print(reconData)
# data = Get_Data_Class()
# print(np.sum(np.square(reconData-data.T)))
print(reconData.shape)

