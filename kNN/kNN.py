__author__ = 'MichaelZhao'

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #计算dataSet中点的个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #差矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1) #计算每个行向量各自的和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #返回数组值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #获得与inX第i近的点的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #label自增
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
    group, labels = createDataSet()
    print(classify0([0.3,0.2], group, labels, 3))



