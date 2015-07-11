# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
from operator import itemgetter
#change all pixel greater than 9 to 1
def changeToInt(ma):
    ma = np.mat(ma)
    m, n = np.shape(ma)
    ret = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ret[i, j] = int(ma[i,j])
            if ret[i, j] > 9:
                ret[i, j] = 1
            else:
                ret[i, j] = 0
    return ret

def readTrainData():
    with open('train.csv', 'r') as file:
        lines = csv.reader(file)
        l = []
        for line in lines:
            l.append(line)
    del l[0]
    l = np.mat(l)
    label = l[:,0]
    data = l[:,1:]
    print("data size:",np.shape(data))
    print("label size:", np.shape(label))
    return changeToInt(data), label

#use l2 distance
def knnClassify(data, label, X, k):
    X = np.mat(X)
    diffMat = np.tile(X, (data.shape[0], 1))-data
    sqDiff = np.array(diffMat)**2
    distance = sqDiff.sum(axis = 1)**0.5
    sortedDistIndex = distance.argsort()
    classCount = dict()
    for i in range(k):
        cc = label[sortedDistIndex[i], 0]
        if cc in classCount:
            classCount[cc] += 1
        else:
            classCount[cc] = 1
    sortedClassCount = sorted(classCount.items(), key = itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
    
def readTestData():
    with open('test.csv', 'r') as file:
        lines = csv.reader(file)
        count = 0
        for line in lines:
#skip the first tag line
            if count == 0:
                count = 1
                continue
            else:
                yield line


#note newline='' must be added in order to eliminate the extra blank lines
def writeTestData(result):
    with open('result.csv', 'w',newline='') as resultFile:
        csv_writer = csv.writer(resultFile)
        csv_writer.writerow(['ImageId', 'Label'])
        id = 1
        for i in result:
            csv_writer.writerow([id, i])
            id += 1
 
            
def main():
    trainData, trainLabel = readTrainData()
    print("train data loaded\n!")
    testData = readTestData()
    result = []
    counter = 0
    for line in testData:
        line = np.array(line)
        line = changeToInt(line)
        result.append(knnClassify(trainData, trainLabel, line, 5))
    #show sth to prove it's still alive
        print("{0} round finished!\n".format(counter))
        counter += 1
    writeTestData(result)

if __name__ == "__main__":
    main()