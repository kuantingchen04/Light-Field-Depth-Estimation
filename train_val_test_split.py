
# coding: utf-8

# In[103]:

import os
import numpy as np
import sys
import collections as coll
def train_val_test_split(trainNum=400,valNum=100,testNum=100):
    datasetName = 'scene12_' + str(trainNum) 
    savePath    = os.path.join(os.getcwd(),'datasets',datasetName)
    loadAPath    = os.path.join(os.getcwd(),'data','A') # containing all 3600 images
    loadBPath    = os.path.join(os.getcwd(),'data','B') # containing all 3600 images
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('trainNum={} valNum={} testNum={}'.format(trainNum,valNum,testNum))
    # load all numpy array names 
    fileNames = [name for name in os.listdir(loadAPath)]
    
    # save random indices for training, validation and testing set
    trainIdx = []
    valIdx   = []
    testIdx  = []
    
    beg = 0
    end = len(fileNames)//6
    delta = end
    size_per_scene = (trainNum + valNum + testNum)//6
    print(size_per_scene)
    for i in range(6):
        print('beg:{} end: {}'.format(beg,end))
        allidx = np.random.choice(range(beg,end),size_per_scene,replace=False)
        if i < 5:
            trainIdx.extend(allidx[0:trainNum//6])
            valIdx.extend(allidx[trainNum//6:(trainNum+valNum)//6])
            testIdx.extend(allidx[(trainNum+valNum)//6:])
        else:
            trainCurrentSize = len(trainIdx)
            valCurrentSize   = len(valIdx)
            testCurrentSize  = len(testIdx)

            trainIdx.extend(allidx[0:trainNum - trainCurrentSize])
            valIdx.extend(allidx[trainNum -trainCurrentSize:(trainNum+valNum) - trainCurrentSize - valCurrentSize])
            testIdx.extend(allidx[(trainNum+valNum) - trainCurrentSize - valCurrentSize:])
        beg += delta
        end += delta 
        #end = min(len(fileNames),end)
        
        print('size of train: {}'.format(len(trainIdx)))
        print('size of val: {}'.format(len(valIdx)))
        print('size of test: {}'.format(len(testIdx)))
        
    # shuffle training idx once more
    np.random.shuffle(trainIdx)
   # print(trainIdx)
    # save training set
    saveData([fileNames[i] for i in trainIdx],'train',savePath,loadAPath)
    saveData([fileNames[i] for i in trainIdx],'train',savePath,loadBPath)

    # save val set
    saveData([fileNames[i] for i in valIdx],'val',savePath,loadAPath)
    saveData([fileNames[i] for i in valIdx],'val',savePath,loadBPath)
    
    # save test set
    saveData([fileNames[i] for i in testIdx],'test',savePath,loadAPath)
    saveData([fileNames[i] for i in testIdx],'test',savePath,loadBPath)
        


def saveData(data,name,savePath,loadPath):
    dataType = loadPath[-1] 
    np.random.shuffle(data)
    for d in data:
        img = np.load(os.path.join(loadPath,d))
        path = os.path.join(savePath,name,dataType)
        if not os.path.exists(path):
            os.makedirs(path)            
        np.save(os.path.join(path,d),img)


                        
            

if __name__ == '__main__':
    if len(sys.argv) == 4:
        trainNum = int(sys.argv[1])
        valNum   = int(sys.argv[2])
        testNum  = int(sys.argv[3])
    else:
        trainNum = 2400
        valNum   = 600
        testNum  = 600
        
    train_val_test_split(trainNum,valNum,testNum)
                        
                        
                        
                        
                        
                        
                        

