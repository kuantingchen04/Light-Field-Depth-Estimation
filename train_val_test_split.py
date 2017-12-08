
# coding: utf-8

# In[103]:

import os
import numpy as np
import sys
import collections as coll
def train_val_test_split(trainNum=400,valNum=100,testNum=100):
    datasetName = 'scene12'
    savePath    = os.path.join(os.getcwd(),'datasets',datasetName)
    loadAPath    = os.path.join(os.getcwd(),'data','A') # containing all 3600 images
    loadBPath    = os.path.join(os.getcwd(),'data','B') # containing all 3600 images
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    # load all numpy array names 
    fileNames = [name for name in os.listdir(loadAPath)]
    
    # save random indices for training, validation and testing set
    trainIdx = []
    valIdx   = []
    testIdx  = []
    
    beg = 0
    end = len(fileNames)//6
    size_per_scene = end//6
    for i in range(6):
        allidx = np.random.choice(range(beg,end+1),size_per_scene,replace=False)
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
        beg += size_per_scene*6
        end += size_per_scene*6
        
        print('size of train: {}'.format(len(trainIdx)))
        print('size of val: {}'.format(len(valIdx)))
        print('size of test: {}'.format(len(testIdx)))
        
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
        trainNum = sys.argv[1]
        valNum   = sys.argv[2]
        testNum  = sys.argv[3]
    else:
        trainNum = 2400
        valNum   = 600
        testNum  = 600
        
    train_val_test_split(trainNum,valNum,testNum)
                        
                        
                        
                        
                        
                        
                        

