import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from filePathAndEpisodes import *
from dataCleaning import *
from dataProcessing import *

def getTimeDate(s) :
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt

def timeInValidEpisode(startTime, endTime, time) :
    startTimeDate = getTimeDate(startTime)
    endTimeDate = getTimeDate(endTime)
    timeDate = getTimeDate(time)
    return timeDate >= startTimeDate and timeDate < endTimeDate

def getMeanOfEmotions(episodes) :
    n = len(episodes)
    afraidArray = np.zeros((n))
    worriedArray = np.zeros((n))
    frustratedArray = np.zeros((n))
    hassledArray = np.zeros((n))
    i = 0
    for episode in episodes :
        afraidArray[i] = episode[2]
        worriedArray[i] = episode[3]
        frustratedArray[i] = episode[4]
        hassledArray[i] = episode[5]
        i += 1
    return [np.mean(afraidArray), np.mean(worriedArray), np.mean(frustratedArray), np.mean(hassledArray)]
        

def getLabel(afraid, worried, frustrated, hassled, meanList) :
    if afraid > meanList[0] :
        return 1
    if worried > meanList[1] :
        return 1
    if frustrated > meanList[2] :
        return 1
    if hassled > meanList[3] :
        return 1
    return 0    
        

def getValidSampleSet(inputReader, episodes) :
    meanList = getMeanOfEmotions(episodes)
    print("Emotion means: " + str(meanList))
    header = True
    attrList = []
    labelList = []
    for row in inputReader :
        if header :
            header = False
        else :
            curTime = row[0]
            for episode in episodes :
                if timeInValidEpisode(episode[0], episode[1], curTime) :
                    attr = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                    label = getLabel(episode[2], episode[3], episode[4], episode[5], meanList)
                    attrList.append(attr)
                    labelList.append(label)
                    break
    return [attrList, labelList]

def kFoldTraining(attrList, labelList, k, trainingMethod) :
    X = np.array(attrList)
    Y = np.array(labelList)
    cv = KFold(n_splits = k)
    clfSVM = SVC()
    clfNN = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,), random_state = 1)
    accuracyList = []
    for train_index, test_index in cv.split(X) :
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        pred = None
        try :
            if trainingMethod == "SVM" :
                clfSVM.fit(x_train, y_train)
                pred = clfSVM.predict(x_test)
            elif trainingMethod == 'NN':
                clfNN.fit(x_train, y_train)
                pred = clfNN.predict(x_test)
            else :
                clfSVM.fit(x_train, y_train)
                pred = clfSVM.predict(x_test)
        except ValueError :
            print(ValueError)
            return -1
        
        accuracyList.append(accuracy_score(np.array(y_test), np.array(pred)) * 100)
    accuracy = np.mean(np.array(accuracyList))
    print(str(k) + " Fold Accuracy: " + str(accuracy))
    return accuracy
    
def getConfusionMatrix(attrList, labelList, k, outputWriter, trainingMethod) :
    X = np.array(attrList)
    Y = np.array(labelList)
    cv = KFold(n_splits = k)
    clfSVM = SVC()
    clfNN = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,), random_state = 1)
    for train_index, test_index in cv.split(X) :
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        pred = None
        try :
            if trainingMethod == "SVM" :
                clfSVM.fit(x_train, y_train)
                pred = clfSVM.predict(x_test)
            elif trainingMethod == "NN" :
                clfNN.fit(x_train, y_train)
                pred = clfNN.predict(x_test)
            else :
                clfSVM.fit(x_train, y_train)
                pred = clfSVM.predict(x_test)
        except ValueError :
            print(ValueError)
        
        mat = confusion_matrix(y_test, pred, labels = [1, 2, 3, 4, 5, 6])
        print(mat)
        outputWriter.write(str(mat) + '\n')
    
    
def training(attrList, labelList, outputWriter, trainingMethod) :
    accuracyList = []
    kFolds = []
    maxAcc = 0
    maxargK = 0
    
    for i in range(2, len(labelList) + 1) :
        acc = kFoldTraining(attrList, labelList, i, trainingMethod)
        if acc > maxAcc :
            maxAcc = acc
            maxargK = i
        accuracyList.append(acc)
        kFolds.append(i)
        outputWriter.write(str(i) + " Fold Accuracy: " + str(acc) + '\n')
    print("maximized at " + str(maxargK) + ": accuracy " + str(maxAcc))
    outputWriter.write("maximized at " + str(maxargK) + ": accuracy " + str(maxAcc))    
    
    # getConfusionMatrix(attrList, labelList, 325, outputWriter, trainingMethod)
    
def doAnalysing(pid, validEpisodes, diff, trainingMethod) :
    processedFilePath = "PID_" + str(pid) + "/intermediate_files/PID_" + str(pid) + "_Processed_" + str(diff) + "_min.csv"
    processedFile = open(processedFilePath, 'r')
    inputReader = csv.reader(processedFile)
    outputFilePath = "PID_" + str(pid) + "/results_files/" + str(diff) + "_min_" + trainingMethod + ".txt"
    outputFile = open(outputFilePath, 'w')
    
    getMeanOfEmotions(validEpisodes)
    result = getValidSampleSet(inputReader, validEpisodes)
    print("Labels: " + str(result[1]))
    training(result[0], result[1], outputFile, trainingMethod)
    
    processedFile.close()
    outputFile.close()
    
def analyseAll(diff, trainingMethod) :    
    filelistList = [PID_1_FILELIST, PID_2_FILELIST, PID_3_FILELIST, PID_4_FILELIST, PID_5_FILELIST, PID_6_FILELIST, PID_7_FILELIST, PID_8_FILELIST]
    episodeList = [PID_1_VALID_EPISODES, PID_2_VALID_EPISODES, PID_3_VALID_EPISODES, PID_4_VALID_EPISODES, PID_5_VALID_EPISODES, PID_6_VALID_EPISODES, PID_7_VALID_EPISODES, PID_8_VALID_EPISODES]
    for i in range(1, 9) :
        # print("-------PID " + str(i) + " Started Cleaning-------")
        # doCleaning(i, filelistList[i - 1])
        # print("-------PID " + str(i) + " Finished Cleaning-------")
        print("-------PID " + str(i) + " Started Processing-------")
        doProcessing(i, diff)
        print("-------PID " + str(i) + " Finished Processing-------")
        print("-------PID " + str(i) + " Started Analysing-------")
        doAnalysing(i, episodeList[i - 1], diff, trainingMethod)

analyseAll(1, "SVM")
    

        
