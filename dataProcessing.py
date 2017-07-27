import numpy as np
import csv
from datetime import datetime
from datetime import timedelta

def getTimeDate(s) :
    dt = datetime.strptime(s, "%b %d, %Y %I:%M:%S %p")
    return dt

def newTimeToCalculate(datetime1, datetime2, diff) :
    if datetime1.day != datetime2.day :
        return True
    
    if datetime1.hour != datetime2.hour :
        if datetime2.hour - datetime1.hour != 1 :
            return True
        else :
            if datetime1.minute < (60 - diff) :
                return True
        
    t = datetime2 - datetime1
    if t.total_seconds() >= (diff * 60) :
        return True
    else :
        if t.total_seconds() < (diff * 60) :
            if datetime2.minute - datetime1.minute == diff : 
                return True
            if datetime1.minute >= (60 - diff) and datetime2.minute - datetime1.minute == (60 - diff) :
                return True
        return False

def processData(inputReader, outputWriter, header, diff) :
    previousDateTime = "Jan 1, 9999 01:01:01 AM"
    previousGSR = -100
    rrIntervalList = []
    gsrDiffList = []
    accList = []
    startTime = "Jan 1, 9999 01:01:01 AM"
    for row in inputReader :
        if header :
            header = [row[1], 'RR Interval Mean', 'RR Interval Std', 'GSR Diff Mean', 'GSR Diff Std', 'Accelerometer Mean', 'Accelerometer Std']
            outputWriter.writerow(header)
            header = False
        else :
            if startTime == "Jan 1, 9999 01:01:01 AM" :
                startTime = row[1]
                
            dt1 = getTimeDate(startTime)
            dt2 = getTimeDate(row[1])
            if newTimeToCalculate(dt1, dt2, diff) :
                if rrIntervalList and gsrDiffList and accList :
                    tempRRIntervalArray = np.array(rrIntervalList)
                    tempGSRDiffArray = np.array(gsrDiffList)
                    tempAccArray = np.array(accList)
                    outputWriter.writerow((getTimeDate(startTime), np.mean(tempRRIntervalArray), np.std(tempRRIntervalArray), np.mean(tempGSRDiffArray), np.std(tempGSRDiffArray), np.mean(tempAccArray), np.std(tempAccArray)))
                rrIntervalList = []
                gsrDiffList = []
                accList = []
                previousGSR = -100
                startTime = row[1]
            else :
                rrIntervalList.append(float(row[29]))
                accList.append(float(row[2]) + float(row[3]) + float(row[4]))
                if not previousGSR == -100 :
                    gsrDiffList.append(float(row[24]) - previousGSR)
                previousGSR = float(row[24])

def doProcessing(pid, diff) :
    cleanedFilePath = "PID_" + str(pid) + "/intermediate_files/PID_" + str(pid) + "_Cleaned.csv"
    cleanedFile = open(cleanedFilePath, 'r')
    inputReader = csv.reader(cleanedFile)
    outputFilePath = "PID_" + str(pid) + "/intermediate_files/PID_" + str(pid) + "_Processed_" + str(diff) + "_min.csv"
    outputFile = open(outputFilePath, 'w')
    outputWriter = csv.writer(outputFile)
    
    processData(inputReader, outputWriter, True, diff)
    
    cleanedFile.close()
    outputFile.close()
    
# doProcessing(5, 5)
