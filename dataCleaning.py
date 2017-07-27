import csv
from datetime import datetime
from filePathAndEpisodes import *

def getTimeDate(s) :
    dt = datetime.strptime(s, "%b %d, %Y %I:%M:%S %p")
    return dt

def cleanData(inputReader, outputWriter, header) :
    previousDateTime = "Jan 1, 9999 01:01:01 AM"
    for row in inputReader :
        if header :
            outputWriter.writerow((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[22], row[23], row[24], row[25], row[26], row[27], row[34], row[35], row[36], row[37], row[38], row[39]))
            header = False
        elif row[35] == "LOCKED" :
            try :
                gsr = int(row[27])
                if gsr <= 300000 :
                    dt1 = getTimeDate(previousDateTime)
                    dt2 = getTimeDate(row[1])
                    outputWriter.writerow((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[22], row[23], row[24], row[25], row[26], row[27], row[34], row[35], row[36], row[37], row[38], row[39]))
                    previousDateTime = row[1]
            except ValueError :
                continue

def doCleaning(pid, inputFilesList) :
    outputFilePath = "PID_" + str(pid) + "/intermediate_files/PID_" + str(pid) + "_Cleaned.csv"
    outputFile = open(outputFilePath, 'w')
    outputWriter =  csv.writer(outputFile)
    header = True
    for filePath in inputFilesList :
        inputFile = open(filePath, 'r')
        inputReader = csv.reader(inputFile)
        cleanData(inputReader, outputWriter, header)
        header = False
        inputFile.close()
    outputFile.close()
                
# doCleaning(1, PID_1_FILELIST)        
