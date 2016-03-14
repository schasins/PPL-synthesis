import sys
import subprocess
import numpy as np

debug = False

synthesizedProgramFilename = sys.argv[1]
testDataFilename = sys.argv[2]

modelStr = open(synthesizedProgramFilename, "r").read()

testData = open(testDataFilename, "r").readlines()
headers = testData[0].split(",")
testData = testData[1:]

difs = []

for dataRow in testData:
    values = dataRow.split(",")
    obsStrings = []
    for i in range(len(values) - 1): # minus one because the last one is the one we want to estimate
        valStr = values[i]
        try:
            valStr = str(float(int(valStr)))
        except Exception as e:
            True
        obsStrings.append("obs "+headers[i]+" = "+valStr+";")

    queryStr = "query "+headers[-1].strip()+";"
    outputModelStr = modelStr+"\n\n"+"\n".join(obsStrings)+"\n\n"+queryStr

    tmpModelFilename = "tmpModel.blog" 
    tmpModel = open(tmpModelFilename, "w")
    tmpModel.write(outputModelStr)
    tmpModel.close()

    numSamples = 100
    
    strOutput = subprocess.check_output(("blog -n "+str(numSamples)+" "+tmpModelFilename).split(" "))
    samples = strOutput.split("Distribution of values for ")[1]
    valueLines = samples.split("\n")[1:-2]
    vals = map(lambda x: float(x.split("\t")[1]), valueLines)
    if len(vals) != numSamples:
        print "WRONG NUM SAMPLES"
    if debug: print vals
    mean = np.mean(vals)
    if debug: print mean
    actualOutput = float(values[-1].strip())
    if debug: print actualOutput
    dif = abs(actualOutput - mean)
    if debug: print dif
    difs.append(dif)

print "Average difference from actual value, across test data:", np.mean(difs)
