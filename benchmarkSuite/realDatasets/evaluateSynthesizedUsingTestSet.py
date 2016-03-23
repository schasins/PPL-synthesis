import sys
import subprocess
import numpy as np

import os.path

debug = False

synthesizedProgramFilename = sys.argv[1]
testDataFilename = sys.argv[2]

modelStr = open(synthesizedProgramFilename, "r").read()

testData = open(testDataFilename, "r").readlines()
headers = testData[0].split(",")
testData = testData[1:]

guessesFileName = "guesses/"+synthesizedProgramFilename.split("/")[-1]+"_"+testDataFilename.split("/")[-1]+".guesses"

print os.path.isfile(guessesFileName)

if not os.path.isfile(guessesFileName):
    
    savedGuesses = open(guessesFileName, "w")

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

        tmpModelFilename = "tmp/"+synthesizedProgramFilename.split("/")[-1]+"_tmpModel.blog" 
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
        savedGuesses.write(str(mean)+"\n")

    savedGuesses.close()

f = open(guessesFileName, "r")
guesses = map(lambda x: float(x.strip()), f.readlines())


difs = []
for i in range(len(testData)):
    dataRow = testData[i]
    values = dataRow.split(",")
    guess = guesses[i]
    if debug: print guess

    actualOutput = float(values[-1].strip())
    if debug: print actualOutput

    dif = abs(actualOutput - guess)
    if debug: print dif
    difs.append(dif)

strs = ["******************", synthesizedProgramFilename]


strs.append("min error: "+str(min(difs)))
strs.append("max error: "+str(max(difs)))
avg = float(sum(difs))/len(difs)
strs.append("mean error: "+str(avg))
squaredErrors = map(lambda x: x**2, difs)
meanSquareError = sum(squaredErrors)/len(squaredErrors)
strs.append("mean square error: "+str(meanSquareError))
rootMeanSquareError = meanSquareError ** .5
strs.append("root mean square error: "+str(rootMeanSquareError))

print "\n".join(strs)
