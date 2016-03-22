import sys
import random

inputDatasetFilename = sys.argv[1]
testingCount = int(sys.argv[2])

lines = open(inputDatasetFilename, "r").readlines()

headers = lines[0]
data = lines[1:]

numDataPoints = len(data)
indexes = range(numDataPoints)
testSetIndexes = random.sample(indexes, testingCount)

testSetIndexes = sorted(testSetIndexes, reverse = True)

testingData = []

for index in testSetIndexes:
    testingData.append(data[index])
    del data[index]

def writeSetToFile(filename, headers, data):
    fileO = open(filename, "w")
    fileO.write(headers)
    for line in data:
        fileO.write(line)

writeSetToFile("trainingData2.csv", headers, data)
print len(data)
writeSetToFile("testingData2.csv", headers, testingData)
print len(testingData)

print numDataPoints
