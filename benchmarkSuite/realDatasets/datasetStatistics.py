import sys

datasetFile = sys.argv[1]

f = open(datasetFile, "r")
lines = f.readlines()

headers = lines[0].split(",")
data = map(lambda x: x.split(","), lines[1:])

for i in range(len(headers)):
    print headers[i]
    try:
        float(data[0][i])
    except:
        print "not numeric"
        print "ex value: ", data[0][i]
        continue
    values = map(lambda x: float(x[i]), data)
    print "min: ", min(values)
    print "max: ", max(values)
    avg = float(sum(values))/len(values)
    print "avg: ", avg
    print "mode: ", max(set(values), key=values.count)
    #predictionErrors = map(lambda x: x - avg, values)
    predictionErrors = map(lambda x: x - 0, values)
    print "mean error: ", sum(map(lambda x: abs(x), predictionErrors))/len(predictionErrors)
    squaredErrors = map(lambda x: x**2, predictionErrors)
    meanSquareError = sum(squaredErrors)/len(squaredErrors)
    print "mean square error: ", meanSquareError
    rootMeanSquareError = meanSquareError ** .5
    print "root mean square error: ", rootMeanSquareError
