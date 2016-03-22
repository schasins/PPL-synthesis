from astDB import *
import subprocess
import math
import sys

def generateQueryString(dataset):
    varNames = dataset.indexesToNames
    rows = dataset.literalRows
    varTypes = dataset.columnDistributionInformation

    # want to make query items like this:
    # query Burglary == false & Earthquake == false & Alarm == true & JohnCalls == false & MaryCalls == true;
    queryStr = ""
    for row in rows:
        equalities = []
        for i in range(len(varNames)):
            if isinstance(varTypes[i], RealDistribution):
                equalities.append(varNames[i]+" > "+str(float(row[i])-.1)+" & "+varNames[i]+" < "+str(float(row[i])+.1))
            else:
                equalities.append(varNames[i]+" == "+str(row[i]))
        queryStr += "query " + (" & ".join(equalities)) + ";\n"
    return queryStr

def parseBLOGOutputForSumLikelihood(blogOutputStr, numRows):
    results = blogOutputStr.split("======== Query Results =========")[1]
    logLikelihood = 0.0
    # any line after the query results point that starts with true is the likelihood of one row
    # for any one that has no true entry (only false), the likelihood is 0, so it's ok to just not advance the sum
    lines = results.split("\n")
    numRowsFound = 0
    for line in lines:
        cleanLine = line.strip()
        if cleanLine.startswith("true"):
            numRowsFound += 1
            num = float(cleanLine.split("\t")[1])
            logNum = math.log(num)
            logLikelihood += logNum
    if numRowsFound < numRows:
        print "SOME ROWS HAD 0 PROBABILITY.", numRows - numRowsFound
        diff = numRows - numRowsFound
        for i in range(diff):
            logLikelihood += math.log(.0000000000000000000000000000000000001)
    return logLikelihood

def blogLikelihoodScore(ast, dataset, filename):
    # generate the query items we need to add to the program to get BLOG to calculate likelihood for us
    queryStr = generateQueryString(dataset)

    # make a file that has the program text and the queries
    programStr = ast.programString()
    tmpProgFile = open(filename, "w")
    tmpProgFile.write(programStr + "\n" + queryStr)
    tmpProgFile.close()

    # run the new BLOG program from the file

    command = "blog -n 10000 "+filename+" -s blog.sample.MHSampler"
    print command
    
    #strOutput = subprocess.check_output(("blog -n 20000 "+filename).split(" "))
    strOutput = subprocess.check_output(command.split(" "))

    # parse the BLOG program's output
    logLikelihood = parseBLOGOutputForSumLikelihood(strOutput, dataset.numRows)
    return logLikelihood

def blogLikelihoodScoreForExistingProg(blogWithQueriesFileName, numRows):
    strOutput = subprocess.check_output(("blog -n 10000 "+blogWithQueriesFileName+" -s blog.sample.MHSampler").split(" "))

    # parse the BLOG program's output
    logLikelihood = parseBLOGOutputForSumLikelihood(strOutput, numRows)
    print blogWithQueriesFileName.split("/")[-1].split(".")[0], ",", logLikelihood

if __name__ == "__main__":
    filename = sys.argv[1]
    numRows = int(sys.argv[2])
    blogLikelihoodScoreForExistingProg(filename, numRows)




