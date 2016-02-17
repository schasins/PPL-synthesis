f = open("airlineDelayData.csv","r")
fOut = open("airlineDelayDataProcessed.csv", "w")

fOutString = ""
origLines = 0
outputLines = 0
for line in f.readlines():
    origLines += 1
    cells = line.split(",")
    if not "" in cells:
        fOutString += line
        outputLines += 1

fOut.write(fOutString)
f.close()
fOut.close()

print "Lines in original file:", origLines
print "Lines in output file:", outputLines
print "Lines removed for lacking an entry:", origLines - outputLines
