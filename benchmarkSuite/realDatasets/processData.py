f = open("airlineDelayData.csv","r")
fOut = open("airlineDelayDataProcessed.csv", "w")
fOut2 = open("airlineDelayDataProcessedCategorical.csv", "w")

fOutString = ""
fOutString2 = ""
origLines = 0
outputLines = 0
lines = f.readlines()
titles = lines[0]
titlesCells = titles.split(",")

dict = {"DAY_OF_MONTH": "m", "DAY_OF_WEEK": "d", "ORIGIN_AIRPORT_ID": "oa", "DEST_AIRPORT_ID": "da"}
indexesToModify = {}
for key in dict:
    index = titlesCells.index(key)
    indexesToModify[index] = dict[key]
print indexesToModify

fOutString += titles
fOutString2 += titles

for line in lines[1:]:
    origLines += 1
    cells = line.split(",")
    if not "" in cells:
        fOutString += line
        for index in indexesToModify:
            cells[index] = indexesToModify[index]+cells[index]
        fOutString2 += ",".join(cells)
        outputLines += 1

fOut.write(fOutString)
fOut2.write(fOutString2)
f.close()
fOut.close()
fOut2.close()

print "Lines in original file:", origLines
print "Lines in output file:", outputLines
print "Lines removed for lacking an entry:", origLines - outputLines
