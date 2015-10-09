import sys

inputfilename = sys.argv[1]
outputfilename = sys.argv[2]

file = open(inputfilename,'r')
output = open(outputfilename,'w')

lines = file.readlines()

#skip over whatever was put in the header stuff
counter = 0
while lines[counter] != "\n":
    counter += 1

#get labels
counter2 = counter+1
while lines[counter2] != "\n":
    line = lines[counter2]
    if " = " in line:
        line = line.strip()
        ls = line.split(" = ")
        label = ls[0]
        output.write(label+",")
    counter2 += 1
output.write("\n")

for i in range(counter+1,len(lines)):
    line = lines[i]
    if " = " in line:
        line = line.strip()
        ls = line.split(" = ")
        val = ls[1]
        output.write(val+",")
    if line == "\n":
        output.write("\n")
file.close()
output.close()
        
