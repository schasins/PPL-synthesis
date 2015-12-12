import numpy as np
import matplotlib.pyplot as plt
import os
import math

stringToSeek = "_100_"
structureGenerationStrategyNames = {"n": "Naive", "c": "Simple Correlation", "d": "Network Deconvolution"}

groundTruthScores = {'biasedtugwar': 195085.7403381909, 'csi': 36709.6095361359, 'hurricanevariation': 17043.29694642003, 'students': 77602.55741867474, 'easytugwar': 55549.3438741628, 'healthiness': 48001.39437425706, 'uniform': 53189.889449844784, 'eyecolor': 24542.81097722689, 'icecream': 79082.25931616548, 'multiplebranches': 50331.30808150756, 'burglary': 708.7526240629371, 'tugwaraddition': 81753.45016682695, 'grass': 41209.701189894804, 'mixedcondition': 41878.02072590537}

dataSets = {}

def bestScoreByTime(timeScoreData, time):
	for row in timeScoreData:
		if row[0] < time:
			print "a"

maxTime = 0
for f in os.listdir(os.getcwd()):
    if stringToSeek in f: 
        fl = open(f, "r")
        benchmarkname = (f.split("-")[0]).lower()
        structureGenerationStrategy = structureGenerationStrategyNames[f.split("_")[-2]]
        lines = fl.readlines()
        data = map(lambda line: map(float, line.strip().split(",")), lines)
        thisMaxTime = data[-1][0]
        if thisMaxTime > maxTime:
        	maxTime = thisMaxTime
        sData = dataSets.get(structureGenerationStrategy, {})
        sBData = sData.get(benchmarkname, [])
        sBData.append(data)
        sData[benchmarkname] = sBData
        dataSets[structureGenerationStrategy] = sData

"""
for strategy in dataSets:
	strategyBenchmarks = dataSets[strategy]
	for benchmarkname in strategyBenchmarks:
		benchmarkRuns = strategyBenchmarks[benchmarkname]
		currBenchmarkY = []
		for time in range(0, maxTime):
"""

def timeToReachScore(timeScoreData, score):
	for line in timeScoreData:
		if line[1] <= score:
			return line[0]
	return None

maxTimeToReachScore = 0
allBars = []
allBarErros = []
for strategy in sorted(dataSets.keys()):
	print "********************************"
	print "Strategy: "+ strategy
	print "********************************"
	strategyBenchmarks = dataSets[strategy]
	bars = []
	barErrors = []
	for benchmarkname in sorted(strategyBenchmarks.keys()):
		benchmarkRuns = strategyBenchmarks[benchmarkname]
		timeLs = []
		for run in benchmarkRuns:
			newTime = timeToReachScore(run, groundTruthScores[benchmarkname])
			if (newTime == None):
				timeLs = [-1] * len(benchmarkRuns)
				break
			timeLs.append(newTime)
		avg = np.mean(timeLs)
		stderr = np.std(timeLs)
		if avg + stderr > maxTimeToReachScore:
			maxTimeToReachScore = avg + stderr
		bars.append(avg)
		barErrors.append(stderr)
		print benchmarkname, ":", timeLs
	allBars.append(bars)
	allBarErros.append(barErrors)
print allBars

timeoutTime = maxTimeToReachScore + 20
yAxisMax = int(20 * math.floor(float(timeoutTime)/20)) # round to lower multiple of 20
for i in range(len(allBars)):
		allBars[i] = [yAxisMax if x == -1 else x for x in allBars[i]] # thse were the timeouts

x = np.array(range(len(strategyBenchmarks)))
my_xticks = sorted(strategyBenchmarks.keys()) # string labels
locs, labels = plt.xticks(x, my_xticks)

strategies = sorted(dataSets.keys())

ax = plt.subplot(111)
ax.bar(x-0.2, allBars[0],width=0.2,color='b',align='center', yerr=allBarErros[0], label=strategies[0], ecolor='k')
ax.bar(x, allBars[1],width=0.2,color='g',align='center', yerr=allBarErros[1], label=strategies[1], ecolor='k')
ax.bar(x+0.2, allBars[2],width=0.2,color='r',align='center', yerr=allBarErros[2], label=strategies[2], ecolor='k')

leg = plt.legend()
plt.setp(labels, rotation=90)
plt.gca().set_ylim(bottom=0)
plt.gca().set_ylim(top=yAxisMax)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off', labelsize=14) # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    top='off', labelsize=16) # labels along the bottom edge are off

plt.draw()

# Get the bounding box of the original legend
bb = leg.legendPatch.get_bbox().inverse_transformed(ax.transAxes)

plt.ylabel('Time in Seconds', size=18)

ax.legend(loc='upper left')

# # Change to location of the legend. 
# newX0 = 0
# newX1 = 10
# bb.set_points([[newX0, bb.y0], [newX1, bb.y1]])
# leg.set_bbox_to_anchor(bb)

fig = plt.gcf()
fig.set_size_inches(13, 8)
fig.subplots_adjust(bottom=0.2)
fig.savefig('timeToReachScore.pdf', edgecolor='none', format='pdf')
plt.close()