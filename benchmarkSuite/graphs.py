import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import math
import sys

debug = False

threshold = float(sys.argv[1])
interval = sys.argv[2]

if interval != "2" and interval != ".2":
    print "sorry, we can only handle intervals 2 and .2.  try again."
    exit()


structureGenerationStrategyNames = {"n": "Complete", "c": "Simple Correlation", "d": "Network Deconvolution"}
orderedStrategyNames = ["Complete", "Simple Correlation", "Network Deconvolution"]

# the scores below were calculated with the BLOG score (2)
groundTruthScores = {"mixedcondition" : -28459.8881144, "easytugwar": -432735.271522, "uniform": -433991.016223, "hurricanevariation": -15238.8490086, "eyecolor": -24451.4123102, "students": -494511.705397, "icecream": -243033.13365, "csi": -17600.7572927, "healthiness": -43402.871318, "grass": -107379.623116, "biasedtugwar": -57695.5202573, "multiplebranches": -38727.3369882, "tugwaraddition": -286552.895459, "burglary": -4769.55908746}
# the scores below were calculated with the BLOG score (.2)
groundTruthScores2 = {"biasedtugwar": -166329.566764, "burglary" : -4769.55908746, "csi" : -17600.7572927, "easytugwar" : -484656.308471, "eyecolor" : -24451.4123102, "grass" : -159582.717229, "healthiness" : -43402.871318, "hurricanevariation" : -15238.8490086, "icecream": -452748.864924, "mixedcondition" : -139665.179804, "multiplebranches" : -127227.39559, "students": -582585.4996, "tugwaraddition" : -775090.947583, "uniform": -465729.222681}

stringToSeek = "_300_"
if interval == ".2":
    stringToSeek = "_250_"
    groundTruthScores = groundTruthScores2


# the scores below were calculated with the current score estimator
groundTruthScoreEstimates = {'biasedtugwar': 189783.6036644863, 'csi': 19255.445232561193, 'hurricanevariation': 16969.33744362955, 'students': 77602.55741867358, 'easytugwar': 55549.343874179234, 'healthiness': 48520.30263919617, 'uniform': 53189.88944984452, 'eyecolor': 24531.83317257066, 'icecream': 77940.73111092376, 'multiplebranches': 50331.30808150735, 'burglary': 3001.9392799884545, 'tugwaraddition': 82914.54836550112, 'grass': 31520.418707250712, 'mixedcondition': 44412.288626323476}
# the scores below were calculated with another old score approach, before we fixed handling boolean distribs, and started using more estimation
# groundTruthScores = {'biasedtugwar': 195085.7403381909, 'csi': 36709.6095361359, 'hurricanevariation': 16969.33744363283, 'students': 77602.55741867474, 'easytugwar': 55549.3438741628, 'healthiness': 48520.30263912328, 'uniform': 53189.889449844784, 'eyecolor': 24531.8331725655, 'icecream': 77940.73111092609, 'multiplebranches': 50331.30808150756, 'burglary': 708.7526240629371, 'tugwaraddition': 81753.45016682695, 'grass': 41209.701189894804, 'mixedcondition': 41878.02072590537}
# the scores below were calculated with the old score approach, before we fixed it to handle ifs with more than 2 branches better
#groundTruthScores = {'biasedtugwar': 195085.7403381909, 'csi': 36709.6095361359, 'hurricanevariation': 17043.29694642003, 'students': 77602.55741867474, 'easytugwar': 55549.3438741628, 'healthiness': 48001.39437425706, 'uniform': 53189.889449844784, 'eyecolor': 24542.81097722689, 'icecream': 79082.25931616548, 'multiplebranches': 50331.30808150756, 'burglary': 708.7526240629371, 'tugwaraddition': 81753.45016682695, 'grass': 41209.701189894804, 'mixedcondition': 41878.02072590537}

dataSets = {}
timingDataDir = "outputs/cleanTimingData"
maxTime = 0
for f in os.listdir(os.getcwd()+"/"+timingDataDir):
    if stringToSeek in f:
        fl = open(timingDataDir+"/"+f, "r")
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

scoreData = {}
scoreDataDir = "outputs/scoreData"
for f in os.listdir(os.getcwd()+"/"+scoreDataDir):
    if stringToSeek in f and ".score" in f:
        fl = open(scoreDataDir+"/"+f, "r")
        benchmarkname = (f.split("-")[0]).lower()
        structureGenerationStrategy = structureGenerationStrategyNames[f.split("_")[-2]]
        lines = fl.readlines()
        sData = scoreData.get(structureGenerationStrategy, {})
        sBData = sData.get(benchmarkname, [])
        sBData.append(float(lines[0].strip()))
        sData[benchmarkname] = sBData
        scoreData[structureGenerationStrategy] = sData

dataSetsDatablind = {}
timingDataDir = "outputs/cleanTimingData"
for f in os.listdir(os.getcwd()+"/"+timingDataDir):
    if "_1000_" in f:
        fl = open(timingDataDir+"/"+f, "r")
        benchmarkname = (f.split("-")[0]).lower()
        structureGenerationStrategy = structureGenerationStrategyNames[f.split("_")[-2]]
        lines = fl.readlines()
        data = map(lambda line: map(float, line.strip().split(",")), lines[:-1]) # -1 bc the last line doesn't include score
        sData = dataSetsDatablind.get(structureGenerationStrategy, {})
        sBData = sData.get(benchmarkname, [])
        sBData.append(data)
        sData[benchmarkname] = sBData
        dataSetsDatablind[structureGenerationStrategy] = sData

def timeToReachScore(timeScoreData, score):
	for line in timeScoreData:
		if line[1] <= score:
			return line[0]
	return None


makeMaxTimeToReachGroundtruth2 = True
if makeMaxTimeToReachGroundtruth2:

	print "\n********************************"
	print "Timing to reach as low as "+str(threshold)+" times ground truth score: means, errors across all runs"
	print "********************************"

	maxTimeToReachScore = 0
	allBars = []
	allBarErros = []
	for strategy in orderedStrategyNames:
		if debug: print "********************************"
		if debug: print "Strategy: "+ strategy
		if debug: print "********************************"
		strategyBenchmarks = dataSets[strategy]
		bars = []
		barErrors = []
		for benchmarkname in sorted(groundTruthScoreEstimates.keys()):
			benchmarkRuns = strategyBenchmarks.get(benchmarkname, None)
			if benchmarkRuns == None:
				print "freak out freak out, no benchmark runs for", benchmarkname, strategy
				print strategyBenchmarks.keys()
				exit()
			timeLs = []
			for run in benchmarkRuns:
				newTime = timeToReachScore(run, groundTruthScoreEstimates[benchmarkname]*threshold) # for this one, we just want something close
				if (newTime == None):
					timeLs = [-1] * len(benchmarkRuns)
					break
				timeLs.append(newTime)
			avg = np.mean(timeLs)
			#print avg
			stderr = np.std(timeLs)
			if avg + stderr > maxTimeToReachScore:
				maxTimeToReachScore = avg + stderr
			bars.append(avg)
			barErrors.append(stderr)
			if debug: print benchmarkname, ":", timeLs
		allBars.append(bars)
		allBarErros.append(barErrors)
		if debug: print "mean: ", np.mean(bars)
	if debug: print allBars


	timeoutTime = maxTimeToReachScore + 20
	yAxisMax = int(20 * math.floor(float(timeoutTime)/20)) # round to lower multiple of 20
	for i in range(len(allBars)):
			allBars[i] = [yAxisMax if x == -1 else x for x in allBars[i]] # thse were the timeouts

	strategies = orderedStrategyNames
	print ' "-" '.join(['"Benchmark"'] + map(lambda x: "\""+x+"\"", strategies))
	for i in range(len(allBars[0])):
		print sorted(groundTruthScores.keys())[i]," ",
		for j in range(len(allBars)):
			print allBars[j][i], " ", allBarErros[j][i], " ",
		print

	print 

        print ",".join([""] + strategies)
        avgs = map(lambda ls: np.mean(ls), allBars)
        print ",".join(map(lambda x: str(x), avgs))
        

	x = np.array(range(len(strategyBenchmarks)))
	my_xticks = sorted(strategyBenchmarks.keys()) # string labels
	locs, labels = plt.xticks(x, my_xticks)


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
	fig.savefig('timeToReachScore2_'+str(threshold)+'.pdf', edgecolor='none', format='pdf')
	plt.close()

makeLowestScore = True
if makeLowestScore:
	print "\n********************************"
	print "BLOG-estimated likelihood score: lowest across all runs"
	print "********************************"
	highestLowesScore = 0
	allBars = []
	allBarErros = []
	for strategy in orderedStrategyNames:
		if debug: print "********************************"
		if debug: print "Strategy: "+ strategy
		if debug: print "********************************"
		strategyBenchmarks = scoreData[strategy]
		bars = []
		barErrors = []
		for benchmarkname in sorted(groundTruthScores.keys()):
			benchmarkRuns = strategyBenchmarks.get(benchmarkname, None)
			if benchmarkRuns == None:
				print "freak out freak out"
				exit()
			lowestScores = []
			lowest = min(map(abs, benchmarkRuns)) # they're negative, so we want the smallest abs value
                        #print lowest
                        #print groundTruthScores[benchmarkname]
			bars.append(lowest/abs(groundTruthScores[benchmarkname])) # normalize
			#barErrors.append(stderr)
			if debug: print benchmarkname, ":", lowest
		allBars.append(bars)
		#allBarErros.append(barErrors)

	strategies = orderedStrategyNames
	print ' '.join(['"Benchmark"'] + map(lambda x: "\""+x+"\"", strategies))
	for i in range(len(allBars[0])):
		print sorted(groundTruthScores.keys())[i]," ",
		for j in range(len(allBars)):
			print allBars[j][i], " ", 
		print

	print 


        print ",".join([""] + strategies)
        avgs = map(lambda ls: np.mean(ls), allBars)
        print ",".join(map(lambda x: str(x), avgs))

	x = np.array(range(len(strategyBenchmarks)))
	my_xticks = sorted(strategyBenchmarks.keys()) # string labels
	locs, labels = plt.xticks(x, my_xticks)


	ax = plt.subplot(111)
	ax.bar(x-0.2, allBars[0],width=0.2,color='b',align='center', label=strategies[0], ecolor='k')
	ax.bar(x, allBars[1],width=0.2,color='g',align='center', label=strategies[1], ecolor='k')
	ax.bar(x+0.2, allBars[2],width=0.2,color='r',align='center', label=strategies[2], ecolor='k')


	plt.gca().set_ylim(bottom=0)
	plt.gca().set_ylim(top=1.4)

	leg = plt.legend()

	plt.setp(labels, rotation=90)
	plt.gca().set_ylim(bottom=0)

        plt.plot([-.5, len(allBars[0])], [1, 1], color='k', linestyle='-', linewidth=1)
        #hlines(1, 0, len(strategies))

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

	plt.ylabel('Final Score (Normalized to Ground Truth Score)', size=18)

	ax.legend(loc='upper left')

	plt.draw() # Draw the figure so you can find the positon of the legend. 

	# Get the bounding box of the original legend
	bb = leg.legendPatch.get_bbox().inverse_transformed(ax.transAxes)

	# Change to location of the legend. 
	yOffset = 1.5
	bb.set_points([[bb.x0, bb.y0 + yOffset], [bb.x1, bb.y1 + yOffset]])
	leg.set_bbox_to_anchor(bb)

	plt.draw()

	# # Change to location of the legend. 
	# newX0 = 0
	# newX1 = 10
	# bb.set_points([[newX0, bb.y0], [newX1, bb.y1]])
	# leg.set_bbox_to_anchor(bb)

        #plt.show()

	fig = plt.gcf()
	fig.set_size_inches(13, 8)
	fig.subplots_adjust(bottom=0.2)
	fig.savefig('lowestScore.pdf', edgecolor='none', format='pdf')
	plt.close()


makeLowestScore2 = True
if makeLowestScore2:
	print "\n********************************"
	print "BLOG-estimated likelihood score: means and errors across all runs"
	print "********************************"
	highestLowesScore = 0
	allBars = []
	allBarErros = []
	for strategy in orderedStrategyNames:
		if debug: print "********************************"
		if debug: print "Strategy: "+ strategy
		if debug: print "********************************"
		strategyBenchmarks = scoreData[strategy]
		bars = []
		barErrors = []
		for benchmarkname in sorted(groundTruthScores.keys()):
			benchmarkRuns = strategyBenchmarks[benchmarkname]
			lowestScores = []
			for run in benchmarkRuns:
				if debug: print run
				lowestScore = run / groundTruthScores[benchmarkname] # normalize
				lowestScores.append(lowestScore)
			bars.append(np.mean(lowestScores))
			barErrors.append(np.std(lowestScores))
			if debug: print benchmarkname, ":", lowestScores
		allBars.append(bars)
		allBarErros.append(barErrors)



	strategies = orderedStrategyNames
	print ' "-" '.join(['"Benchmark"'] + map(lambda x: "\""+x+"\"", strategies))
	for i in range(len(allBars[0])):
		print sorted(groundTruthScores.keys())[i]," ",
		for j in range(len(allBars)):
			print allBars[j][i], " ", allBarErros[j][i], " ",
		print

        print

        print ",".join([""] + strategies)
        avgs = map(lambda ls: np.mean(ls), allBars)
        print ",".join(map(lambda x: str(x), avgs))

	x = np.array(range(len(strategyBenchmarks)))
	my_xticks = sorted(strategyBenchmarks.keys()) # string labels
	locs, labels = plt.xticks(x, my_xticks)

	strategies = orderedStrategyNames

	ax = plt.subplot(111)
	ax.bar(x-0.2, allBars[0],width=0.2,color='b',align='center', label=strategies[0], ecolor='k', yerr=allBarErros[0])
	ax.bar(x, allBars[1],width=0.2,color='g',align='center', label=strategies[1], ecolor='k', yerr=allBarErros[1])
	ax.bar(x+0.2, allBars[2],width=0.2,color='r',align='center', label=strategies[2], ecolor='k', yerr=allBarErros[2])

	leg = plt.legend()
	plt.setp(labels, rotation=90)
	plt.gca().set_ylim(bottom=0)

        plt.plot([-.5, len(allBars[0])], [1, 1], color='k', linestyle='-', linewidth=1)
        #hlines(1, 0, len(strategies))

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

	plt.ylabel('Final Score (Normalized to Ground Truth Score)', size=18)

	ax.legend(loc='upper left')

	# # Change to location of the legend. 
	# newX0 = 0
	# newX1 = 10
	# bb.set_points([[newX0, bb.y0], [newX1, bb.y1]])
	# leg.set_bbox_to_anchor(bb)

        #plt.show()

	fig = plt.gcf()
	fig.set_size_inches(13, 8)
	fig.subplots_adjust(bottom=0.2)
	fig.savefig('lowestScore2.pdf', edgecolor='none', format='pdf')
	plt.close()

makeDataGuided = True
if makeDataGuided:
	strategy = "Network Deconvolution" # only going to do deconv for this one, since the data-blind verison was done with deconv
	strategyBenchmarks = dataSets[strategy]
	strategyBenchmarksDataBlind = dataSetsDatablind[strategy]


	numBenchmarks = len(groundTruthScores.keys())
	height = 5
	width = numBenchmarks/height
	if numBenchmarks % height > 0:
		width += 1
	# row and column sharing
	f, axarr = plt.subplots(height, width, sharey='row')

	benchmarkNamesSorted = sorted(groundTruthScores.keys())
	redData = []
	blueData = []
        dataGuidedTimesToGetClose = []
        dataBlindTimesToGetClose = []
        dataGuidedLowestScores = []
        dataBlindLowestScores = []
	for i in range(len(benchmarkNamesSorted)):
		benchmarkname = benchmarkNamesSorted[i]
		y = i/width
		x = i%width
		ax = axarr[y, x] # is this the element we want?

		dataGuidedTimeLists = []
		dataGuidedScoreLists = []

		# data guided
		benchmarkRuns = strategyBenchmarks[benchmarkname]
		for run in benchmarkRuns:
			dataGuidedTimeLists.append(map(lambda x: x[0], run))
			dataGuidedScoreLists.append(map(lambda x: x[1], run))
                        timeToGetClose = timeToReachScore(run, groundTruthScoreEstimates[benchmarkname]*threshold)
                        dataGuidedTimesToGetClose.append(timeToGetClose)
                        dataGuidedLowestScores.append(min(map(lambda x: x[1]/groundTruthScoreEstimates[benchmarkname], run)))

		dataBlindTimeLists = []
		dataBlindScoreLists = []

		# data blind		
		benchmarkRuns = strategyBenchmarksDataBlind[benchmarkname]
		for run in benchmarkRuns:
			dataBlindTimeLists.append(map(lambda x: x[0], run))
			dataBlindScoreLists.append(map(lambda x: x[1], run))
                        timeToGetClose = timeToReachScore(run, groundTruthScoreEstimates[benchmarkname]*threshold)
                        dataBlindTimesToGetClose.append(timeToGetClose)
                        dataBlindLowestScores.append(min(map(lambda x: x[1]/groundTruthScoreEstimates[benchmarkname], run)))

		maxTime = 0
		for run in dataBlindTimeLists:
			maxTimeForRun = run[-1]
			if maxTimeForRun > maxTime:
				maxTime = maxTimeForRun

		for i in range(len(dataGuidedTimeLists)):
			dataGuidedTimeLists[i].append(maxTime)
			dataGuidedScoreLists[i].append(dataGuidedScoreLists[i][-1]) # just continue the last reached score till the end since search stopped before maxtime
		for i in range(len(dataBlindTimeLists)):
			dataBlindTimeLists[i].append(maxTime)
			dataBlindScoreLists[i].append(dataBlindScoreLists[i][-1]) # just continue the last reached score till the end since search stopped before maxtime

		for i in range(len(dataBlindTimeLists)):
			ax.plot(dataBlindTimeLists[i], dataBlindScoreLists[i], "red")
			redData.append(dataBlindScoreLists[i])
		for i in range(len(dataGuidedTimeLists)):
			ax.plot(dataGuidedTimeLists[i], dataGuidedScoreLists[i], "blue")
			blueData.append(dataGuidedScoreLists[i])

		ax.set_title(benchmarkname)

		ax.set_ylim(bottom=0)
		ax.set_ylim(top=200000)
		ax.set_xlim(right=maxTime)
		if x == 0: ax.set_ylabel('Score', size=10)
		ax.set_xlabel('Time (in Seconds)', size=10)

	plt.subplots_adjust(hspace=.5, wspace=0.1)
	plt.draw()
	fig = plt.gcf()
	fig.set_size_inches(13, 19)
	#fig.subplots_adjust(bottom=0.2)

	# we want to put the legend in the position of the last axarr item
	pos1 = axarr[-1][-1].get_position()

	for i in range(len(benchmarkNamesSorted), width * height):
		y = i/width
		x = i%width
		fig.delaxes(axarr[y, x])

	red_patch = mpatches.Patch(color='red', label='Data-blind')
	blue_patch = mpatches.Patch(color='blue', label='Data-guided')
	fig.legend((red_patch,blue_patch), ('Data-blind', 'Data-guided'), bbox_to_anchor = pos1, fontsize = 20, loc='center')

	fig.savefig('dataGuidedVsDataBlind.pdf', edgecolor='none', format='pdf')
	plt.close()

        # print "time for data-guided to get close to ground truth: \t", dataGuidedTimesToGetClose
        dgLen = len(dataGuidedTimesToGetClose) 
        print dgLen
        neverReached = len(filter(lambda x: x == None, dataGuidedTimesToGetClose))
        print "percent reached: ", float(dgLen - neverReached)/dgLen
        print "among those that reach, average time to reach: ", 
        dgAvg = np.mean(filter(lambda x: x != None, dataGuidedTimesToGetClose))
        print dgAvg

        # print "time for data-blind to get close to ground truth: \t", dataBlindTimesToGetClose
        dbLen = len(dataBlindTimesToGetClose)
        print dbLen
        neverReached = len(filter(lambda x: x == None, dataBlindTimesToGetClose))
        print "percent reached: ", float(dbLen - neverReached)/dbLen
        print "among those that reach, average time to reach: ", 
        dbAvg = np.mean(filter(lambda x: x != None, dataBlindTimesToGetClose))
        print dbAvg

        ratio = dbAvg/dgAvg
        print "data-blind average over data-guided average", ratio

        dgLowestScoreAvg = np.mean(dataGuidedLowestScores)
        dbLowestScoreAvg = np.mean(dataBlindLowestScores)
        print "Average Normalized Best Score Per Run (data-guided):", dgLowestScoreAvg
        print "Average Normalized Best Score Per Run (data-blind):", dbLowestScoreAvg

makeDataGuidedNormalized = True
if makeDataGuidedNormalized:
	strategy = "Network Deconvolution" # only going to do deconv for this one, since the data-blind verison was done with deconv
	strategyBenchmarks = dataSets[strategy]
	strategyBenchmarksDataBlind = dataSetsDatablind[strategy]


	numBenchmarks = len(groundTruthScores.keys())
	height = 5
	width = numBenchmarks/height
	if numBenchmarks % height > 0:
		width += 1
	# row and column sharing
	f, axarr = plt.subplots(height, width, sharey='row')

	benchmarkNamesSorted = sorted(groundTruthScores.keys())
	redData = []
	blueData = []
	for i in range(len(benchmarkNamesSorted)):
		benchmarkname = benchmarkNamesSorted[i]
		y = i/width
		x = i%width
		ax = axarr[y, x] # is this the element we want?

		dataGuidedTimeLists = []
		dataGuidedScoreLists = []

		# data guided
		benchmarkRuns = strategyBenchmarks[benchmarkname]
		for run in benchmarkRuns:
			dataGuidedTimeLists.append(map(lambda x: x[0], run))
			dataGuidedScoreLists.append(map(lambda x: x[1]/groundTruthScoreEstimates[benchmarkname], run))

		dataBlindTimeLists = []
		dataBlindScoreLists = []

		# data blind		
		benchmarkRuns = strategyBenchmarksDataBlind[benchmarkname]
		for run in benchmarkRuns:
			dataBlindTimeLists.append(map(lambda x: x[0], run))
			dataBlindScoreLists.append(map(lambda x: x[1]/groundTruthScoreEstimates[benchmarkname], run))

		maxTime = 0
		for run in dataBlindTimeLists:
			maxTimeForRun = run[-1]
			if maxTimeForRun > maxTime:
				maxTime = maxTimeForRun

		for i in range(len(dataGuidedTimeLists)):
			dataGuidedTimeLists[i].append(maxTime)
			dataGuidedScoreLists[i].append(dataGuidedScoreLists[i][-1]) # just continue the last reached score till the end since search stopped before maxtime
		for i in range(len(dataBlindTimeLists)):
			dataBlindTimeLists[i].append(maxTime)
			dataBlindScoreLists[i].append(dataBlindScoreLists[i][-1]) # just continue the last reached score till the end since search stopped before maxtime

		for i in range(len(dataBlindTimeLists)):
			ax.plot(dataBlindTimeLists[i], dataBlindScoreLists[i], "red")
			redData.append(dataBlindScoreLists[i])
		for i in range(len(dataGuidedTimeLists)):
			ax.plot(dataGuidedTimeLists[i], dataGuidedScoreLists[i], "blue")
			blueData.append(dataGuidedScoreLists[i])

		ax.set_title(benchmarkname)

		ax.set_ylim(bottom=0)
		#ax.set_ylim(top=200000)
		ax.set_ylim(top=6)
		ax.set_xlim(right=maxTime)
		if x == 0: ax.set_ylabel('Normalized Score', size=10)
		ax.set_xlabel('Time (in Seconds)', size=10)

	plt.subplots_adjust(hspace=.5, wspace=0.1)
	plt.draw()
	fig = plt.gcf()
	fig.set_size_inches(13, 19)
	#fig.subplots_adjust(bottom=0.2)

	# we want to put the legend in the position of the last axarr item
	pos1 = axarr[-1][-1].get_position()

	for i in range(len(benchmarkNamesSorted), width * height):
		y = i/width
		x = i%width
		fig.delaxes(axarr[y, x])

	red_patch = mpatches.Patch(color='red', label='Data-blind')
	blue_patch = mpatches.Patch(color='blue', label='Data-guided')
	fig.legend((red_patch,blue_patch), ('Data-blind', 'Data-guided'), bbox_to_anchor = pos1, fontsize = 20, loc='center')

	fig.savefig('dataGuidedVsDataBlindNormalized.pdf', edgecolor='none', format='pdf')
	plt.close()
