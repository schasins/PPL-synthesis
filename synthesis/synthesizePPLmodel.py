import random
from astDB import *
from score import *
from BLOGScore import *
from subprocess import call
from copy import deepcopy
from scipy.stats.stats import pearsonr
from itertools import combinations
import sys
from simanneal import Annealer
from ND import *
from scipy.stats import spearmanr
from cStringIO import StringIO
import pickle
import time

# **********************************************************************
# Helpers
# **********************************************************************

class Capturing(list):
	def __enter__(self):
		self._stdout = sys.stdout
		sys.stdout = self._stringio = StringIO()
		return self
	def __exit__(self, *args):
		self.extend(self._stringio.getvalue().splitlines())
		sys.stdout = self._stdout

# **********************************************************************
# Data structures for representing structure hints
# **********************************************************************

class Graph:
	def __init__(self):
		self.nodes = []
		self.names = {}

	def addNode(self, node):
		self.nodes.append(node)
		self.names[node.name] = node

	def getNode(self, name, distribInfo):
		if name in self.names:
			return self.names[name]
		else:
			newNode = Node(name, distribInfo)
			self.addNode(newNode)
			return newNode

	def getNodesInDependencyOrder(self):
		outputLs = []
		for node in self.nodes:
			Graph.addNodeToDepOrderLs(outputLs, node)
		return outputLs

	def isDescendedFrom(self, name1, info1, name2, info2):
		n1 = self.getNode(name1, info1)
		n2 = self.getNode(name2, info2)
		frontier = []
		for child in n1.children:
			frontier.append(child)
		while len(frontier) > 0:
			node = frontier.pop(0)
			if node == n2:
				return True
			for child in node.children:
				frontier.append(child)

	@staticmethod
	def addNodeToDepOrderLs(ls, node):
		if node in ls:
			return
		parents = node.parents
		for parent in parents:
			Graph.addNodeToDepOrderLs(ls, parent)
		ls.append(node)

class Node:
	def __init__(self, name, distribInfo):
		self.name = name
		self.parents = []
		self.children = []
		self.distribInfo = distribInfo


# **********************************************************************
# Helper functions
# **********************************************************************

def makeProgram(scriptStrings, holes, numHoles):
	outputString = scriptStrings[0]
	for i in range(0, numHoles):
		string = scriptStrings[i+1]
		holeFiller = holes[i]
		outputString += str(holeFiller)
		outputString += string
	return outputString

# **********************************************************************
# Evaluate generated programs' proximity to spec
# **********************************************************************

def summarizeDataset(fileName):
	f = open(fileName, "r")
	lines = f.readlines()

	line = lines[0].strip()
	lineItems = line.split(",")
	sums = {}
	names = {}
	numItems = 0
	for i in range(len(lineItems)):
		lineItem = lineItems[i]
		if lineItem != "":
			sums[i] = 0
			names[i] = lineItem
			numItems = i
	numItems += 1

	for line in lines[1:]:
		entries = line.strip().split(",")
		for i in range(numItems):
			entry = entries[i]
			if entry == "true":
				entry = 1
			else:
				entry = 0
			sums[i] = sums[i] + entry

	numLines = len(lines) - 1
	means = {}
	for i in range(numItems):
		means[names[i]] = float(sums[i])/numLines
	return means

def distance(summary1, summary2):
	res = 0
	for key in summary1:
		v1 = summary1[key]
		v2 = summary2[key]	
		res += abs(v1 - v2)
	return res

# **********************************************************************
# Simulated annealing
# **********************************************************************

class PPLSynthesisProblem(Annealer):

	def move(self):
		self.state.mutate()

	def energy(self):
		global startTime, cleanTimingData
		currTime = time.clock()
		score = -1*estimateScore(self.state.root, self.dataset)
		cleanTimingData.append([currTime, score])
		return score

	@staticmethod
	def makeInitialState(prog):
		return prog

	def setNeeded(self, dataset):
		self.dataset = dataset

# **********************************************************************
# Generate structures based on input dataset correlation
# **********************************************************************

def correlationHelper(dataset, i, j):
	iCols = dataset.columnNumericColumns[i]
	jCols = dataset.columnNumericColumns[j]
	correlations = []
	correlations_2 = []
	for iCol in iCols:
		for jCol in jCols:
			res = spearmanr(iCol, jCol)
			#res2 = pearsonr(iCol, jCol)
			correlations.append(res)
			#correlations_2.append(res2[0])
	correlation1 = max(correlations, key=lambda item:item[1])
	correlation2 = min(correlations, key=lambda item:item[1])
	correlation = correlation1
	if abs(correlation2[0]) > abs(correlation1[0]):
		correlation = correlation2

	#correlation1_2 = max(correlations_2)
	#correlation2_2 = min(correlations_2)
	#correlation_2 = correlation1_2
	#if abs(correlation2_2) > abs(correlation1_2):
	#	correlation_2 = correlation2_2	
	return correlation

def generateStructureFromDatasetNetworkDeconvolution(dataset, connectionThreshold):
	correlationsMatrix = [ [ 0 for i in range(dataset.numColumns) ] for j in range(dataset.numColumns) ]
	for i in range(dataset.numColumns):
		for j in range(i + 1, dataset.numColumns):
			correlation = correlationHelper(dataset, i, j)[0]

			#correlation = pearsonr(dataset.columns[i], dataset.columns[j])
			correlationsMatrix[i][j] = correlation
			correlationsMatrix[j][i] = correlation

			for i in range(len(correlationsMatrix)):
				print correlationsMatrix[i]

	a = np.array(correlationsMatrix)
	x = ND(a)

	g = Graph()
	for i in range(dataset.numColumns):
		name1 = dataset.indexesToNames[i]
		a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
		for j in range(i + 1, dataset.numColumns):
			if x[i][j] > connectionThreshold:
				name2 = dataset.indexesToNames[j]
				a2 = g.getNode(name2, dataset.columnDistributionInformation[j])
				a1.children.insert(0, a2)
				a2.parents.insert(0, a1)
	return g


def generateReducibleStructuresFromDataset(dataset):
	g = Graph()
	for i in range(dataset.numColumns):
		name1 = dataset.indexesToNames[i]
		a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
		for j in range(i+1, dataset.numColumns):
			name2 = dataset.indexesToNames[j]
			a2 = g.getNode(name2, dataset.columnDistributionInformation[j])
			a1.children.insert(0, a2)
			a2.parents.insert(0, a1)
			#print name1, "->", name2, correlationHelper(dataset, i, j)

	return g

statisticalSignificanceThreshold = 0.05
correlationThreshold = 0.01

def generatePotentialStructuresFromDataset(dataset):
	global statisticalSignificanceThreshold, correlationThreshold
	columns = range(dataset.numColumns)
	combos = combinations(columns, 2)
	correlations = []
	for combo in combos:
		correlationPair = correlationHelper(dataset, combo[0], combo[1])
		#correlationPair = pearsonr(dataset.columns[combo[0]], dataset.columns[combo[1]])
		correlations.append((combo, correlationPair))
	sortedCorrelations = sorted(correlations, key=lambda x: abs(x[1][0]), reverse=True)
	#print sortedCorrelations

	g = Graph()
	# make sure we add all nodes
	for i in range(dataset.numColumns):
		name1 = dataset.indexesToNames[i]
		a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
	# now add relationships
	for correlation in sortedCorrelations:
		i = correlation[0][0]
		j = correlation[0][1]
		name1 = dataset.indexesToNames[i]
		name2 = dataset.indexesToNames[j]
		statisticalSignificance = correlation[1][1]
		correlationAmount = abs(correlation[1][0])
		#print name1, name2, correlationAmount, statisticalSignificance
		if statisticalSignificance > statisticalSignificanceThreshold:
			#print "non sig:", statisticalSignificance
			continue
		if correlationAmount < correlationThreshold:
			#print "not cor:", correlationAmount
			break
		if not g.isDescendedFrom(name1, dataset.columnDistributionInformation[i], name2, dataset.columnDistributionInformation[j]):
			# we don't yet have an explanation from the connection between these two.  add one.
			a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
			a2 = g.getNode(name2, dataset.columnDistributionInformation[j])
			# for now we'll assume the causation goes from left to right in input dataset
			# TODO: eventually should create multiple different prog structures from single dataset
			a1.children.insert(0, a2)
			a2.parents.insert(0, a1) # for how we process, it's nicer to have parents in reverse order of correlation
			#print name1, "->", name2, correlation[1]
		# else:
		# 	print "already descended"
	return g

# **********************************************************************
# Consume the structure hints, generate a program
# **********************************************************************


def deepcopyNode(node):
	newNode = deepcopy(node)
	return newNode

startTime = 0
cleanTimingData = []


dataset = None

def main():
	global startTime, cleanTimingData, dataset

	debug = False

	inputFile = sys.argv[1]
	SAiterations = int(sys.argv[2])
	outputDirectory = sys.argv[3]
	outputFilename = sys.argv[4]
	structureGenerationStrategy = sys.argv[5]
	mode = sys.argv[6]
	if len(sys.argv) > 7:
		debug = True if sys.argv[7] == "t" else False
		print "Debugging messages on."

	startTime = time.clock()

	print "about to make dataset"
	dataset = Dataset(inputFile)

	g = None
	if structureGenerationStrategy == "n":
		# naive generation strategy
		g = generateReducibleStructuresFromDataset(dataset)
	elif structureGenerationStrategy == "d":
		# deconvolution strategy
		g = generateStructureFromDatasetNetworkDeconvolution(dataset, .1)
	elif structureGenerationStrategy == "c":
		# simple correlation strategy
		g = generatePotentialStructuresFromDataset(dataset)
	else:
		raise Exception("We don't know this structure generation strategy.")
	

	nodesInDependencyOrder = g.getNodesInDependencyOrder()

	AST = ASTNode()
	for node in nodesInDependencyOrder:

		parents = node.parents
                parentNames = map(lambda x: x.name, parents)
                dataset.addIndex(parentNames)

		if isinstance(node.distribInfo, BooleanDistribution):
			internal = BooleanDistribNode(node.name)
		elif isinstance(node.distribInfo, CategoricalDistribution):
			typeDecl = TypeDeclNode(node.distribInfo.typeName, node.distribInfo.values)
			AST.addChild(typeDecl)
			internal = CategoricalDistribNode(node.name, node.distribInfo.values)
		elif isinstance(node.distribInfo, IntegerDistribution):
			# don't currently have integerdistribnode :(  use reals for now
			# internal = IntegerDistribNode(node.name)
			internal = RealDistribNode(node.name)
		elif isinstance(node.distribInfo, RealDistribution):
			internal = RealDistribNode(node.name)

		for parent in parents:
			conditionNodes = []
			bodyNodes = []
			if isinstance(parent.distribInfo, BooleanDistribution):
				conditionNodes.append(VariableUseNode(parent.name, parent.distribInfo.typeName))
				for i in range(2):
					bodyNodes.append(deepcopyNode(internal))
			elif isinstance(parent.distribInfo, CategoricalDistribution):
				numValues = len(parent.distribInfo.values)
				if numValues == 1:
					# doesn't make sense to depend on this
					continue
				for i in range(numValues):
					conditionNodes.append(ComparisonNode(VariableUseNode(parent.name, parent.distribInfo.typeName), "=", StringValue(parent.distribInfo.values[i])))
				for i in range(numValues):
					bodyNodes.append(deepcopyNode(internal))
			elif isinstance(parent.distribInfo, IntegerDistribution) or isinstance(parent.distribInfo, RealDistribution):
				conditionNodes.append(ComparisonNode(VariableUseNode(parent.name, parent.distribInfo.typeName)))
				for i in range(2):
					bodyNodes.append(deepcopyNode(internal))
			internal = IfNode(conditionNodes, bodyNodes)

		variableNode = VariableDeclNode(node.name, node.distribInfo.typeName, internal)
		AST.addChild(variableNode)

	prog = Program(dataset)
	prog.setRoot(AST)

	if debug: print "filling holes for concrete path conditions."

	AST.fillHolesForConcretePathConditions(dataset)

	if mode == "reduction" or mode == "reductionProg":

		def writeFile(prog, label):
			outputString = prog.programString()
			inputName = inputFile.split("/")[-1].split(".")[0]
			outputFilename = inputName+"_"+label+"_"+str(prog.distribNodes)+".blog"
			output = open(outputDirectory+"/reducedPrograms/"+outputFilename, "w")
			output.write(outputString)

		def printStats(prog, label, lastNumDistribNodes, lastScore):
			numDistribNodes = prog.distribNodes
			score = lastScore
			if numDistribNodes != lastNumDistribNodes: # let's avoid extra score calculations for equivalent programs, since they're expensive
				score = blogLikelihoodScore(prog, dataset)
			print  label,",", score,",",numDistribNodes,",",prog.varUseNodes,",",prog.comparisonNodes
			return numDistribNodes, lastScore

		lastScore = None
		lastNumDistribNodes = -1

		# now let's actually do some stuff!
		if mode == "reduction":
			lastNumDistribNodes, lastScore = printStats(prog, str(i), lastNumDistribNodes, lastScore)
		else :
			writeFile(prog, str(i))
								
		i = -.5
		while i < 20:
			i += .5
			progCopy = deepcopy(prog)
			progCopy.root.setProgram(progCopy)
			progCopy.thresholdMaker = i
			progCopy.root.reduce(dataset)
			progCopy.distribNodes = 0 # let's count those nodes now
			progCopy.varUseNodes = 0
			progCopy.comparisonNodes = 0
			progCopy.root.setProgram(progCopy)
			if mode == "reduction":
				lastNumDistribNodes, lastScore = printStats(progCopy, str(i), lastNumDistribNodes, lastScore)
			else :
				writeFile(progCopy, str(i))
								
	elif mode == "annealing":
		# below is simulated annealing

		AST.fillHolesRandomly()
		if debug: print prog.programString()
		# if debug: print AST.strings()

		cleanTimingData = []

		initState = PPLSynthesisProblem.makeInitialState(prog)
		saObj = PPLSynthesisProblem(initState)
		saObj.setNeeded(dataset)
		saObj.steps = SAiterations # 100000 #how many iterations will we do?
		saObj.updates = SAiterations # 100000 # how many times will we print current status
		saObj.Tmax = 50000.0 #(len(scriptStrings)-1)*.1 # how big an increase in distance are we willing to accept at start?

		saObj.Tmin = 1 # how big an increase in distance are we willing to accept at the end?

		endTime = time.clock()
		# distanceFromDataset = blogLikelihoodScore(prog, dataset)
		cleanTimingData.append([endTime]) #, distanceFromDataset])
		annealingOutput = []
		if debug: print "About to anneal."
		if len(prog.randomizeableNodes) > 0:
                        if not debug:
                                with Capturing() as annealingOutput:
                                        progOutput, distanceFromDataset = saObj.anneal()
                        else:
                                progOutput, distanceFromDataset = saObj.anneal()
		else:
			progOutput = prog


		#print "\n".join(annealingOutput)
		if debug: print progOutput.programString()
                if debug: print cleanTimingData[-1]

		#AST.reduce(dataset) # todo: control how much we reduce, make sure this checks path conditions before reducing

		outputString = progOutput.programString()+"\n\n//"+str(distanceFromDataset)
		output = open(outputDirectory+"/synthesizedBLOGPrograms/"+outputFilename+"_"+str(SAiterations)+"_"+str(structureGenerationStrategy)+"_.blog", "w")
		output.write(outputString)
		output2 = open(outputDirectory+"/pickles/"+outputFilename+"_"+str(SAiterations)+"_"+str(structureGenerationStrategy)+"_.pickle", "w")
		pickle.dump(prog, output2)
		output3 = open(outputDirectory+"/timingData/"+outputFilename+"_"+str(SAiterations)+"_"+str(structureGenerationStrategy)+"_.timing", "w")
		output3.write("\n".join(annealingOutput))
		output4 = open(outputDirectory+"/cleanTimingData/"+outputFilename+"_"+str(SAiterations)+"_"+str(structureGenerationStrategy)+"_.timing", "w")
		output4.write("\n".join(map(lambda row: ",".join(map(str, row)), cleanTimingData)))

	else:
		raise Exception("Don't recognize the requested mode: "+mode)

main()



