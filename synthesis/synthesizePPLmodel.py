import random
from ast import *
from score import *
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

	def isDescendedFrom(self, name1, name2):
		n1 = self.getNode(name1)
		n2 = self.getNode(name2)
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
		return -1*estimateScore(self.state.root, self.dataset)

	@staticmethod
	def makeInitialState(prog):
		return prog

	def setNeeded(self, dataset):
		self.dataset = dataset

# **********************************************************************
# Generate structures based on input dataset correlation
# **********************************************************************

def generateStructureFromDatasetNetworkDeconvolution(dataset, connectionThreshold):
	correlationsMatrix = [ [ 0 for i in range(dataset.numColumns) ] for j in range(dataset.numColumns) ]
	for i in range(dataset.numColumns):
		for j in range(i + 1, dataset.numColumns):
			iCols = dataset.columnNumericColumns[i]
			jCols = dataset.columnNumericColumns[j]
			correlations = []
			for iCol in iCols:
				for jCol in jCols:
					print len(iCol), len(jCol)
					res = spearmanr(iCol, jCol)
					correlations.append(res[0])
			print correlations
			correlation1 = max(correlations)
			correlation2 = min(correlations)
			correlation = correlation1
			if abs(correlation2) > abs(correlation1):
				correlation = correlation2

			#correlation = pearsonr(dataset.columns[i], dataset.columns[j])
			correlationsMatrix[i][j] = correlation
			correlationsMatrix[j][i] = correlation
	for row in correlationsMatrix:
		print row

	x = np.array(correlationsMatrix)
	print ND(x)
	exit()


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

	return g

statisticalSignificanceThreshold = 0.05
correlationThreshold = .05

def generatePotentialStructuresFromDataset(dataset):
	global statisticalSignificanceThreshold, correlationThreshold
	columns = range(dataset.numColumns)
	combos = combinations(columns, 2)
	correlations = []
	for combo in combos:
		correlationPair = pearsonr(dataset.columns[combo[0]], dataset.columns[combo[1]])
		correlations.append((combo, correlationPair))
	sortedCorrelations = sorted(correlations, key=lambda x: abs(x[1][0]), reverse=True)

	g = Graph()
	for correlation in sortedCorrelations:
		name1 = dataset.indexesToNames[correlation[0][0]]
		name2 = dataset.indexesToNames[correlation[0][1]]
		statisticalSignificance = correlation[1][1]
		correlationAmount = abs(correlation[1][0])
		print name1, name2
		if statisticalSignificance > statisticalSignificanceThreshold:
			print "non sig:", statisticalSignificance
			continue
		if correlationAmount < correlationThreshold:
			print "not cor:", correlationAmount
			break
		if not g.isDescendedFrom(name1, name2):
			# we don't yet have an explanation from the connection between these two.  add one.
			a1 = g.getNode(name1)
			a2 = g.getNode(name2)
			# for now we'll assume the causation goes from left to right in input dataset
			# TODO: eventually should create multiple different prog structures from single dataset
			a1.children.insert(0, a2)
			a2.parents.insert(0, a1) # for how we process, it's nicer to have parents in reverse order of correlation
			print name1, "->", name2, correlation[1]
		else:
			print "already descended"
	return [g]

# **********************************************************************
# Consume the structure hints, generate a program
# **********************************************************************

def deepcopyNode(node):
	newNode = deepcopy(node)
	return newNode

def main():
	inputFile = sys.argv[1]
	SAiterations = int(sys.argv[2])
	connectionThreshold = float(sys.argv[3])
	outputDirectory = sys.argv[4]
	outputFilename = sys.argv[5]

	dataset = Dataset(inputFile)
	#g = generateReducibleStructuresFromDataset(dataset)[0]
	#g = generateStructureFromDatasetNetworkDeconvolution(dataset, connectionThreshold)
	g = generateReducibleStructuresFromDataset(dataset)

	nodesInDependencyOrder = g.getNodesInDependencyOrder()

	AST = ASTNode()
	for node in nodesInDependencyOrder:

		parents = node.parents

		if isinstance(node.distribInfo, BooleanDistribution):
			internal = BooleanDistribNode(node.name)
		elif isinstance(node.distribInfo, CategoricalDistribution):
			typeDecl = TypeDeclNode(node.distribInfo.typeName, node.distribInfo.values)
			AST.addChild(typeDecl)
			internal = CategoricalDistribNode(node.name, node.distribInfo.values)
		elif isinstance(node.distribInfo, IntegerDistribution):
			internal = IntegerDistribNode(node.name)
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
					conditionNodes.append(ComparisonNode(VariableUseNode(parent.name, parent.distribInfo.typeName), "==", parent.distribInfo.values[i]))
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



	AST.fillHolesForConcretePathConditions(dataset)

	# TODO: no longer want to reduce ahead of time, but may want to reduce the final output?  after we use BLOG inference?
	
	print prog.programString()

	AST.reduce(dataset)

	#print prog.programString()

	AST.fillHolesRandomly()

	print prog.programString()
	print prog.randomizeableNodes

	# print "actual score: ", estimateScore(prog.root, dataset)
	# for i in range(10):
	# 	prog.mutate()
	# 	print estimateScore(prog.root, dataset)
	# print prog.root.randomizeableVariables
	# prog.mutate()
	# print prog.programString()
	# prog.mutate()
	# print prog.programString()
	# prog.mutate()
	# print prog.programString()

	# TODO: do we want to figure out the proper params for the distributions we've added?
	#AST.fillHolesForConcretePathConditions(dataset)

	# below is simulated annealing

	initState = PPLSynthesisProblem.makeInitialState(prog)
	saObj = PPLSynthesisProblem(initState)
	saObj.setNeeded(dataset)
	saObj.steps = SAiterations # 100000 #how many iterations will we do?
	saObj.updates = SAiterations # 100000 # how many times will we print current status
	saObj.Tmax = 50000.0 #(len(scriptStrings)-1)*.1 # how big an increase in distance are we willing to accept at start?

	saObj.Tmin = 1 # how big an increase in distance are we willing to accept at the end?

	#print AST.strings()
	distanceFromDataset = -1*estimateScore(prog.root, dataset)
	#print len(prog.randomizeableNodes)
	annealingOutput = []
	if len(prog.randomizeableNodes) > 0:
		with Capturing() as annealingOutput:
			ast, distanceFromDataset = saObj.anneal()

	print prog.programString()

	AST.reduce(dataset) # todo: control how much we reduce, make sure this checks path conditions before reducing

	scriptStrings = AST.strings()
	outputString = scriptStrings[0]+"\n\n//"+str(distanceFromDataset)
	output = open(outputDirectory+"/synthesizedBLOGPrograms/"+outputFilename+"_"+str(SAiterations)+"_"+str(correlationThreshold)+"_.blog", "w")
	output.write(outputString)
	output2 = open(outputDirectory+"/pickles/"+outputFilename+"_"+str(SAiterations)+"_"+str(correlationThreshold)+"_.pickle", "w")
	pickle.dump(prog, output2)
	output3 = open(outputDirectory+"/timingData/"+outputFilename+"_"+str(SAiterations)+"_"+str(correlationThreshold)+"_.timing", "w")
	output3.write("\n".join(annealingOutput))


	#print scriptStrings

main()



