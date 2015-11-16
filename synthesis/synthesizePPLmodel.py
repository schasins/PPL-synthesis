import random
from ast import *
from score import *
from subprocess import call
from copy import deepcopy
from scipy.stats.stats import pearsonr
from itertools import combinations

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

	def getNode(self, name):
		if name in self.names:
			return self.names[name]
		else:
			newNode = Node(name)
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
	def __init__(self, name):
		self.name = name
		self.parents = []
		self.children = []


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

# class PPLSynthesisProblem(Annealer):

# 	def move(self):
# 		# changes the value in one hole of the program
# 		# TODO: is this actually the way we want to do a move?
# 		numHoles = self.state[0]
# 		holeFillers = []
# 		for i in range(numHoles):
# 			holeFillers.append(random.uniform(.000001, .999999))
# 		self.state = [numHoles]+holeFillers

# 	def energy(self):
# 		# calculates the distance from the target distributions
# 		outputString = makeProgram(self.programStrings, self.state[1:], self.state[0])

# 		f = open("output.blog", "w")
# 		f.write(outputString)
# 		f.close()

# 		f = open("output.output", "w")
# 		call(["blog", "output.blog", "--generate", "-n", "1000"], stdout=f)
# 		call(["python", "blogOutputToCSV.py", "output.output", "output.csv"])

# 		summaryCandidate = summarizeDataset("output.csv")
# 		return distance(summaryCandidate, self.targetSummary) # want to minimize this

# 	@staticmethod
# 	def makeInitialState(programStrings):
# 		numHoles = len(programStrings) - 1
# 		state = [numHoles]
# 		for i in range(numHoles):
# 			state.append(random.uniform(.000001, .999999))
# 		return state

# 	def setNeeded(self, programStrings, targetSummary):
# 		self.programStrings = programStrings
# 		self.targetSummary = targetSummary

# **********************************************************************
# Generate structures based on input dataset correlation
# **********************************************************************

def generatePotentialStructuresFromDataset(dataset):
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
		if statisticalSignificance > .05:
			print "non sig:", statisticalSignificance
			continue
		if correlationAmount < .1:
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

def main():
	dataset = Dataset("../data-generation/csi.csv")
	g = generatePotentialStructuresFromDataset(dataset)[0]

	nodesInDependencyOrder = g.getNodesInDependencyOrder()

	AST = ASTNode()
	for node in nodesInDependencyOrder:

		parents = node.parents
		internal = BooleanDistribNode()
		for parent in parents:
			conditionNode = VariableUseNode(parent.name)
			thenNode = deepcopy(internal)
			elseNode = deepcopy(internal)
			internal = IfNode(conditionNode, thenNode, elseNode)

		variableNode = VariableDeclNode(node.name, "Boolean", internal)
		AST.children.append(variableNode)

	AST.fillHolesForConcretePathConditions(dataset)
        # testEstimateScore(AST,dataset)

	scriptStrings = AST.strings()
	output = open("../synthesized/csi-outputDeterministic.blog", "w")
	output.write(scriptStrings[0])
	print scriptStrings[0]

main()



