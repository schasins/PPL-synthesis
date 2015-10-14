import random
from subprocess import call
from simanneal import Annealer
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
# Data structures for creating PPL ASTs
# **********************************************************************

class Dataset:
	def __init__(self, filename):
		f = open(filename, "r")
		lines = f.readlines()

		line = lines[0].strip()
		lineItems = line.split(",")
		names = {}
		indexes = {}
		numItems = 0
		for i in range(len(lineItems)):
			lineItem = lineItems[i]
			if lineItem != "":
				names[i] = lineItem
				indexes[lineItem] = i
				numItems = i

		numItems += 1

		self.numColumns = numItems

		self.indexesToNames = names
		self.namesToIndexes = indexes

		columns = []
		for i in range(numItems):
			columns.append([])
		rows = []
		for line in lines[1:]:
			cells = []
			entries = line.strip().split(",")
			for i in range(numItems):
				entry = entries[i]
				if entry == "true":
					entry = 1
				else:
					entry = 0
				cells.append(entry)
				columns[i].append(entry)
			rows.append(cells)

		self.columns = columns
		self.rows = rows

	def makePathConditionFilter(self, pathCondition):
		pairs = [] # (index, targetVal) pairs
		for pathConditionComponent in pathCondition:
			pairs.append((self.namesToIndexes[pathConditionComponent.varName], pathConditionComponent.value))
		return lambda row : reduce(lambda x, pair : x and row[pair[0]] == pair[1], pairs, True)

	def makeCurrVariableGetter(self, currVariable):
		index = self.namesToIndexes[currVariable.name]
		return lambda row: row[index]

class PathConditionComponent:
	def __init__(self, varName, relationship, value):
		self.varName = varName
		self.relationship = relationship
		self.value = value

# **********************************************************************
# Data structures for representing PPL ASTs
# **********************************************************************

class ASTNode:
	def __init__(self):
		self.children = []

	def strings(self, tabs=0):
		outputStrings = []
		for child in self.children:
			childStrings = child.strings()
			outputStrings = combineStrings([outputStrings, childStrings])
		return outputStrings

	def fillHolesForConcretePathConditions(self, dataset, pathCondition =[], currVariable = None):
		for node in self.children:
			node.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable)

class VariableDeclNode(ASTNode):
	def __init__(self, name, varType, RHS):
		ASTNode.__init__(self)
		self.name = name
		self.varType = varType
		self.RHS = RHS

	def strings(self, tabs=0):
		s = ["\nrandom "+self.varType+" "+self.name+" ~ "]
		RHSStrings = self.RHS.strings()
		return combineStrings([s, RHSStrings, [";\n"]])

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		self.RHS.fillHolesForConcretePathConditions(dataset, pathCondition, self) # the current node is now the variable being defined

class BooleanDistribNode(ASTNode):
	def __init__(self, percentTrue=None):
		ASTNode.__init__(self)
		self.percentTrue = percentTrue

	def strings(self, tabs=0):
		components = ["BooleanDistrib(", ")"]
		if self.percentTrue:
			return [components[0]+str(self.percentTrue)+components[1]]
		else:
			return components

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		pathConditionFilter = dataset.makePathConditionFilter(pathCondition)
		currVariableGetter = dataset.makeCurrVariableGetter(currVariable)
		matchingRowsCounter = 0
		matchingRowsSum = 0
		for row in dataset.rows:
			if pathConditionFilter(row):
				matchingRowsCounter += 1
				val = currVariableGetter(row)
				matchingRowsSum += val
		self.percentTrue = 0.5 if matchingRowsCounter == 0 else float(matchingRowsSum)/matchingRowsCounter

class IfNode(ASTNode):
	def __init__(self, conditionNode, thenNode, elseNode):
		self.conditionNode = conditionNode
		self.thenNode = thenNode
		self.elseNode = elseNode

	def strings(self, tabs=0):
		tabs = tabs + 1
		return combineStrings([["\n"+"\t"*tabs+"if "], self.conditionNode.strings(tabs), ["\n"+"\t"*tabs+"then "], self.thenNode.strings(tabs), ["\n"+"\t"*tabs+"else "], self.elseNode.strings(tabs)])

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		pathConditions = self.conditionNode.pathConditions()
		truePathCondition = pathCondition + [pathConditions[0]]
		self.thenNode.fillHolesForConcretePathConditions(dataset, truePathCondition, currVariable)
		falsePathCondition = pathCondition + [pathConditions[1]]
		self.elseNode.fillHolesForConcretePathConditions(dataset, falsePathCondition, currVariable)

class VariableUseNode(ASTNode):
	def __init__(self, name):
		self.name = name

	def strings(self, tabs=0):
		return [self.name]

	def pathConditions(self):
		return [PathConditionComponent(self.name, "eq", True), PathConditionComponent(self.name, "eq", False)]

# **********************************************************************
# Helper functions
# **********************************************************************

def combineStrings(ls):
	if len(ls) < 2:
		return ls[0]
	else:
		return combineStrings([combineStringsTwo(ls[0], ls[1])] + ls[2:])

def combineStringsTwo(n1Strings, n2Strings):
	if len(n2Strings) < 1:
		return n1Strings
	if len(n1Strings) < 1:
		return n2Strings
	resStrings = n1Strings[:-1]	+ [n1Strings[-1]+n2Strings[0]] + n2Strings[1:]
	return resStrings

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
		# changes the value in one hole of the program
		# TODO: is this actually the way we want to do a move?
		numHoles = self.state[0]
		holeFillers = []
		for i in range(numHoles):
			holeFillers.append(random.uniform(.000001, .999999))
		self.state = [numHoles]+holeFillers

	def energy(self):
		# calculates the distance from the target distributions
		outputString = makeProgram(self.programStrings, self.state[1:], self.state[0])

		f = open("output.blog", "w")
		f.write(outputString)
		f.close()

		f = open("output.output", "w")
		call(["blog", "output.blog", "--generate", "-n", "1000"], stdout=f)
		call(["python", "blogOutputToCSV.py", "output.output", "output.csv"])

		summaryCandidate = summarizeDataset("output.csv")
		return distance(summaryCandidate, self.targetSummary) # want to minimize this

	@staticmethod
	def makeInitialState(programStrings):
		numHoles = len(programStrings) - 1
		state = [numHoles]
		for i in range(numHoles):
			state.append(random.uniform(.000001, .999999))
		return state

	def setNeeded(self, programStrings, targetSummary):
		self.programStrings = programStrings
		self.targetSummary = targetSummary

# **********************************************************************
# Generate structures based on input dataset correlation
# **********************************************************************

def generatePotentialStructuresFromDataset(dataset):
	columns = range(dataset.numColumns)
	combos = combinations(columns, 2)
	correlations = []
	for combo in combos:
		print dataset.indexesToNames[combo[0]], dataset.indexesToNames[combo[1]]
		correlationPair = pearsonr(dataset.columns[combo[0]], dataset.columns[combo[1]])
		correlations.append((combo, correlationPair))
	sortedCorrelations = sorted(correlations, key=lambda x: x[1][1])

	g = Graph()
	for correlation in sortedCorrelations:
		name1 = dataset.indexesToNames[correlation[0][0]]
		name2 = dataset.indexesToNames[correlation[0][1]]
		statisticalSignificance = correlation[1][1]
		if statisticalSignificance > .05:
			break
		if not g.isDescendedFrom(name1, name2):
			# we don't yet have an explanation from the connection between these two.  add one.
			a1 = g.getNode(name1)
			a2 = g.getNode(name2)
			# for now we'll assume the causation goes from left to right in input dataset
			# TODO: eventually should create multiple different prog structures from single dataset
			a1.children.append(a2)
			a2.parents.append(a1)
			print name1, "->", name2, correlation[1]
	return [g]

# **********************************************************************
# Consume the structure hints, generate a program
# **********************************************************************

def main():
	dataset = Dataset("burglary.csv")
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

	scriptStrings = AST.strings()
	output = open("outputDeterministic.blog", "w")
	output.write(scriptStrings[0])
	print scriptStrings[0]

main()



