import random
from subprocess import call
from simanneal import Annealer

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

class IfNode(ASTNode):
	def __init__(self, conditionNode, thenNode, elseNode):
		self.conditionNode = conditionNode
		self.thenNode = thenNode
		self.elseNode = elseNode

	def strings(self, tabs=0):
		tabs = tabs + 1
		return combineStrings([["\n"+"\t"*tabs+"if "], self.conditionNode.strings(tabs), ["\n"+"\t"*tabs+"then "], self.thenNode.strings(tabs), ["\n"+"\t"*tabs+"else "], self.elseNode.strings(tabs)])

class VariableUseNode(ASTNode):
	def __init__(self, name):
		self.name = name

	def strings(self, tabs=0):
		return [self.name]

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
# Consume the structure hints, generate a program
# **********************************************************************

def main():

	targetSummary = summarizeDataset("burglary.csv")

	g = Graph()
	f = open("burglary.hints", "r")
	lines = f.readlines()
	for line in lines:
		actors = line.strip().split(" -> ")
		a1 = g.getNode(actors[0])
		a2 = g.getNode(actors[1])
		a1.children.append(a2)
		a2.parents.append(a1)

	nodesInDependencyOrder = g.getNodesInDependencyOrder()

	AST = ASTNode()
	for node in nodesInDependencyOrder:
		print node.name
		print len(node.parents)

		parents = node.parents
		if len(parents) > 0:
			internal = BooleanDistribNode()
			for parent in parents:
				conditionNode = VariableUseNode(parent.name)
				thenNode = internal
				elseNode = internal
				internal = IfNode(conditionNode, thenNode, elseNode)
		else:
			# we don't need a hole, can just calculate the percent of time to give 1
			percent = targetSummary[node.name]
			print percent
			internal = BooleanDistribNode(percent)

		variableNode = VariableDeclNode(node.name, "Boolean", internal)
		AST.children.append(variableNode)

	scriptStrings = AST.strings()

	initState = PPLSynthesisProblem.makeInitialState(scriptStrings)
	saObj = PPLSynthesisProblem(initState)
	saObj.setNeeded(scriptStrings, targetSummary)
	saObj.steps = 1000 #how many iterations will we do?
	saObj.updates = 1000 # how many times will we print current status
	saObj.Tmax = (len(scriptStrings)-1)*.1 # how big an increase in distance are we willing to accept at start?
	print "---"
	print saObj.Tmax
	print (len(scriptStrings)-1)*.9
	saObj.Tmin = .001 # how big an increase in distance are we willing to accept at the end?

	state, distanceFromTargetSummary = saObj.anneal()
	print state
	print distanceFromTargetSummary
	print scriptStrings
	print
	print "************"

	outputString = makeProgram(scriptStrings, state[1:], state[0])
	print outputString

main()



