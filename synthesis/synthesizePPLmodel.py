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

class ASTNode:
	def __init__(self):
		self.children = []

	def strings(self):
		outputStrings = []
		for child in self.children:
			childStrings = child.strings()
			outputStrings = combineNodeStringsToMaintainHoles(outputStrings, childStrings)
		return outputStrings

class VariableNode(ASTNode):
	def __init__(self, name, varType, RHS):
		ASTNode.__init__(self)
		self.name = name
		self.varType = varType
		self.RHS = RHS

	def strings(self):
		s = ["\nrandom "+self.varType+" "+self.name+" ~ "]
		RHSStrings = self.RHS.strings()
		return combineNodeStringsToMaintainHoles(s, RHSStrings)

class BooleanDistrib(ASTNode):
	def __init__(self):
		ASTNode.__init__(self)

	def strings(self):
		return ["BooleanDistrib(", ")"]

def combineNodeStringsToMaintainHoles(n1Strings, n2Strings):
	if len(n2Strings) < 1:
		return n1Strings
	if len(n1Strings) < 1:
		return n2Strings
	resStrings = n1Strings[:-1]	+ [n1Strings[-1]+n2Strings[0]] + n2Strings[1:]
	return resStrings

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
	if len(node.parents) == 0:
		rhs = BooleanDistrib()
		variableNode = VariableNode(node.name, "Boolean", rhs)
		AST.children.append(variableNode)

scriptStrings = AST.strings()

print scriptStrings



