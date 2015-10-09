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

	def strings(self):
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

	def strings(self):
		s = ["\nrandom "+self.varType+" "+self.name+" ~ "]
		RHSStrings = self.RHS.strings()
		return combineStrings([s, RHSStrings, [";\n"]])

class BooleanDistribNode(ASTNode):
	def __init__(self):
		ASTNode.__init__(self)

	def strings(self):
		return ["BooleanDistrib(", ")"]

class IfNode(ASTNode):
	def __init__(self, conditionNode, thenNode, elseNode):
		self.conditionNode = conditionNode
		self.thenNode = thenNode
		self.elseNode = elseNode

	def strings(self):
		return combineStrings([["if "], self.conditionNode.strings(), ["\n\tthen "], self.thenNode.strings(), ["\n\telse "], self.elseNode.strings()])

class VariableUseNode(ASTNode):
	def __init__(self, name):
		self.name = name

	def strings(self):
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

# **********************************************************************
# Consume the structure hints, generate a program
# **********************************************************************

def main():
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
		if len(node.parents) == 0:
			rhs = BooleanDistribNode()
			variableNode = VariableDeclNode(node.name, "Boolean", rhs)
			AST.children.append(variableNode)
		elif len(node.parents) == 1:
			conditionNode = VariableUseNode(node.parents[0].name)
			thenNode = BooleanDistribNode()
			elseNode = BooleanDistribNode()
			ifNode = IfNode(conditionNode, thenNode, elseNode)
			variableNode = VariableDeclNode(node.name, "Boolean", ifNode)

			AST.children.append(variableNode)


	scriptStrings = AST.strings()

	print scriptStrings
	print "??".join(scriptStrings)

main()



