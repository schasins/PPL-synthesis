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

for node in nodesInDependencyOrder:
	print node.name

