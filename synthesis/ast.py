from itertools import combinations

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
		self.parent = None

	def replace(self, nodeToCut, nodeToAdd):
		foundNode = False
		for i in range(len(self.children)):
			if self.children[i] == nodeToCut:
				self.children[i] = nodeToAdd
				nodeToAdd.parent = self
				foundNode = True
		if not foundNode:
			raise Exception("Tried to replace a node that wasn't actually a child.")
		return nodeToAdd

	def addChild(self, node):
		self.children.append(node)
		node.parent = self

	def strings(self, tabs=0):
		outputStrings = []
		for child in self.children:
			childStrings = child.strings()
			outputStrings = combineStrings([outputStrings, childStrings])
		return outputStrings

	def fillHolesForConcretePathConditions(self, dataset, pathCondition =[], currVariable = None):
		for node in self.children:
			node.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable)
        
        def accept(self, visitor):
            visitor.visit(self)

class VariableDeclNode(ASTNode):
	def __init__(self, name, varType, RHS):
		ASTNode.__init__(self)
		self.name = name
		self.varType = varType
		self.RHS = RHS
		
		self.RHS.parent = self

	def replace(self, nodeToCut, nodeToAdd):
		if self.RHS == nodeToCut:
			self.RHS = nodeToAdd
			nodeToAdd.parent = self
		else:
			raise Exception("Tried to replace a node that wasn't actually a child.")
		return nodeToAdd

	def strings(self, tabs=0):
		s = ["\nrandom "+self.varType+" "+self.name+" ~ "]
		RHSStrings = self.RHS.strings()
		return combineStrings([s, RHSStrings, [";\n"]])

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		self.RHS.fillHolesForConcretePathConditions(dataset, pathCondition, self) # the current node is now the variable being defined

class DistribNode(ASTNode):
  def __init__(self):
		ASTNode.__init__(self)

class BooleanDistribNode(DistribNode):
	def __init__(self, percentTrue=None):
		DistribNode.__init__(self)
		self.percentTrue = percentTrue

	def params(self):
		return self.percentTrue

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

		percentTrue = .5
		if matchingRowsCounter > 0:
			percentTrue = float(matchingRowsSum)/matchingRowsCounter
		self.percentTrue = percentTrue

class GaussianDistribNode(ASTNode):
	def __init__(self, mu=None, sig=None):
		ASTNode.__init__(self)
		self.mu = mu
                self.sig = sig

	def strings(self, tabs=0):
		if self.mu:
			return ["Gaussian(%f,%f)" % (self.mu, self.sig)]
		else:
			return ["Gaussian(",",", ")"]

class IfNode(ASTNode):
	def __init__(self, conditionNode, thenNode, elseNode):
		self.conditionNode = conditionNode
		self.thenNode = thenNode
		self.elseNode = elseNode
		self.concreteChildHolesFilled = []

		self.conditionNode.parent = self
		self.thenNode.parent = self
		self.elseNode.parent = self

	def strings(self, tabs=0):
		tabs = tabs + 1
		return combineStrings([["\n"+"\t"*tabs+"if "], self.conditionNode.strings(tabs), ["\n"+"\t"*tabs+"then "], self.thenNode.strings(tabs), ["\n"+"\t"*tabs+"else "], self.elseNode.strings(tabs)])

	def replace(self, nodeToCut, nodeToAdd):

		nodeToAdd.parent = self
		if self.conditionNode == nodeToCut:
			self.conditionNode = nodeToAdd
		elif self.thenNode == nodeToCut:
			self.thenNode = nodeToAdd
		elif self.elseNode == nodeToCut:
			self.elseNode = nodeToAdd
		else:
			raise Exception("Tried to replace a node that wasn't actually a child.")
		return nodeToAdd

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		pathConditions = self.conditionNode.pathConditions()
		truePathCondition = pathCondition + [pathConditions[0]]
		self.thenNode.fillHolesForConcretePathConditions(dataset, truePathCondition, currVariable)
		falsePathCondition = pathCondition + [pathConditions[1]]
		self.elseNode.fillHolesForConcretePathConditions(dataset, falsePathCondition, currVariable)
		if (isinstance(self.thenNode, DistribNode) and isinstance(self.elseNode, DistribNode) and abs(self.thenNode.params() - self.elseNode.params()) < .03):
			# replace this node with one of the branches
			self.parent.replace(self, self.thenNode)
			self.thenNode.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable)
			# and then we'll go back up to the parent and continue with hole filling

class VariableUseNode(ASTNode):
	def __init__(self, name):
		self.name = name

	def strings(self, tabs=0):
		return [self.name]

	def pathConditions(self):
		return [PathConditionComponent(self.name, "eq", True), PathConditionComponent(self.name, "eq", False)]

class BoolBinExpNode(ASTNode):
    def __init__(self, op, e1, e2):
        # op should be in {'&&','||'}
        self.op = op
        self.e1 = e1
        self.e2 = e2

    def strings(self, tabs=0):
        return self.e1.strings(tabs) + self.op + self.e2.strings(tabs)

class BinExpNode(ASTNode):
    def __init__(self, op, e1, e2):
        # op should be in {'+','-','*'}
        self.op = op
        self.e1 = e1
        self.e2 = e2

    def strings(self, tabs=0):
        return self.e1.strings(tabs) + self.op + self.e2.strings(tabs)

class UnaryExpNode(ASTNode):
    def __init__(self, op, e):
        # op should be in {'!'}
        self.op = op
        self.e = e

    def strings(self, tabs=0):
        return self.op + self.e.strings(tabs)


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
