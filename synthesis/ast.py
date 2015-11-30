from itertools import combinations

# **********************************************************************
# Supported distributions
# **********************************************************************

class BooleanDistribution:
	def __init__(self):
		return

class CategoricalDistribution:
	def __init__(self, values):
		self.values = values

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
		columnValues = []
		for i in range(numItems):
			columns.append([])
			columnValues.append(set())
		rows = []
		for line in lines[1:]:
			cells = []
			entries = line.strip().split(",")
			for i in range(numItems):
				entry = entries[i]
				cells.append(entry)
				columns[i].append(entry)
				columnValues[i].add(entry)
			rows.append(cells)

		self.columns = columns
		self.rows = rows
		self.numRows = len(rows)

		columnDistributionInformation = []
		for i in range(len(columnValues)):
			currColumnValues = columnValues[i]
			if currColumnValues == set(["true", "false"]):
				columnDistributionInformation.append(BooleanDistribution())
			else:
				columnDistributionInformation.append(CategoricalDistribution(list(currColumnValues)))
		self.columnDistributionInformation = columnDistributionInformation

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

	def reduce(self, dataset, pathCondition =[], currVariable = None):
		for node in self.children:
			node.reduce(dataset, pathCondition, currVariable)
        
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

	def reduce(self, dataset, pathCondition, currVariable):
		self.RHS.reduce(dataset, pathCondition, self) # the current node is now the variable being defined

class DistribNode(ASTNode):
  def __init__(self):
		ASTNode.__init__(self)

class BooleanDistribNode(DistribNode):
	def __init__(self, percentTrue=None, percentMatchingRows = 0):
		DistribNode.__init__(self)
		self.percentTrue = percentTrue
		self.percentMatchingRows = percentMatchingRows

	def params(self):
		return [("Boolean", self.percentTrue, self.percentMatchingRows)]

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
				if val == "true":
					matchingRowsSum += 1

		percentTrue = "??"
		if matchingRowsCounter > 0:
			percentTrue = float(matchingRowsSum)/matchingRowsCounter
		self.percentTrue = percentTrue
		self.percentMatchingRows = float(matchingRowsCounter)/dataset.numRows

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

class CategoricalDistribNode(DistribNode):
	def __init__(self, values, valuesToPercentages = None, percentMatchingRows = 0):
		DistribNode.__init__(self)
		self.values = values
		self.valuesToPercentages = valuesToPercentages
		self.percentMatchingRows = percentMatchingRows

	def params(self):
		return [("Categorical", self.valuesToPercentages, self.percentMatchingRows)]

	def strings(self, tabs=0):
		components = ["Categorical({", "})"]
		if self.valuesToPercentages:
			innards = []
			for value in self.values: # use values because it has the guaranteed stable ordering
				innards.append(str(value) + " -> " + str(self.valuesToPercentages[value]))
			return [components[0]+(",".join(innards))+components[1]]
		else:
			return components

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		pathConditionFilter = dataset.makePathConditionFilter(pathCondition)
		currVariableGetter = dataset.makeCurrVariableGetter(currVariable)
		matchingRowsCounter = 0
		matchingRowsSums = {}
		for row in dataset.rows:
			if pathConditionFilter(row):
				matchingRowsCounter += 1
				val = currVariableGetter(row)
				count = matchingRowsSums.get(val, 0)
				matchingRowsSums[val] = count + 1

		self.percentMatchingRows = float(matchingRowsCounter)/dataset.numRows

		if matchingRowsCounter < 1:
			self.valuesToPercentages = "??"
			return

		self.valuesToPercentages = {}
		for value in self.values:
			percentMatching = float(matchingRowsSums[value])/matchingRowsCounter
			self.valuesToPercentages[value] = percentMatching

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

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

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

class IfNode(ASTNode):
	def __init__(self, conditionNodes, bodyNodes):
		self.conditionNodes = conditionNodes
		self.bodyNodes = bodyNodes
		for node in self.conditionNodes:
			node.parent = self
		for node in self.bodyNodes:
			node.parent = self

	def params(self):
		paramsLs = []
		for bodyNode in self.bodyNodes:
			paramsLs = paramsLs + bodyNode.params()
		return paramsLs

	def strings(self, tabs=0):
		tabs = tabs + 1
		stringListsToCombine = []
		first = True

		for i in range(len(self.bodyNodes) - 1):
			conditionNodeStrings = self.conditionNodes[i].strings(tabs)
			bodyNodeStrings = self.bodyNodes[i].strings(tabs)
			if first:
				stringListsToCombine.append(["\n"+"\t"*tabs+"if "])
				first = False
			else:
				stringListsToCombine.append(["\n"+"\t"*tabs+"else if "])

			stringListsToCombine.append(conditionNodeStrings)
			stringListsToCombine.append(["\n"+"\t"*tabs+"then "])
			stringListsToCombine.append(bodyNodeStrings)

		# the last one is always an else
		bodyNodeStrings = self.bodyNodes[-1].strings(tabs)
		stringListsToCombine.append(["\n"+"\t"*tabs+"else "])
		stringListsToCombine.append(bodyNodeStrings)

		return combineStrings(stringListsToCombine)

	def replace(self, nodeToCut, nodeToAdd):
		replaced = False
		nodeToAdd.parent = self
		for i in range(len(self.conditionNodes)):
			if self.conditionNodes[i] == nodeToCut:
				self.conditionNodes[i] = nodeToAdd
				replaced = True
		for i in range(len(self.bodyNodes)):
			if self.bodyNodes[i] == nodeToCut:
				self.bodyNodes[i] = nodeToAdd
				replaced = True
		if not replaced:
			raise Exception("Tried to replace a node that wasn't actually a child.")
		return nodeToAdd

	def reduce(self, dataset, pathCondition, currVariable):
		for pair in combinations(range(len(self.bodyNodes)), 2):
			i = pair[0]
			j = pair[1]
			params1 = self.bodyNodes[i].params()
			params2 = self.bodyNodes[j].params()
			match = True
			# because we always construct then and else branches to be the same, we can rely on the structure to be the same, don't need to check path conditions
			for i in range(len(params1)):
				param1 = params1[i] # a tuple of distrib type, relevant parmas, num of rows on which based; should eventually add path condition
				param2 = params2[i]
				if (param1[0] == "Boolean" and param2[0] == "Boolean"):
					if (param1[1] == "??" or param2[1] == "??"):
						# for this param, let anything match, since we don't know what its value should be
						continue

					# threshold to beat should depend on how much data we have
					# if we have a huge dataset, should only collapse if the variation is pretty big
					# if we have a smaller dataset, even a biggish variation could be noise
					# for a dataset of size 10,000, I've found .02 seems pretty good (thresholdmaker 200)
					# for a dataset of size 500,000, .0001 was better (thresholdmaker 50)

					thresholdMaker = 150.0
					thresholdToBeat = thresholdMaker/dataset.numRows
					# the threshold to beat should depend on how much data we used to make each estimate
					# if the data is evenly divided between the if and the else, we should use the base thresholdToBeat.  else, should use higher
					minNumRows = min(param1[2], param2[2])
					rowsRatio = minNumRows/.5
					# if small number of rows, can see a big difference and still consider them equiv, so use a higher threshold before we declare them different
					thresholdToBeat = thresholdToBeat/rowsRatio
					if (abs(param1[1] - param2[1]) > thresholdToBeat):
						match = False
						break
				if (param1[0] == "Categorical" and param2[0] == "Categorical"):
					if (param1[1] == "??" or param2[1] == "??"):
						continue
					thresholdMaker = 150.0
					thresholdToBeat = thresholdMaker/dataset.numRows
					minNumRows = min(param1[2], param2[2])
					rowsRatio = minNumRows/.5
					thresholdToBeat = thresholdToBeat/rowsRatio

					allValuesMatch = True
					dict1 = param1[1]
					dict2 = param2[1]
					for value in dict1:
						if (abs(dict1[value] - dict2[value]) > thresholdToBeat):
							allValuesMatch = False
							break
					if not allValuesMatch:
						match = False
						break

			if match:
				# we're either in a case where we were down to 2 conditions, should just eliminate the if
				# or we're in the case where we should combine 2 conditions among a number of other conditions

				if len(self.bodyNodes) == 2:
					# replace this node with one of the branches
					self.parent.replace(self, self.bodyNodes[0])
					self.bodyNodes[0].fillHolesForConcretePathConditions(dataset, pathCondition, currVariable)
					# and now we need to continue with reductions
					self.bodyNodes[0].reduce(dataset, pathCondition, currVariable)

					# don't want to keep trying to reduce the body nodes of this node, since it's now been eliminated, so return
					return
				else:
					# combine a couple branches
					newConditionNode = OrNode(self.conditionNodes[i], self.conditionNodes[j])
					self.conditionNodes[i] = newConditionNode

					# adapt the body node fillded holes to our new condition
					pathConditionAdditional = self.conditionNodes[i].pathCondition()
					self.bodyNodes[i].fillHolesForConcretePathConditions(dataset, pathCondition + [pathConditionAdditional], currVariable)

					# now delete the ones we're getting rid of
					del self.conditionNodes[j]
					del self.bodyNodes[j]

		# once we've gotten rid of all the things we can reduce, go ahead and descend down all the remaining body nodes
		for i in range(len(self.bodyNodes)):
			if i < len(self.conditionNodes):
				pathConditionAdditional = self.conditionNodes[i].pathCondition()
			else:
				# if there's no condition associated with the last, it better be because the last one was a condition that has a false associated
				pathConditionAdditional = self.conditionNodes[i-1].pathConditionFalse()
			newPathCondition = pathCondition + [pathConditionAdditional]
			self.bodyNodes[i].reduce(dataset, newPathCondition, currVariable)

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		for i in range(len(self.bodyNodes)):
			if i < len(self.conditionNodes):
				pathConditionAdditional = self.conditionNodes[i].pathCondition()
			else:
				# if there's no condition associated with the last, it better be because the last one was a condition that has a false associated
				pathConditionAdditional = self.conditionNodes[i-1].pathConditionFalse()
			newPathCondition = pathCondition + [pathConditionAdditional]
			self.bodyNodes[i].fillHolesForConcretePathConditions(dataset, newPathCondition, currVariable)

class VariableUseNode(ASTNode):
	def __init__(self, name):
		self.name = name

	def strings(self, tabs=0):
		return [self.name]

	def pathCondition(self):
		return PathConditionComponent(self.name, "eq", "true")

	def pathConditionFalse(self):
		return PathConditionComponent(self.name, "eq", "false")

class ComparisonNode(ASTNode):
	def __init__(self, variableNode, value):
		self.node = variableNode
		self.value = value

	def strings(self, tabs=0):
		return [self.node.name + " == " + self.value]

	def pathCondition(self):
		return PathConditionComponent(self.node.name, "eq", self.value)

class OrNode(ASTNode):
	def __init__(self, leftNode, rightNode):
		self.lNode = leftNode
		self.rNode = rightNode

	def strings(self, tabs=0):
		return combineStrings(self.lNode.strings(), [" | "], self.rNode.strings())

	def pathConditions(self):
		raise "Freak out don't know how this is used"

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
