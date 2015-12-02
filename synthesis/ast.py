from itertools import combinations
import operator
import random

# **********************************************************************
# Supported distributions
# **********************************************************************

class BooleanDistribution:
	def __init__(self):
		self.typeName = "Boolean"

class CategoricalDistribution:
	def __init__(self, values, typeName):
		self.values = values
		self.typeName = typeName

class IntegerDistribution:
	def __init__(self):
		self.typeName = "Integer"

class RealDistribution:
	def __init__(self):
		self.typeName = "Real"

# **********************************************************************
# Data structures for creating PPL ASTs
# **********************************************************************

def isInteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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
				names[i] = lineItem.replace("(", "").replace(")", "")
				indexes[names[i]] = i
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
		columnMaxes = {}
		columnMins = {}
		for i in range(len(columnValues)):
			currColumnValues = columnValues[i]
			if currColumnValues == set(["true", "false"]):
				columnDistributionInformation.append(BooleanDistribution())
			elif reduce(lambda x, y: x and isInteger(y), currColumnValues, True):
				columnDistributionInformation.append(IntegerDistribution())
				self.columns[i] = (map(lambda x: int(x), currColumnValues))
				columnMaxes[names[i]] = max(self.columns[i])
				columnMins[names[i]] = min(self.columns[i])
				columnMins.append(min(self.columns[i]))
			elif reduce(lambda x, y: x and isFloat(y), currColumnValues, True):
				columnDistributionInformation.append(RealDistribution())
				self.columns[i] = (map(lambda x: float(x), currColumnValues))
				columnMaxes[names[i]] = max(self.columns[i])
				columnMins[names[i]] = min(self.columns[i])
			else:
				columnDistributionInformation.append(CategoricalDistribution(list(currColumnValues), names[i]+"Type"))
		self.columnDistributionInformation = columnDistributionInformation
		self.columnMaxes = columnMaxes
		self.columnMins = columnMins

	def makePathConditionFilter(self, pathCondition):
		pairs = [] # (index, func) pairs
		for pathConditionComponent in pathCondition:
			pairs.append(( map( lambda x: self.namesToIndexes[x], pathConditionComponent.varNames), pathConditionComponent.func))
		return lambda row : reduce(lambda x, pair : x and apply(pair[1], map( lambda arg: row[arg], pair[0])), pairs, True)

	def makeCurrVariableGetter(self, currVariable):
		index = self.namesToIndexes[currVariable.name]
		return lambda row: row[index]

class PathConditionComponent:
	def __init__(self, varNames, func):
		self.varNames = varNames
		self.func = func

# **********************************************************************
# Data structures for representing PPL ASTs
# **********************************************************************

class Program:
	def __init__(self, dataset):
		self.randomizeableNodes = []
		self.variables = []
		self.dataset = dataset
		self.root = None

	def setRoot(self, root):
		self.root = root
		root.setProgram(self)

	def variableRange(self, variableName):
		return (self.dataset.columnMins[variableName], self.dataset.columnMaxes[variableName])

	def mutate(self):
		node = random.choice(list(self.randomizeableNodes))
		#print "********"
		#print node
		#print node.strings()
		#print "********"
		node.mutate()

	def programString(self):
		return self.root.strings()[0]

class ASTNode:
	def __init__(self):
		self.children = []
		self.parent = None

	def setProgram(self, program):
		self.program = program
		for child in self.children:
			child.setProgram(program)

	def setParent(self, parentNode):
		self.parent = parentNode

	def replace(self, nodeToCut, nodeToAdd):
		foundNode = False
		for i in range(len(self.children)):
			if self.children[i] == nodeToCut:
				self.children[i] = nodeToAdd
				nodeToAdd.setParent(self)
				foundNode = True
		if not foundNode:
			raise Exception("Tried to replace a node that wasn't actually a child.")
		return nodeToAdd

	def addChild(self, node):
		self.children.append(node)
		node.setParent(self)

	def strings(self, tabs=0):
		outputStrings = []
		for child in self.children:
			childStrings = child.strings()
			outputStrings = combineStrings([outputStrings, childStrings])
		return outputStrings

	def fillHolesForConcretePathConditions(self, dataset, pathCondition =[], currVariable = None):
		for node in self.children:
			node.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable)

	def fillHolesRandomly(self):
		for node in self.children:
			node.fillHolesRandomly()

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
		
		self.RHS.setParent(self)

	def setProgram(self, program):
		self.program = program
		self.RHS.setProgram(program)

	def replace(self, nodeToCut, nodeToAdd):
		if self.RHS == nodeToCut:
			self.RHS = nodeToAdd
			nodeToAdd.setParent(self)
		else:
			raise Exception("Tried to replace a node that wasn't actually a child.")
		return nodeToAdd

	def strings(self, tabs=0):
		s = ["\nrandom "+self.varType+" "+self.name+" ~ "]
		RHSStrings = self.RHS.strings()
		return combineStrings([s, RHSStrings, [";\n"]])

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		self.RHS.fillHolesForConcretePathConditions(dataset, pathCondition, self) # the current node is now the variable being defined

	def fillHolesRandomly(self):
		self.RHS.fillHolesRandomly()
		self.program.variables.append(self)

	def reduce(self, dataset, pathCondition, currVariable):
		self.RHS.reduce(dataset, pathCondition, self) # the current node is now the variable being defined

class TypeDeclNode(ASTNode):

	def __init__(self, name, values):
		ASTNode.__init__(self)
		self.name = name
		self.values = values

	def setProgram(self, program):
		self.program = program

	def strings(self, tabs = 0):
		vals = ", ".join(self.values)
		return ["\ntype " + self.name + ";\ndistinct " + self.name + " " + vals + ";"]

class DistribNode(ASTNode):
	def __init__(self):
		ASTNode.__init__(self)

	def setProgram(self, program):
		self.program = program

class BooleanDistribNode(DistribNode):
	def __init__(self, varName, percentTrue=None, percentMatchingRows = None):
		DistribNode.__init__(self)
		self.percentTrue = percentTrue
		self.varName = varName
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

		percentTrue = None
		if matchingRowsCounter > 0:
			percentTrue = float(matchingRowsSum)/matchingRowsCounter
		self.percentTrue = percentTrue
		self.percentMatchingRows = float(matchingRowsCounter)/dataset.numRows

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def mutate(self):
		if self.percentTrue == None or random.uniform(0, 1) < .3:
			# completely overwrite
			self.percentTrue = random.uniform(0, 1)
		else:
			# make a small adjustment
			self.percentTrue = self.percentTrue + random.uniform(-.1,.1)

class CategoricalDistribNode(DistribNode):
	def __init__(self, varName, values, valuesToPercentages = None, percentMatchingRows = None):
		DistribNode.__init__(self)
		self.varName = varName
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
			return [components[0]+(", ".join(innards))+components[1]]
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
			self.valuesToPercentages = None
			return

		self.valuesToPercentages = {}
		for value in self.values:
			percentMatching = float(matchingRowsSums[value])/matchingRowsCounter
			self.valuesToPercentages[value] = percentMatching

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def mutate(self):
		# TODO : go through all the mutates and make sure they don't go beyond the bounds
		if self.valuesToPercentages == None:
			self.valuesToPercentages = {}
			for value in self.values:
				self.valuesToPercentages[value] = random.uniform(0, 1) # note that BLOG automatically normalizes so they sum to 1
		elif random.uniform(0, 1) < .3:
			# completely overwrite
			value = random.choice(self.values)
			self.valuesToPercentages[value] = random.uniform(0, 1)
		else:
			# make a small adjustment
			value = random.choice(self.values)
			self.valuesToPercentages[value] = self.valuesToPercentages[value] + random.uniform(-.1,.1)


class RealDistribNode(DistribNode):

	def __init__(self, varName, actualDistribNode = None):
		DistribNode.__init__(self)
		self.varName = varName
		self.actualDistribNode = actualDistribNode
		self.availableNodeTypes = [GaussianDistribNode, BetaDistribNode]

	def strings(self, tabs=0):
		if self.actualDistribNode == None:
			return [",", ","]
		else:
			return self.actualDistribNode.strings()

	def params(self):
		return None

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def fillHolesRandomly(self):
		self.mutate()
		# add this to the set of randomizeable nodes since we can replace the actualDistribNode
		self.program.randomizeableNodes.append(self)

	def mutate(self):
		nodeType = random.choice(self.availableNodeTypes)
		self.actualDistribNode = nodeType(self.varName)
		self.actualDistribNode.setProgram(self.program)
		self.actualDistribNode.fillHolesRandomly() # fill in params, add the node to the randomizeable nodes

def overwriteOrModifyOneParam(overWriteProb, paramsLs, lowerLimit, upperLimit, modificationLowerLimit, modificationUpperLimit):
	indexToChange = random.choice(range(len(paramsLs)))
	if random.uniform(0,1) < overWriteProb:
		paramsLs[indexToChange] = random.uniform(lowerLimit, upperLimit)
	else:
		paramsLs[indexToChange] = paramsLs[indexToChange] + random.uniform(modificationLowerLimit, modificationUpperLimit)
		if paramsLs[indexToChange] > upperLimit:
			paramsLs[indexToChange] = upperLimit
		elif paramsLs[indexToChange] < lowerLimit:
			paramsLs[indexToChange] = lowerLimit
	return paramsLs

class GaussianDistribNode(RealDistribNode):
	def __init__(self, varName, mu=None, sig=None, percentMatchingRows = None):
		RealDistribNode.__init__(self, varName)
		self.mu = mu
		self.sig = sig
		self.percentMatchingRows = percentMatchingRows

	def strings(self, tabs=0):
		if self.mu:
			return ["Gaussian(%f,%f)" % (self.mu, self.sig)]
		else:
			return ["Gaussian(",",", ")"]

	def params(self):
		return [("Gaussian", (self.mu, self.sig), self.percentMatchingRows)]

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def fillHolesRandomly(self):
		self.mutate()
		self.program.randomizeableNodes.append(self)

	def mutate(self):
		(lowerBound, upperBound) = self.program.variableRange(self.varName)
		if self.mu == None:
			self.mu = random.uniform(lowerBound, upperBound) # TODO what's actually a good upper limit?
			self.sig = random.uniform(0, upperBound)
		else:
			# TODO: putting in a temporary fix for the fact they don't have the same bounds, should do something nicer
			if random.uniform(0,1) < .5:
				modParams = overwriteOrModifyOneParam(.3, [self.mu], lowerBound, upperBound, -.1, .1)
				self.mu = modParams[0]
			else:
				modParams = overwriteOrModifyOneParam(.3, [self.sig], 0, upperBound, -.1, .1)
				self.sig = modParams[0]

class BetaDistribNode(RealDistribNode):
	def __init__(self, varName, alpha=None, beta=None, percentMatchingRows = None):
		RealDistribNode.__init__(self, varName)
		self.alpha = alpha
		self.beta = beta
		self.percentMatchingRows = percentMatchingRows

	def strings(self, tabs=0):
		if self.alpha:
			return ["Beta(%f,%f)" % (self.alpha, self.beta)]
		else:
			return ["Beta(",",", ")"]

	def params(self):
		return [("Beta", (self.alpha, self.beta), self.percentMatchingRows)]

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def fillHolesRandomly(self):
		self.mutate()
		self.program.randomizeableNodes.append(self)

	def mutate(self):
		(lowerBound, upperBound) = self.program.variableRange(self.varName) # TODO: what is this param?  what's a good limit?
		# both alpha and beta need to be > 0
		lowerBound = .000000000000001
		if self.alpha == None:
			self.alpha = random.uniform(lowerBound, upperBound) 
			self.beta = random.uniform(lowerBound, upperBound)
		else:
			modParams = overwriteOrModifyOneParam(.3, [self.alpha, self.beta], lowerBound, upperBound, -.1, .1)
			self.alpha = modParams[0]
			self.beta = modParams[1]

class UniformRealDistribNode(RealDistribNode):
	def __init__(self, varName, a=None, b=None, percentMatchingRows = None):
		RealDistribNode.__init__(self, varName)
		self.a = a
		self.b = b
		self.percentMatchingRows = percentMatchingRows

	def strings(self, tabs=0):
		if self.a:
			return ["UniformReal(%f,%f)" % (self.a, self.b)]
		else:
			return ["UniformReal(",",", ")"]

	def params(self):
		return [("UniformReal", (self.a, self.b), self.percentMatchingRows)]

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def fillHolesRandomly(self):
		self.mutate()
		self.program.randomizeableNodes.append(self)

	def mutate(self):
		(lowerBound, upperBound) = self.program.variableRange(self.varName)
		if self.a == None:
			self.a = random.uniform(lowerBound, upperBound) 
			self.b = random.uniform(self.a, upperBound)
		else:
			modParams = overwriteOrModifyOneParam(.3, [self.a, self.b], lowerBound, upperBound, -.1, .1)
			self.a = modParams[0]
			self.b = modParams[1]
			if self.b < self.a:
				tmp = self.b
				self.b = self.a
				self.a = self.b

class IfNode(ASTNode):
	def __init__(self, conditionNodes, bodyNodes):
		ASTNode.__init__(self)
		self.conditionNodes = conditionNodes
		self.bodyNodes = bodyNodes
		for node in self.conditionNodes:
			node.setParent(self)
		for node in self.bodyNodes:
			node.setParent(self)

	def setProgram(self, program):
		self.program = program
		for child in self.conditionNodes:
			child.setProgram(program)
		for child in self.bodyNodes:
			child.setProgram(program)

	def params(self):
		paramsLs = []
		for bodyNode in self.bodyNodes:
			newParams = bodyNode.params()
			if newParams == None:
				# if any of the branches have non-concrete params, the whole thing has non-concrete
				return None
			paramsLs = paramsLs + newParams
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
		nodeToAdd.setParent(self)
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
			p1i = pair[0]
			p2i = pair[1]
			params1 = self.bodyNodes[p1i].params()
			params2 = self.bodyNodes[p2i].params()
			if params1 == None or params2 == None:
				# the path conditions going down aren't concrete, so it doesn't make sense to reduce yet
				continue
			match = True
			# because we always construct then and else branches to be the same, we can rely on the structure to be the same, don't need to check path conditions
			for i in range(len(params1)):
				param1 = params1[i] # a tuple of distrib type, relevant parmas, num of rows on which based; should eventually add path condition
				param2 = params2[i]
				if (param1[0] == "Boolean" and param2[0] == "Boolean"):
					if (param1[1] == None or param2[1] == None):
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
					if (param1[1] == None or param2[1] == None):
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
					newConditionNode = BoolBinExpNode("|", self.conditionNodes[p1i], self.conditionNodes[p2i])
					self.conditionNodes[p1i] = newConditionNode

					# adapt the body node fillded holes to our new condition
					pathConditionAdditional = self.conditionNodes[p1i].pathCondition()
					self.bodyNodes[p1i].fillHolesForConcretePathConditions(dataset, pathCondition + [pathConditionAdditional], currVariable)

					# now delete the ones we're getting rid of
					del self.conditionNodes[p2i]
					del self.bodyNodes[p2i]

					# now that we've deleted stuff, the i and j indexes don't work anymore
					self.reduce(dataset, pathCondition, currVariable)
					return

		# once we've gotten rid of all the things we can reduce, go ahead and descend down all the remaining body nodes
		for i in range(len(self.bodyNodes)):
			if i < len(self.conditionNodes):
				pathConditionAdditional = self.conditionNodes[i].pathCondition()
				if pathConditionAdditional == None:
					# the path condition is no longer concrete
					continue
			else:
				# if there's no condition associated with the last, it better be because the last one was a condition that has a false associated
				pathConditionAdditional = self.conditionNodes[i-1].pathConditionFalse()
			newPathCondition = pathCondition + [pathConditionAdditional]
			self.bodyNodes[i].reduce(dataset, newPathCondition, currVariable)

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		for i in range(len(self.bodyNodes)):
			if i < len(self.conditionNodes):
				pathConditionAdditional = self.conditionNodes[i].pathCondition()
				if pathConditionAdditional == None:
					# the path condition is no longer concrete
					continue
			else:
				# if there's no condition associated with the last, it better be because the last one was a condition that has a false associated
				pathConditionAdditional = self.conditionNodes[i-1].pathConditionFalse()
			newPathCondition = pathCondition + [pathConditionAdditional]
			self.bodyNodes[i].fillHolesForConcretePathConditions(dataset, newPathCondition, currVariable)

	def fillHolesRandomly(self):
		for node in self.conditionNodes:
			node.fillHolesRandomly()
		for node in self.bodyNodes:
			node.fillHolesRandomly()

		# TODO: should probably only add this to randomizeable if some subset of the child nodes actually get randomized
		# should start recording that

		# can randomize by removing the if
		self.program.randomizeableNodes.append(self)

	def mutate(self):
		self.parent.replace(self, self.bodyNodes[0])
		self.program.randomizeableNodes.remove(self)

class VariableUseNode(ASTNode):
	def __init__(self, name, typeName):
		ASTNode.__init__(self)
		self.name = name
		self.typeName = typeName

	def setProgram(self, program):
		self.program = program

	def strings(self, tabs=0):
		return [self.name]

	def pathCondition(self):
		return PathConditionComponent([self.name], lambda x: x == "true")

	def pathConditionFalse(self):
		return PathConditionComponent([self.name], lambda x: x == "false")

	def range(self):
		return self.program.variableRange(self.name)

class ComparisonNode(ASTNode):

	ops = {	"==": operator.eq,
			">": operator.gt,
			"<": operator.lt}

	def __init__(self, variableNode, relationship = None, value = None):
		ASTNode.__init__(self)
		self.node = variableNode
		self.relationship = relationship
		self.value = value

	def setProgram(self, program):
		self.program = program
		self.node.setProgram(program)

	def strings(self, tabs=0):
		if self.relationship:
			return [self.node.name + " " + self.relationship + " " + str(self.value)]
		else:
			return [self.node.name, ""]

	def pathCondition(self):
		return PathConditionComponent([self.node.name], lambda x: self.ops[self.relationship](x, self.value))

	def pathConditionFalse(self):
		return PathConditionComponent([self.node.name], lambda x: not self.ops[self.relationship](x, self.value))

	def fillHolesRandomly(self):
		self.mutate()
		self.program.randomizeableNodes.append(self)

	def mutate(self):
		(lowerBound, upperBound) = self.node.range()
		if (self.relationship == None or random.uniform(0,1) < .1):
			self.relationship = random.choice(self.ops.keys())
		else:
			overwriteOrModifyOneParam(.3, [self.value], lowerBound, upperBound, -.1, .1)
		if (self.value == None):
			self.value = random.uniform(lowerBound, upperBound)

class BoolBinExpNode(ASTNode):

	ops = {	"&": operator.__and__,
			"|": operator.__or__}

	def __init__(self, op, e1, e2):
		ASTNode.__init__(self)
		# op should be in {'&&','||'}
		self.op = op
		self.e1 = e1
		self.e2 = e2

	def setProgram(self, program):
		self.program = program
		self.e1.setProgram(program)
		self.e2.setProgram(program)

	def strings(self, tabs=0):
		return combineStrings([self.e1.strings(), [" "+self.op+" "], self.e2.strings()])

	def pathCondition(self):
		p1 = self.e1.pathCondition()
		p2 = self.e2.pathCondition()
		return PathConditionComponent(p1.varNames + p2.varNames, lambda x, y: self.ops[self.op](p1.func(x), p2.func(y)))

class BinExpNode(ASTNode):
	def __init__(self, op, e1, e2):
		ASTNode.__init__(self)
		# op should be in {'+','-','*'}
		self.op = op
		self.e1 = e1
		self.e2 = e2

	def strings(self, tabs=0):
		return self.e1.strings(tabs) + self.op + self.e2.strings(tabs)

class UnaryExpNode(ASTNode):
	def __init__(self, op, e):
		ASTNode.__init__(self)
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
