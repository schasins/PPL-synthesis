from itertools import combinations
import operator
import random
import numpy as np
from copy import deepcopy

debug = False
mutationDebug = False

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
		columnNumericColumns = []
		columnMaxes = {}
		columnMins = {}
		for i in range(len(columnValues)):
			currColumnValues = columnValues[i]
			if currColumnValues == set(["true", "false"]):
				columnDistributionInformation.append(BooleanDistribution())
				ls = map(lambda x: 1*(x == "true"), self.columns[i])
				columnNumericColumns.append([ls])
			elif reduce(lambda x, y: x and isInteger(y), currColumnValues, True):
				columnDistributionInformation.append(IntegerDistribution())
				self.columns[i] = (map(lambda x: int(x), currColumnValues))
				columnMaxes[names[i]] = max(self.columns[i])
				columnMins[names[i]] = min(self.columns[i])
				columnMins.append(min(self.columns[i]))
				columnNumericColumns.append([self.columns[i]])
				for row in self.rows:
					row[i] = int(row[i])
			elif reduce(lambda x, y: x and isFloat(y), currColumnValues, True):
				columnDistributionInformation.append(RealDistribution())
				self.columns[i] = (map(lambda x: float(x), currColumnValues))
				columnMaxes[names[i]] = max(self.columns[i])
				columnMins[names[i]] = min(self.columns[i])
				columnNumericColumns.append([self.columns[i]])
				for row in self.rows:
					row[i] = float(row[i])
			else:
				columnDistributionInformation.append(CategoricalDistribution(list(currColumnValues), names[i]+"Type"))
				lists = []
				for val in currColumnValues:
					ls = map(lambda x: 1*(x == val), self.columns[i])
					lists.append(ls)
				columnNumericColumns.append(lists)

		self.columnDistributionInformation = columnDistributionInformation
		self.columnNumericColumns = columnNumericColumns
		self.columnMaxes = columnMaxes
		self.columnMins = columnMins

	def makePathConditionFilter(self, pathCondition):
		return lambda row : reduce(lambda x, condComponent : x and condComponent.func(row), pathCondition, True) # each pathconditioncomponent in the pathcondition has a func associated

	def makeCurrVariableGetter(self, currVariable):
		index = self.namesToIndexes[currVariable.name]
		return lambda row: row[index]

class PathConditionComponent:
	def __init__(self, func):
		self.func = func

# **********************************************************************
# Data structures for representing PPL ASTs
# **********************************************************************

class Program:
	def __init__(self, dataset):
		self.randomizeableNodes = {}
		self.variables = []
		self.dataset = dataset
		self.root = None

	def setRoot(self, root):
		self.root = root
		root.setProgram(self)

	def variableRange(self, variableName):
		return (self.dataset.columnMins[variableName], self.dataset.columnMaxes[variableName])

	def instanceToKey(self, instance):
		return instance.__class__.__name__

	def addRandomizeableNode(self, node):
		key = self.instanceToKey(node)
		nodes = self.randomizeableNodes.get(key, set())
		nodes.add(node)
		self.randomizeableNodes[key] = nodes

	def removeRandomizeableNode(self, node):
		key = self.instanceToKey(node)
		nodes = self.randomizeableNodes.get(key)
		nodes.remove(node)
		self.randomizeableNodes[key] = nodes

	def mutate(self):
		# TODO: adding too many ifs stop

		totalWeight = 0
		thresholds = []
		associatedKeys = []
		for key in self.randomizeableNodes:
			nodes = self.randomizeableNodes[key]
			if key == "IfNode":
				weight = .05*len(nodes)
			elif key == "ComparisonNode":
				weight = .7*len(nodes)
			elif key == "RealDistribNode":
				weight = .25*len(nodes)
			else:
				raise Exception("hey we haven't handled this key yet: "+key)
			totalWeight += weight
			thresholds.append(totalWeight)
			associatedKeys.append(key)
		
		decision = random.uniform(0, totalWeight)
		for i in range(len(thresholds)):
			if decision < thresholds[i]:
				node = random.choice(list(self.randomizeableNodes[associatedKeys[i]]))
				# print "********"
				# print node
				# print node.strings()
				# print "********"
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
		if debug: print "concrete: ast"
		for node in self.children:
			node.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable)

	def fillHolesRandomly(self):
		if debug: print "randomly: ast"
		filledSomeHoles = False
		for node in self.children:
			filledSomeHoles = node.fillHolesRandomly() or filledSomeHoles
		return filledSomeHoles

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
		self.allowableVariables = []
		
		self.RHS.setParent(self)

	def setProgram(self, program):
		self.program = program

		self.allowableVariables = self.program.variables[:]

		useNode = VariableUseNode(self.name, self.varType)
		useNode.setProgram(program)
		self.program.variables.append(useNode)

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
		if debug: print "concrete: vardecl", self.name
		self.RHS.fillHolesForConcretePathConditions(dataset, pathCondition, self) # the current node is now the variable being defined

	def fillHolesRandomly(self):
		if debug: print "randomly: vardecl", self.name
		filledSomeHoles = self.RHS.fillHolesRandomly()
		return filledSomeHoles

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
		components = ["BooleanDistrib(", ") /*"+str(self.percentMatchingRows)+"*/"]
		return [components[0]+str(self.percentTrue)+components[1]]

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
		else:
			# there were no matching rows, doesn't matter what we put here
			percentTrue = .5
		self.percentTrue = percentTrue
		self.percentMatchingRows = float(matchingRowsCounter)/dataset.numRows
		if debug: print "concrete: bool", self.strings()

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

	def getRandomizeableNodes(self):
		return []

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
		#print ";".join(map(str, pathCondition))
		for row in dataset.rows:
			if pathConditionFilter(row):
				matchingRowsCounter += 1
				val = currVariableGetter(row)
				count = matchingRowsSums.get(val, 0)
				matchingRowsSums[val] = count + 1

		self.percentMatchingRows = float(matchingRowsCounter)/dataset.numRows

		if matchingRowsCounter < 1:
			self.valuesToPercentages = None
			#TODO: do we want to add this to randomizable nodes?
			return

		self.valuesToPercentages = {}
		for value in self.values:
			matching = 0
			if value in matchingRowsSums:
				matching = matchingRowsSums[value]
			percentMatching = float(matching)/matchingRowsCounter
			self.valuesToPercentages[value] = percentMatching
		if debug: print "concrete: categorical", self.strings()

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
		self.availableNodeTypes = [UniformRealDistribNode, GaussianDistribNode]
		self.availableNodes = []
		self.matchingRowsValues = []
		self.randomizeable = False

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

	def getRandomizeableNodes(self):
		ls = []
		if self.randomizeable:
			ls.append(self)
		return ls

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		pathConditionFilter = dataset.makePathConditionFilter(pathCondition)
		currVariableGetter = dataset.makeCurrVariableGetter(currVariable)
		matchingRowsCounter = 0
		matchingRowsValues = []
		for row in dataset.rows:
			if pathConditionFilter(row):
				matchingRowsCounter += 1
				val = currVariableGetter(row)
				matchingRowsValues.append(val)

		self.matchingRowsValues = matchingRowsValues
		self.availableNodes = []
		for distribType in self.availableNodeTypes:
			if distribType == BetaDistribNode:
				if len(self.matchingRowsValues) > 0 and min(self.matchingRowsValues) >= 0 and max(self.matchingRowsValues) <= 1:
					newNode = BetaDistribNode(self.varName)
					newNode.setProgram(self.program)
					newNode.fillHolesRandomly() # fill in params, add the node to the randomizeable nodes
					self.availableNodes.append(newNode)
			elif distribType == GaussianDistribNode:
					newNode = GaussianDistribNode(self.varName)
					newNode.setProgram(self.program)
					newNode.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable, self.matchingRowsValues) # fill in params, add the node to the randomizeable nodes
					self.availableNodes.append(newNode)
			elif distribType == UniformRealDistribNode:
					newNode = UniformRealDistribNode(self.varName)
					newNode.setProgram(self.program)
					newNode.fillHolesForConcretePathConditions(dataset, pathCondition, currVariable, self.matchingRowsValues) # fill in params, add the node to the randomizeable nodes
					self.availableNodes.append(newNode)
			else:
				raise Exception("Tried to make a type of real distribution we don't know about.")
		self.actualDistribNode = random.choice(self.availableNodes)

		if len(self.availableNodes) > 1:
			self.program.addRandomizeableNode(self)
			self.randomizeable = True
		elif self in self.program.randomizeableNodes:
			self.program.removeRandomizeableNodes(self)
			self.randomizeable = False

		if debug: print "concrete: real", self.strings()

	def fillHolesRandomly(self):
		if debug: print "randomly: real before", self.strings()
		if len(self.availableNodes) > 0:
			# nothing to fill randomly here, already filled it based on a concrete path condition
			return True
		self.mutate()
		# add this to the set of randomizeable nodes since we can replace the actualDistribNode
		self.program.addRandomizeableNode(self)
		self.randomizeable = True
		if debug: print "randomly: real", self.strings()
		return True

	def mutate(self):
		self.actualDistribNode = random.choice(self.availableNodes)

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
			return ["Gaussian(%f,%f)" % (self.mu, self.sig**2)]
		else:
			return ["Gaussian(",",", ")"]

	def params(self):
		return [("Gaussian", (self.mu, self.sig), self.percentMatchingRows)]

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable, matchingRowsValues):
		if len(matchingRowsValues) < 1:
			# can use anything; there's no dataset data on this, so it doesn't matter
			self.mu = 1
			self.sig = 1
			self.percentMatchingRows = 0
		else:
			self.mu = np.mean(matchingRowsValues)
			self.sig = np.std(matchingRowsValues)
			if abs(self.sig) < .00001:
				self.sig = .00001 # shouldn't be using Gaussian to model constants
			self.percentMatchingRows = len(matchingRowsValues)/self.program.dataset.numRows

		if debug: print "concrete: gaussian", self.strings()

class BetaDistribNode(RealDistribNode):
	def __init__(self, varName, alpha=None, beta=None, percentMatchingRows = None):
		RealDistribNode.__init__(self, varName)
		self.alpha = alpha
		self.beta = beta
		self.percentMatchingRows = percentMatchingRows
		self.randomizeable = False

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
		self.program.addRandomizeableNode(self)
		self.randomizeable = True
		if debug: print "random: beta", self.strings()
		return True

	def getRandomizeableNodes(self):
		ls = []
		if self.randomizeable:
			ls.append(self)
		return ls

	def mutate(self):
		lowerBound = .000000000000001
		upperBound = 40
		if self.alpha == None:
			self.alpha = random.uniform(lowerBound, upperBound) 
			self.beta = random.uniform(lowerBound, upperBound)
		else:
			modParams = overwriteOrModifyOneParam(.3, [self.alpha, self.beta], lowerBound, upperBound, -3, 3)
			self.alpha = modParams[0]
			self.beta = modParams[1]

                        
class GammaDistribNode(RealDistribNode):
	def __init__(self, varName, k=None, l=None, percentMatchingRows = None):
		RealDistribNode.__init__(self, varName)
		self.k = k
		self.l = l
		self.percentMatchingRows = percentMatchingRows
		self.randomizeable = False

	def strings(self, tabs=0):
		if self.k:
			return ["Gamma(%f,%f)" % (self.k, self.l)]
		else:
			return ["Gamma(",",", ")"]

	def params(self):
		return [("Gamma", (self.k, self.l), self.percentMatchingRows)]

	def reduce(self, dataset, pathCondition, currVariable):
		# no reduction to do here
		return

	def fillHolesRandomly(self):
		self.mutate()
		self.program.addRandomizeableNode(self)
		self.randomizeable = True
		if debug: print "random: beta", self.strings()
		return True

	def getRandomizeableNodes(self):
		ls = []
		if self.randomizeable:
			ls.append(self)
		return ls

	def mutate(self):
		lowerBound = .000000000000001
		upperBound = 40
		if self.alpha == None:
			self.k = random.uniform(lowerBound, upperBound) 
			self.l = random.uniform(lowerBound, upperBound)
		else:
			modParams = overwriteOrModifyOneParam(.3, [self.k, self.l], lowerBound, upperBound, -3, 3)
			self.k = modParams[0]
			self.l = modParams[1]


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

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable, matchingRowsValues):
		if len(matchingRowsValues) < 1:
			# can use anything; there's no dataset data on this, so it doesn't matter
			self.a = 0
			self.b = 1
			self.percentMatchingRows = 0
		else:
			self.a = min(matchingRowsValues) - .00001
			self.b = max(matchingRowsValues) + .00001
			self.percentMatchingRows = len(matchingRowsValues)/self.program.dataset.numRows

		if debug: print "concrete: uniform", self.strings()


class IfNode(ASTNode):
	def __init__(self, conditionNodes, bodyNodes):
		ASTNode.__init__(self)
		self.conditionNodes = conditionNodes
		self.bodyNodes = bodyNodes
		self.randomizeable = False

		self.allowableVariables = []
		for node in self.conditionNodes:
			node.setParent(self)
		for node in self.bodyNodes:
			node.setParent(self)

	def setProgram(self, program):
		self.program = program

		self.allowableVariables = self.parent.allowableVariables

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
		#print currVariable.name, len(self.bodyNodes)
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
			pathConditionAdditional = self.pathConditionForConditionNode(i)
			if pathConditionAdditional == None:
				# the path condition is no longer concrete
				continue
			newPathCondition = pathCondition + [pathConditionAdditional]
			self.bodyNodes[i].reduce(dataset, newPathCondition, currVariable)

	def pathConditionForConditionNode(self, i):
		# must not all preceding branches

		# if any of the branches have non-concrete conditions, should return None
		pathConditions = map(lambda x: x.pathCondition(), self.conditionNodes)
		nonConcrete = reduce(lambda x, y: x or y == None, pathConditions, False)
		if nonConcrete: return None

		if i == 0:
			return pathConditions[i]

		conditionsToNot = pathConditions[0:i]

		if i == len(self.conditionNodes):
			newFunc = lambda row: reduce(lambda a, b: a and not b.func(row), conditionsToNot, True)
		else:
			conditionToAdd = pathConditions[i]
			newFunc = lambda row: conditionToAdd.func(row) and reduce(lambda a, b: a and not b.func(row), conditionsToNot, True) 

		return PathConditionComponent(newFunc)

	def pathCondition(self):
		conditionSoFar = []
		child = self
		parent = self.parent
		while not isinstance(parent, VariableDeclNode):
			bodyIndex = parent.bodyNodes.index(child)
			pathConditionAdditional = parent.pathConditionForConditionNode(bodyIndex)
			conditionSoFar = [pathConditionAdditional] + conditionSoFar
			child = parent
			parent = child.parent
		currentVariable = parent
		return conditionSoFar, currentVariable

	def fillHolesForConcretePathConditionsHelper(self):
		pathCondition, currentVariable = self.pathCondition()
		self.fillHolesForConcretePathConditions(self.program.dataset, pathCondition, currentVariable)

	def fillHolesForConcretePathConditions(self, dataset, pathCondition, currVariable):
		if debug: print "concrete: if"
		for i in range(len(self.bodyNodes)):
			pathConditionAdditional = self.pathConditionForConditionNode(i)
			if pathConditionAdditional == None:
				# the path condition is no longer concrete, stop descending
				continue
			newPathCondition = pathCondition + [pathConditionAdditional]
			self.bodyNodes[i].fillHolesForConcretePathConditions(dataset, newPathCondition, currVariable)

	def getRandomizeableNodes(self):
		ls = []
		if self.randomizeable:
			ls.append(self)
		for node in self.bodyNodes:
			ls += node.getRandomizeableNodes()
		for node in self.conditionNodes:
			ls += node.getRandomizeableNodes()
		return ls

	def fillHolesRandomly(self):
		if debug: print "randomly: if"
		filledSomeHoles = False
		for node in self.conditionNodes:
			filledSomeHoles = node.fillHolesRandomly() or filledSomeHoles
			# note that calling fillHolesRandomly on comparison nodes will cause us to call fillHolesForConcrete on this node

		# but there could still be conditions down there that aren't concrete yet, so keep going
		for node in self.bodyNodes:
			filledSomeHoles = node.fillHolesRandomly() or filledSomeHoles 

		if self.conditionNodes[0].randomizeable:
			# can randomize if this if conditions on a real or int
			# doesn't make sense to add or remove branches for conditioning on bools, categoricals
			self.program.addRandomizeableNode(self)
			self.randomizeable = True
		return filledSomeHoles

	def mutate(self):
		# acck!  we better add and remove all the child nodes ot the randomizeable list as necessary

		if len(self.bodyNodes) < 3 or random.uniform(0,1) < .2: # adding an if shouldn't be that likely
			# we may mutate the if statement by adding an additional branch
			newBodyNode = copyNode(self.bodyNodes[0])
			newConditionNode = copyNode(self.conditionNodes[0])
			newConditionNode.mutate() # shouldn't be exactly the same
			self.bodyNodes.insert(0, newBodyNode)
			self.conditionNodes.insert(0, newConditionNode)

			# add in the new nodes that we can randomize
			randomizeableNodesToAdd = newBodyNode.getRandomizeableNodes()
			randomizeableNodesToAdd += newConditionNode.getRandomizeableNodes()
			for node in randomizeableNodesToAdd:
				self.program.addRandomizeableNode(node)

		else:
			# or by removing a branch
			indexToRemove = random.choice(range(len(self.conditionNodes)))

			# kick out the nodes we shouldn't randomize anymore
			randomizeableNodesToRemove = self.bodyNodes[indexToRemove].getRandomizeableNodes()
			randomizeableNodesToRemove += self.conditionNodes[indexToRemove].getRandomizeableNodes()
			for node in randomizeableNodesToRemove:
				self.program.removeRandomizeableNode(node)

			del self.bodyNodes[indexToRemove]
			del self.conditionNodes[indexToRemove]

		# we just changed the conditions, better recalculate
		self.fillHolesForConcretePathConditionsHelper()

def copyNode(node):
	newNode = deepcopy(node)
	newNode.setParent(node.parent)
	newNode.setProgram(node.program)
	return newNode

class VariableUseNode(ASTNode):
	def __init__(self, name, typeName):
		ASTNode.__init__(self)
		self.name = name
		self.typeName = typeName
		self.randomizeable = False

	def setProgram(self, program):
		self.program = program

	def strings(self, tabs=0):
		return [self.name]

	def pathCondition(self):
		index = self.program.dataset.namesToIndexes[self.name]
		return PathConditionComponent(lambda x: x[index] == "true") # x is a list of args

	def pathConditionFalse(self):
		index = self.program.dataset.namesToIndexes[self.name]
		return PathConditionComponent(lambda x: x[index] == "false") # x is a list of args

	def range(self):
		return self.program.variableRange(self.name)

	def lambdaToCalculate(self):
		index = self.program.dataset.namesToIndexes[self.name]
		return lambda row: row[index]

	def getRandomizeableNodes(self):
		return []

class ComparisonNode(ASTNode):

	ops = {	"==": operator.eq,
			">": operator.gt,
			"<": operator.lt}

	def __init__(self, variableNode, relationship = None, value = None):
		ASTNode.__init__(self)
		self.node = variableNode
		self.relationship = relationship
		self.value = value
		self.randomizeable = False
		self.allowableVariables = None
		self.numberVariables = []

	def setProgram(self, program):
		self.program = program
		if self.allowableVariables == None:
			self.allowableVariables = self.parent.allowableVariables
			self.numberVariables = filter(lambda x: (x.typeName == "Real" or x.typeName == "Integer") and x.name != self.node.name, self.allowableVariables)
		self.node.setProgram(program)

	def strings(self, tabs=0):
		if self.relationship:
			strs = [["(" + self.node.name + " " + self.relationship + " "], self.value.strings(),[")"]]
			return combineStrings(strs)
		else:
			return [self.node.name, ""]

	def getRandomizeableNodes(self):
		ls = []
		if self.randomizeable:
			ls.append(self)
		return ls

	def pathCondition(self):
		if self.relationship == None or self.value == None:
			return None
		index = self.program.dataset.namesToIndexes[self.node.name]
		valFunc = self.value.lambdaToCalculate()
		return PathConditionComponent(lambda x: self.ops[self.relationship](x[index], valFunc(x))) # x is a list of args

	def pathConditionFalse(self):
		if self.relationship == None or self.value == None:
			return None
		index = self.program.dataset.namesToIndexes[self.node.name]
		valFunc = self.value.lambdaToCalculate()
		return PathConditionComponent(lambda x: not self.ops[self.relationship](x[index], valFunc(x))) # x is a list of args

	def fillHolesRandomly(self):
		if debug: print "randomly: comparison before", self.strings()
		if self.node.typeName == "Real" or self.node.typeName == "Integer":
			self.program.addRandomizeableNode(self)
			self.randomizeable = True
			self.firstMutate()
			if debug: print "randomly: comparison", self.strings()
			return True
		return False

	def randomizeOperator(self):
		if self.node.typeName == "Real":
			self.relationship = random.choice([">", "<"]) # using eq for reals is just silly
		else:
			self.relationship = random.choice([self.ops.keys()])

	def firstMutate(self):
		(lowerBound, upperBound) = self.node.range()
		if lowerBound > -1:
			lowerBound = -1 # you never know when you might need some low numbers for multiplying and such...
		newNumber = NumberWrapper(self, lowerBound, upperBound)
		self.value = newNumber
		self.randomizeOperator()

		# we've changed the conditions.  better recalculate the things that depend on path conditions
		self.parent.fillHolesForConcretePathConditionsHelper()

	def mutate(self):

		# We can:
		# 1. Randomly fill a numeric slot with a new constant or variable use
		# 2. Slightly adjust a current constant
		# 3. Add an operator
		# 4. Remove an operator
		# 5. Change an operator
		# 6. Change the top level comparison

		#TODO: change numbers
		#TODO: add restrictions like shouldn't have 2+4?

		# refresh our lists:
		RHSNumericSlots = self.value.numericSlots()
		RHSConstants = map(lambda x: x.val, filter(lambda x: isinstance(x.val, NumericValue), RHSNumericSlots))
		RHSOperators = []
		if isinstance(self.value, BinExpNode):
			RHSOperators = self.value.operators()

		if mutationDebug: print "******"
		if mutationDebug: print "before"
		if mutationDebug: print self.strings()

		# sometimes we may go through motions of mutation but not end up with something different
		# (randomly select same RV to use, attempted mutation not allowed)
		# so compare strings at end to make sure something actually changed
		preString = self.strings()

		decision = random.uniform(0,1)
		if decision < .15 and len(RHSNumericSlots) > 0:
			# Randomly fill a numeric slot with a new constant or variable use
			if mutationDebug: print "Randomly fill a numeric slot with a new constant or variable use"
			random.choice(RHSNumericSlots).randomizeVal()
		elif decision < .5 and len(RHSConstants) > 0:
			# Slightly adjust a current constant
			if mutationDebug: print "Slightly adjust a current constant"
			random.choice(RHSConstants).adjustVal(-2, 2)
		elif decision < .6:
			# Add an operator
			if mutationDebug: print "Add an operator"
			(lowerBound, upperBound) = self.node.range()
			if lowerBound > -1:
				lowerBound = -1 # you never know when you might need some low numbers for multiplying and such...
			subExps = [self.value, NumberWrapper(self, lowerBound, upperBound)]
			random.shuffle(subExps)
			newExpression = BinExpNode("+", subExps[0], subExps[1])
			newExpression.randomizeOp()
			RHSOperators.append(newExpression)
			self.value = newExpression
			for expr in subExps:
				expr.setParent(newExpression)
			newExpression.setParent(self)
		elif decision < .8 and len(RHSOperators) > 0:
			# Remove an operator
			if mutationDebug: print "Remove an operator"
			opToRemove = random.choice(RHSOperators)
			opToRemove.removeOp()
		elif decision < .9 and len(RHSOperators) > 0:
			# Change an operator
			if mutationDebug: print "Change an operator"
			random.choice(RHSOperators).randomizeOp()
		else:
			# Change the top level comparison
			if mutationDebug: print "Change the top level comparison"
			self.randomizeOperator()		

		# we don't want to be needlessly combining constants
		if isinstance(self.value, BinExpNode):
			self.value.partiallyEvaluate()

		postString = self.strings()
		if postString == preString:
			# it's the same.  try again
			return self.mutate()

		# ok we've made a mutation that works
		# we've changed the conditions.  better recalculate the things that depend on path conditions
		self.parent.fillHolesForConcretePathConditionsHelper()

		if mutationDebug: print "after"
		if mutationDebug: print self.strings()

class NumberWrapper(ASTNode):
	def __init__(self, comparisonNode, lowerBound, upperBound, val = None):
		ASTNode.__init__(self)
		self.comparisonNode = comparisonNode
		self.lowerBound = lowerBound
		self.upperBound = upperBound
		self.val = val
		self.randomizeVal()

	def randomizeVal(self):
		if isinstance(self.val, VariableUseNode):
			# we've been using an R.V. but we're about to overwrite it, so we should put it back into circulation
			self.comparisonNode.numberVariables.append(self.val)

		if random.uniform(0,1) > .5 or len(self.comparisonNode.numberVariables) < 1:
			self.val = NumericValue(random.uniform(self.lowerBound, self.upperBound))
		else:
			self.val = random.choice(self.comparisonNode.numberVariables)
			self.comparisonNode.numberVariables.remove(self.val) # so an R.V. can only be used once in a comparison expression

	def strings(self):
		return self.val.strings()

	def lambdaToCalculate(self):
		return self.val.lambdaToCalculate()

	def numericSlots(self):
		return [self]

	def partiallyEvaluate(self):
		return

	def getRandomizeableNodes(self):
		return []

class NumericValue(ASTNode):

	def __init__(self, val):
		ASTNode.__init__(self)
		self.val = val

	def strings(self, tabs=0):
		return [str(self.val)]

	def adjustVal(self, lower, upper):
		self.val = self.val + random.uniform(lower, upper)

	def lambdaToCalculate(self):
		return lambda row: self.val

	def getRandomizeableNodes(self):
		return []

class StringValue(ASTNode):

	def __init__(self, val):
		ASTNode.__init__(self)
		self.val = val

	def strings(self, tabs=0):
		return [str(self.val)]

	def lambdaToCalculate(self):
		return lambda row: self.val

	def getRandomizeableNodes(self):
		return []

class BinExpNode(ASTNode):

	ops = {	"+": operator.__add__,
			"-": operator.__sub__,
			"*": operator.__mul__}

	def __init__(self, op, e1, e2):
		ASTNode.__init__(self)
		self.op = op
		self.e1 = e1
		self.e2 = e2

	def setProgram(self, program):
		self.program = program
		self.e1.setProgram(program)
		self.e2.setProgram(program)

	def randomizeOp(self):
		self.op = random.choice(self.ops.keys())

	def strings(self, tabs=0):
		return combineStrings([["("], self.e1.strings(), [" "+self.op+" "], self.e2.strings(), [")"]])

	def lambdaToCalculate(self):
		l1 = self.e1.lambdaToCalculate()
		l2 = self.e2.lambdaToCalculate()
		return lambda row: self.ops[self.op](l1(row), l2(row))

	def numericSlots(self):
		ls = []
		for e in [self.e1, self.e2]:
			ls = ls + e.numericSlots()
		return ls

	def operators(self):
		ls = [self]
		for e in [self.e1, self.e2]:
			if isinstance(e, BinExpNode):
				ls = ls + e.operators()
		return ls

	def removeOp(self, replacement = None):
		if replacement == None:
			useLeft = random.choice([True, False])
			if useLeft:
				replacement = self.e1
			else:
				replacement = self.e2
		parent = self.parent
		replacement.setParent(parent)
		if (isinstance(parent, ComparisonNode)):
			parent.value = replacement
		else:
			# it's another op node
			if parent.e1 == self:
				parent.e1 = replacement
			elif parent.e2 == self:
				parent.e2 = replacement
			else:
				raise Exception("Freak out!  Trying to remove op that's not here...")

	def getRandomizeableNodes(self):
		return []

	def partiallyEvaluate(self):
		# first descend
		n1 = self.e1
		n2 = self.e2
		n1.partiallyEvaluate()
		n2.partiallyEvaluate()
		# nodes may have changed now that we did partial eval
		n1 = self.e1
		n2 = self.e2
		if isinstance(n1, NumberWrapper) and isinstance(n2, NumberWrapper):
			if isinstance(n1.val, NumericValue) and isinstance(n2.val, NumericValue):
				# we can collapse some stuff!
				val1 = n1.val.val
				val2 = n2.val.val
				output = self.ops[self.op](val1, val2)
				newNum = NumberWrapper(n1.comparisonNode, n1.lowerBound, n1.upperBound, NumericValue(output))
				self.removeOp(newNum)
		else:
			if (self.op == "+" or self.op == "*"):
				if isinstance(n1, BinExpNode) and isinstance(n2, BinExpNode):
					# both are bin expnodes.  these are interesting if we're in the case of (+ (+ A 5) (+ B 6))
					if (self.op == n1.op and self.op == n2.op):

						numberWrapperNodes = []
						otherNodes = []
						for node in [n1.e1, n1.e2, n2.e1, n2.e2]:
							if isinstance(node, NumberWrapper) and isinstance(node.val, NumericValue):
								numberWrapperNodes.append(node)
							elif isinstance(node, BinExpNode):
								otherNodes.append(node)
						if len(numberWrapperNodes) > 1:
							# we can collapse some stuff!  the length is 2 because if it was bigger we'd have collapsed n1 or n2 further
							newExpNode = BinExpNode(self.op, otherNodes[0], otherNodes[1])
							for node in otherNodes:
								node.setParent(newExpNode)
							numberWrapperNodes[0].val.val = self.ops[self.op](numberWrapperNodes[0].val.val, numberWrapperNodes[1].val.val)
							newExpNode2 = BinExpNode(self.op, newExpNode, numberWrapperNodes[0]) # let's try to make constants rise to the top so we can combine.  not thorough, but...
							numberWrapperNodes[0].setParent(newExpNode2)
							newExpNode.setParent(newExpNode2)
							self.removeOp(newExpNode2)
					return

				# one of these is a numberwrapper and the other is a binexpnode or we would have gone into one of the other cases
				# the question is whether we have a couple numericvalues
				expNode = None
				numberWrapperNode = None
				if isinstance(n2, NumberWrapper) and isinstance(n2.val, NumericValue) and isinstance(n1, BinExpNode):
					expNode = n1
					numberWrapperNode = n2
				elif isinstance(n1, NumberWrapper) and isinstance(n1.val, NumericValue) and isinstance(n2, BinExpNode):
					expNode = n2
					numberWrapperNode = n1
				if expNode == None:
					return
				if (isinstance(expNode.e1, NumberWrapper) and isinstance(expNode.e1.val, NumericValue)) or (isinstance(expNode.e2, NumberWrapper) and isinstance(expNode.e2.val, NumericValue)):
					# we can collapse some stuff!
					expNodeNumberWrapperNode = expNode.e1
					if (isinstance(expNode.e2, NumberWrapper) and isinstance(expNode.e2.val, NumericValue)):
						expNodeNumberWrapperNode = expNode.e2
					expNodeNumberWrapperNode.val.val = expNodeNumberWrapperNode.val.val + numberWrapperNode.val.val
					self.removeOp(expNode)

class BoolBinExpNode(ASTNode):

	ops = {	"&": operator.__and__,
				"|": operator.__or__}

	def __init__(self, op, e1, e2):
		ASTNode.__init__(self)
		self.op = op
		self.e1 = e1
		self.e2 = e2

        def strings(self, tabs=0):
		return combineStrings([["("], self.e1.strings(), [" "+self.op+" "], self.e2.strings(), [")"]])

	def getRandomizeableNodes(self):
		return []

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
