from ast import *
from math import log
from random import random

# **********************************************************************
# Distributions
# **********************************************************************

class Bernoulli:
    def __init__(self, p):
        self.p = p

    def __str__(self):
        return "Bernoulli(%.6f)" % self.p

    def prob(self, data):
        if data == 1:
            return self.p
        else:
            return 1 - self.p

class MoG:
    def __init__(self, n, w, mean, var):
        self.n = n
        self.w = w        # list of mixing fractions
        self.mean = mean  # list of means
        self.var = var    # list of variances

# **********************************************************************
# AST visitor
# **********************************************************************

class visitor:
    def visit(self, ast):
        if isinstance(ast, VariableDeclNode):
            return self.visit_VariableDeclNode(ast)
        elif isinstance(ast, BooleanDistribNode):
            return self.visit_BooleanDistribNode(ast)
        elif isinstance(ast, VariableUseNode):
            return self.visit_VariableUseNode(ast)
        elif isinstance(ast, IfNode):
            return self.visit_IfNode(ast)
        else:
            return self.visit_ASTNode(ast)

# **********************************************************************
# Likelihood estimator
# **********************************************************************

class ScoreEstimator(visitor):
    def __init__(self, dataset):
        self.env = {}
        self.dataset = dataset

    def print_env(self):
        for v in self.env:
            print v, "=", self.env[v]

    def score(self):
        loglik = 0
        for col in self.dataset.indexesToNames:
            name = self.dataset.indexesToNames[col]
            dist = self.env[name]
            for val in self.dataset.columns[col]:
                loglik = loglik + log(dist.prob(val))
        return loglik

    def visit_ASTNode(self, ast):
        for c in ast.children:
            self.visit(c)

    def visit_VariableDeclNode(self, ast):
        self.env[ast.name] = self.visit(ast.RHS)

    def visit_BooleanDistribNode(self, ast):
        return Bernoulli(ast.percentTrue)

    def visit_VariableUseNode(self, ast):
        return self.env[ast.name]

    def visit_IfNode(self, ast):
        cond = self.visit(ast.conditionNode)
        true = self.visit(ast.thenNode)
        false = self.visit(ast.elseNode)
        if isinstance(cond, Bernoulli) and \
           isinstance(true, Bernoulli) and isinstance(false, Bernoulli):
            return Bernoulli(cond.p * true.p + (1 - cond.p) * false.p)

# **********************************************************************
# AST Mutator
# **********************************************************************

class Mutator(visitor):
    def __init__(self, level):
        self.level = level

    def visit_ASTNode(self, ast):
        new = ASTNode()
        new.children = [self.visit(c) for c in ast.children]
        return new

    def visit_VariableDeclNode(self, ast):
        return VariableDeclNode(ast.name, ast.varType, self.visit(ast.RHS))

    def visit_BooleanDistribNode(self, ast):
        if self.level == "low":
            return BooleanDistribNode(ast.percentTrue + (random()-0.5)/5)
        else:
            return BooleanDistribNode(random())

    def visit_VariableUseNode(self, ast):
        return VariableUseNode(ast.name)

    def visit_IfNode(self, ast):
        return IfNode(self.visit(ast.conditionNode), self.visit(ast.thenNode), self.visit(ast.elseNode))
        

def estimateScore(ast,dataset):
    estimator = ScoreEstimator(dataset)
    estimator1 = ScoreEstimator(dataset)
    estimator2 = ScoreEstimator(dataset)
    mutator1 = Mutator("low")
    mutator2 = Mutator("high")
    
    ast1 = mutator1.visit(ast)
    ast2 = mutator2.visit(ast)
    
    ast.accept(estimator)
    # estimator.print_env()
    ast1.accept(estimator1)
    # estimator1.print_env()
    ast2.accept(estimator2)
    # estimator2.print_env()
    
    print "score = ", estimator.score()
    print "score = ", estimator1.score()
    print "score = ", estimator2.score()
    # print dataset.numColumns
    # print dataset.columns[0][0]
    print "done"
