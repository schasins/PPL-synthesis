from ast import *
from math import log, sqrt
from random import random
import math
import numpy as np

# **********************************************************************
# Distributions
# **********************************************************************

def guass(x, mu, sig):
    return 1/(sqrt(2*math.pi)*sig) * math.exp(-0.5*(float(x-mu)/sig)**2)
    
class Bernoulli:
    def __init__(self, p):
        self.p = p

    def __str__(self):
        return "Bernoulli(%.6f)" % self.p

    def at(self, data):
        # print "p = ", self.p
        if data == 1:
            return self.p
        else:
            return 1 - self.p

class MoG:
    def __init__(self, n, w, mu, sig):
        self.n = n
        self.w = w     # list of mixing fractions
        self.mu = mu   # list of means
        self.sig = sig # list of variances

    def __str__(self):
        return "MoG(" + str(self.n) + "," + str(self.w) + "," + str(self.mu) + "," + str(self.sig) + ")"

    def at(self, x):
        ans = 0
        for i in xrange(self.n):
            ans = ans + (self.w[i] * guass(x, self.mu[i], self.sig[i]))
        return ans

# **********************************************************************
# AST visitor
# **********************************************************************

class visitor:
    def visit(self, ast):
        if isinstance(ast, VariableDeclNode):
            return self.visit_VariableDeclNode(ast)
        elif isinstance(ast, BooleanDistribNode):
            return self.visit_BooleanDistribNode(ast)
        elif isinstance(ast, GaussianDistribNode):
            return self.visit_GaussianDistribNode(ast)
        elif isinstance(ast, VariableUseNode):
            return self.visit_VariableUseNode(ast)
        elif isinstance(ast, IfNode):
            return self.visit_IfNode(ast)
        elif isinstance(ast, BoolBinExpNode):
            return self.visit_BoolBinExpNode(ast)
        elif isinstance(ast, BinExpNode):
            return self.visit_BinExpNode(ast)
        elif isinstance(ast, UnaryExpNode):
            return self.visit_UnaryExpNode(ast)
        elif isinstance(ast, int) or isinstance(ast, float):
            return self.visit_Constant(ast)
        else:
            return self.visit_ASTNode(ast)

# **********************************************************************
# Likelihood estimator
# **********************************************************************

class ScoreEstimator(visitor):
    def __init__(self, dataset):
        self.dataset = dataset

    def reset(self):
        self.env = {}

    def print_env(self):
        for v in self.env:
            print v, "=", self.env[v]

    def evaluate(self, ast):
        self.reset()
        self.visit(ast)
        loglik = 0
        for col in self.dataset.indexesToNames:
            name = self.dataset.indexesToNames[col]
            dist = self.env[name]
            for val in self.dataset.columns[col]:
                # print "val = ", dist.at(val)
                # print "log(val) = ", log(dist.at(val))
                loglik = loglik + log(dist.at(val))
        return loglik

    def visit_ASTNode(self, ast):
        for c in ast.children:
            self.visit(c)

    def visit_VariableDeclNode(self, ast):
        self.env[ast.name] = self.visit(ast.RHS)

    def visit_Constant(self, ast):
        return MoG(1,np.array([1]),np.array([ast]),np.array([0]))

    def visit_BooleanDistribNode(self, ast):
        return Bernoulli(ast.percentTrue)

    def visit_GaussianDistribNode(self, ast):
        if (isinstance(ast.mu,int) or isinstance(ast.mu,float)) and \
           (isinstance(ast.sig,int) or isinstance(ast.sig,float)):
            return MoG(1,np.array([1]),np.array([ast.mu]),np.array([ast.sig]))
        else:
            x1 = self.visit(ast.mu)
            x2 = self.visit(ast.sig)
            if not(x1.n == x2.n):
                raise "ScoreEstimator: GaussianDistribNode: mu and sigma need to have the same number of Gaussian components."
            
            my_sig = []
            for i in xrange(x1.n):
                my_sig.append(x2.mu[i] + x1.sig[i]**2 + x2.sig[i])
            return MoG(x1.n, x1.w, x1.mu, np.array(my_sig))

    def visit_VariableUseNode(self, ast):
        return self.env[ast.name]

    def visit_IfNode(self, ast):
        cond = self.visit(ast.conditionNode)
        true = self.visit(ast.thenNode)
        false = self.visit(ast.elseNode)
        if isinstance(cond, Bernoulli) and \
           isinstance(true, Bernoulli) and isinstance(false, Bernoulli):
            return Bernoulli(cond.p * true.p + (1 - cond.p) * false.p)
        elif isinstance(cond, Bernoulli) and \
           isinstance(true, MoG) and isinstance(false, MoG):
            return MoG(true.n + false.n, \
                       np.concatenate((cond.p * true.w, (1-cond.p) * false.w), axis=0), \
                       np.concatenate((true.mu, false.mu), axis=0), \
                       np.concatenate((true.sig, false.sig), axis=0))

    def visit_BoolBinExpNode(self, ast):
        x1 = self.visit(ast.e1)
        x2 = self.visit(ast.e2)
        if ast.op == '&&':
            q = x1.p * x2.p
        elif ast.op == '||':
            q = x1.p + x2.p - (x1.p*x2.p)
        return Bernoulli(q)

    def visit_BinExpNode(self, ast):
        mog1 = self.visit(ast.e1)
        mog2 = self.visit(ast.e2)
        if ast.op == '+':
            mu_op = lambda x,y: x+y
            sig_op = lambda x,y: math.sqrt(x**2 + y**2) # + 2*alpha*x*y
        elif ast.op == '-':
            mu_op = lambda x,y: x-y
            sig_op = lambda x,y: math.sqrt(x**2 + y**2) # + 2*alpha*x*y
        elif ast.op == '*':
            mu_op = lambda x,y: x*y
            sig_op = lambda x,y: x*y # + 2*alpha*x*y
        else:
            raise "ScoreEstimator: BinExpNode: do not support" + ast.op

        w = []
        mu = []
        sig = []
        for i in xrange(mog1.n):
            for j in xrange(mog2.n):
                w.append(mog1.w[i] * mog2.w[j])
                mu.append(mu_op(mog1.mu[i], mog2.mu[j]))
                sig.append(sig_op(mog1.sig[i], mog2.sig[j]))

        return MoG(mog1.n*mog2.n, np.array(w), np.array(mu), np.array(sig))

    def visit_UnaryExpNode(self, ast):
        e = self.visit(ast.e)
        if isinstance(e,Bernoulli) and ast.op == '!':
            return Bernoulli(1 - e.p)
        else:
            raise "ScoreEstimator: UnaryExpNode: currently only support '!' with bernoulli variable"


# **********************************************************************
# AST Mutator
# **********************************************************************

def change(n,f):
    new = (random()-0.5)*f*n
    if n >= 0:
        if new >= 0:
            return new
        else:
            return 0.000001
    else:
        if new < 0:
            return new
        else:
            return -0.000001

class Mutator(visitor):
    def __init__(self, level):
        self.level = level

    def visit_ASTNode(self, ast):
        new = ASTNode()
        new.children = [self.visit(c) for c in ast.children]
        return new

    def visit_VariableDeclNode(self, ast):
        return VariableDeclNode(ast.name, ast.varType, self.visit(ast.RHS))

    def visit_Constant(self, ast):
        return ast

    def visit_BooleanDistribNode(self, ast):
        if self.level == "low":
            p = ast.percentTrue + (random()-0.5)/5
            if p < 0:
                p = 0.000001
            return BooleanDistribNode(p)
        else:
            return BooleanDistribNode(random())
        
    def visit_GaussianDistribNode(self, ast):
        if self.level == "low":
            return GaussianDistribNode(change(ast.mu,0.2), change(ast.sig,0.2))
        else:
            return GaussianDistribNode(change(ast.mu,2), change(ast.sig,2))

    def visit_VariableUseNode(self, ast):
        return VariableUseNode(ast.name)

    def visit_IfNode(self, ast):
        return IfNode(self.visit(ast.conditionNode), self.visit(ast.thenNode), self.visit(ast.elseNode))
    
    def visit_BoolBinExpNode(self, ast):
        return BoolBinExpNode(ast.op, self.visit(ast.e1), self.visit(ast.e2))
    
    def visit_BinExpNode(self, ast):
        return BinExpNode(ast.op, self.visit(ast.e1), self.visit(ast.e2))
    
    def visit_UnaryExpNode(self, ast):
        return UnaryExpNode(ast.op, self.visit(ast.e))

def estimateScore(ast, dataset):
    estimator = ScoreEstimator(dataset)
    return estimator.evaluate(ast)

def testEstimateScore(ast, dataset):
    estimator = ScoreEstimator(dataset)
    mutator1 = Mutator("low")
    mutator2 = Mutator("high")
    
    ast1 = mutator1.visit(ast)
    ast2 = mutator2.visit(ast)

    print "score = ", estimator.evaluate(ast)
    print "score = ", estimator.evaluate(ast1)
    print "score = ", estimator.evaluate(ast2)


