from ast import *
from math import log, sqrt
from random import random
import math
import numpy as np
import scipy.special
from sys import maxint

# **********************************************************************
# Distributions
# **********************************************************************

def guass(x, mu, sig):
    return 1/(sqrt(2*math.pi)*sig) * math.exp(-0.5*(float(x-mu)/sig)**2)
    
class Bernoulli:
    def __init__(self, p):
        self.p = 1.0 * p

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
        self.mu = 1.0 * mu   # list of means
        self.sig = 1.0 * sig # list of variances

    def __str__(self):
        return "MoG(" + str(self.n) + "," + str(self.w) + "," + str(self.mu) + "," + str(self.sig) + ")"

    def at(self, x):
        ans = 0
        for i in xrange(self.n):
            ans = ans + (self.w[i] * guass(x, self.mu[i], self.sig[i]))
        return ans

class Categorical:
    def __init__(self, values, valuesToPercentages):
        self.values = values
        self.valuesToPercentages = valuesToPercentages

    def __str__(self):
        l = []
        for val in self.values:
            l.append(val + "->" + str(self.valuesToPercentages[val]))
        return "Categorical(" + ",".join(l) + ")"

    def at(self, x):
        return self.valuesToPercentages[x]

def uniform(a,b):
    L = b - a
    n = 32
    w = [1.0/n for i in range(n)]
    mu = [(i+0.5)*L/n+a for i in range(n)]
    sig = [1.0*L/n for i in range(n)]
    mog = MoG(n, w, np.array(mu), np.array(sig))
    # print mog
    # xs = np.linspace(a-.2*L,b+.2*L,10000)
    # ys = [mog.at(x) for x in xs]
    # #plt.axis((-4,4,0,2))
    # plt.plot(xs, ys)
    # plt.show()
    return mog

# **********************************************************************
# AST visitor
# **********************************************************************

class ScoreError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class visitor:
    def visit(self, ast):
        if isinstance(ast, VariableDeclNode):
            return self.visit_VariableDeclNode(ast)
        elif isinstance(ast, BooleanDistribNode):
            return self.visit_BooleanDistribNode(ast)
        elif isinstance(ast,CategoricalDistribNode):
            return self.visit_CategoricalDistribNode(ast)
        elif isinstance(ast, GaussianDistribNode):
            return self.visit_GaussianDistribNode(ast)
        elif isinstance(ast,BetaDistribNode):
            return self.visit_BetaDistribNode(ast)
        elif isinstance(ast,GammaDistribNode):
            return self.visit_GammaDistribNode(ast)
        elif isinstance(ast,UniformRealDistribNode):
            raise self.visit_UniformRealDistribNode(ast)
        elif isinstance(ast,RealDistribNode):
            return self.visit_RealDistribNode(ast)
        
        elif isinstance(ast, VariableUseNode):
            return self.visit_VariableUseNode(ast)
        elif isinstance(ast, IfNode):
            return self.visit_IfNode(ast)
        elif isinstance(ast, ComparisonNode):
            return self.visit_ComparisonNode(ast)
        elif isinstance(ast, BoolBinExpNode):
            return self.visit_BoolBinExpNode(ast)
        elif isinstance(ast, BinExpNode):
            return self.visit_BinExpNode(ast)
        elif isinstance(ast, UnaryExpNode):
            return self.visit_UnaryExpNode(ast)
        elif isinstance(ast, bool):
            return self.visit_BoolConstant(ast)
        elif isinstance(ast, NumericValue):
            return self.visit_Constant(ast)
        elif isinstance(ast, NumberWrapper):
            return self.visit(ast.val)
        elif isinstance(ast, StringValue):
            return self.visit_String(ast)
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
        # print "-------- AST---------"
        # print ast.strings()[0]
        self.reset()
        self.visit(ast)
        # print "-------- ENV --------"
        # print self.env
        loglik = 0
        for col in self.dataset.indexesToNames:
            name = self.dataset.indexesToNames[col]
            dist = self.env[name]
            for val in self.dataset.columns[col]:
                pdf = dist.at(val)
                if pdf < 0:
                    print "name", name
                    print "dist", dist
                    print "val", val
                    print "pdf = ", pdf

                if pdf == 0:
                    return -maxint
                log_pdf = log(pdf)
                loglik = loglik + log_pdf
        return loglik

    def visit_ASTNode(self, ast):
        for c in ast.children:
            self.visit(c)

    def visit_VariableDeclNode(self, ast):
        rhs = self.visit(ast.RHS)
        self.env[ast.name] = self.visit(ast.RHS)

    def visit_Constant(self, ast):
        return MoG(1,np.array([1]),np.array([1.0 * ast.val]),np.array([0.0]))

    def visit_BoolConstant(self, ast):
        if ast:
            return Bernoulli(1)
        else:
            return Bernoulli(0)

    def visit_String(self,ast):
        return Categorical([ast.val],{ast.val:1.0})

    def visit_BooleanDistribNode(self, ast):
        return Bernoulli(ast.percentTrue)

    def visit_CategoricalDistribNode(self, ast):
        return Categorical(ast.values, ast.valuesToPercentages)

    def visit_RealDistribNode(self, ast):
        return self.visit(ast.actualDistribNode)

    def visit_BetaDistribNode(self, ast):
        alpha = ast.alpha
        beta = ast.beta
        return MoG(1,np.array([1]), \
                   np.array([(1.0*alpha)/(alpha + beta)]), \
                   np.array([math.sqrt(1.0*alpha*beta / ((alpha+beta)**2 * (alpha+beta+1)))]))

    def visit_GammaDistribNode(self, ast):
        k = ast.k
        l = ast.l
        return MoG(1,np.array([1]), \
                   np.array([k*l]), \
                   np.array([math.sqrt(k*l)]))

    def visit_GaussianDistribNode(self, ast):
        if (isinstance(ast.mu,int) or isinstance(ast.mu,float)) and \
           (isinstance(ast.sig,int) or isinstance(ast.sig,float)):
            return MoG(1,np.array([1]),np.array([1.0*ast.mu]),np.array([1.0*ast.sig]))
        else:
            x1 = self.visit(ast.mu)
            x2 = self.visit(ast.sig)
            if not(x1.n == x2.n):
                raise ScoreError("ScoreEstimator: GaussianDistribNode: mu and sigma need to have the same number of Gaussian components.")
            
            my_sig = []
            for i in xrange(x1.n):
                my_sig.append(x2.mu[i] + x1.sig[i]**2 + x2.sig[i])
            return MoG(x1.n, x1.w, x1.mu, np.array(my_sig))
        
    def visit_UniformRealDistribNode(self,ast):
        return uniform(ast.a,ast.b)
    
    def visit_VariableUseNode(self, ast):
        return self.env[ast.name]

    def ite(self,cond, true, false):
        # cond = self.visit(ast.conditionNode)
        # true = self.visit(ast.thenNode)
        # false = self.visit(ast.elseNode)
        if isinstance(cond, Bernoulli):
            if isinstance(true, Bernoulli) and isinstance(false, Bernoulli):
                return Bernoulli(cond.p * true.p + (1 - cond.p) * false.p)
            elif isinstance(true, MoG) and isinstance(false, MoG):
                return MoG(true.n + false.n, \
                           np.concatenate((cond.p * true.w, (1-cond.p) * false.w), axis=0), \
                           np.concatenate((true.mu, false.mu), axis=0), \
                           np.concatenate((true.sig, false.sig), axis=0))
            elif isinstance(true, Categorical) and isinstance(false, Categorical):
                if not(sorted(true.values) == sorted(false.values)):
                    raise ScoreError("if-else: categorical of the two branches contain differnet values")
                tmap = true.valuesToPercentages
                fmap = false.valuesToPercentages
                p = cond.p
                newmap = {}
                for val in true.values:
                    newmap[val] = p * tmap[val] + (1-p) * fmap[val]
                return Categorical(true.values,newmap)
            else:
                raise ScoreError("if-else: true and false branches' types mistmatch")
        else:
            raise ScoreError("if-else: condition type should be boolean distribution")

    def visit_IfNode(self, ast):
        conditions = [self.visit(x) for x in ast.conditionNodes]
        bodies = [self.visit(x) for x in ast.bodyNodes]

        working = bodies[-1]
        # traverse in reversed order
        for i in range(len(conditions))[::-1]:
            working = self.ite(conditions[i],bodies[i],working)

        return working

    def visit_BoolBinExpNode(self, ast):
        x1 = self.visit(ast.e1)
        x2 = self.visit(ast.e2)
        if ast.op == '&&' or ast.op == '&':
            q = x1.p * x2.p
            return Bernoulli(q)
        elif ast.op == '||' or ast.op == '|':
            q = x1.p + x2.p - (x1.p*x2.p)
            return Bernoulli(q)
        else:
            raise ScoreError("ScoreEstimator: BoolBinExpNode: do not support " + ast.op)

    def visit_BinExpNode(self, ast):
        mog1 = self.visit(ast.e1)
        mog2 = self.visit(ast.e2)
        if ast.op == '+':
            mu_op = lambda x,y,s1,s2: x+y
            sig_op = lambda x,y: math.sqrt(x**2 + y**2) # + 2*alpha*x*y
        elif ast.op == '-':
            mu_op = lambda x,y,s1,s2: x-y
            sig_op = lambda x,y: math.sqrt(x**2 + y**2) # + 2*alpha*x*y
        elif ast.op == '*':
            mu_op = lambda m1,m2,s1,s2: (m1*(s2**1) + m2*(s1**2))/(s1**2 + s2**2)
            sig_op = lambda x,y: ((x**2) * (y**2))/(x**2 + y**2)
        else:
            raise ScoreError("ScoreEstimator: BinExpNode: do not support " + ast.op)

        w = []
        mu = []
        sig = []
        for i in xrange(mog1.n):
            for j in xrange(mog2.n):
                w.append(mog1.w[i] * mog2.w[j])
                mu.append(mu_op(mog1.mu[i], mog2.mu[j],mog1.sig[i], mog2.sig[j]))
                sig.append(sig_op(mog1.sig[i], mog2.sig[j]))

        return MoG(mog1.n*mog2.n, np.array(w), np.array(mu), np.array(sig))

    def visit_UnaryExpNode(self, ast):
        e = self.visit(ast.e)
        if isinstance(e,Bernoulli) and ast.op == '!':
            return Bernoulli(1 - e.p)
        else:
            raise ScoreError("ScoreEstimator: UnaryExpNode: currently only support '!' with bernoulli variable")

    def visit_ComparisonNode(self, ast):
        # print "ast.node = ", ast.node
        # print "ast.value = ", ast.value
        e1 = self.visit(ast.node)
        e2 = self.visit(ast.value)
        # print "e1 = ", e1
        # print "e2 = ", e2
        # == for boolean, categorical, real
        if ast.relationship == "==":
            if isinstance(e1,Bernoulli) and isinstance(e2,Bernoulli):
                return Bernoulli((e1.p*e2.p) + (1-e1.p)*(1-e2.p))
            elif isinstance(e1,MoG) and isinstance(e2,MoG):
                p = 0
                for i in xrange(e1.n):
                    for j in xrange(e2.n):
                        if e1.mu[i] == e2.mu[j] and \
                           e1.sig[i] == 0 and e2.sig[j] == 0:
                            p = p + e1.w[i]*e2.w[j]
                return Bernoulli(p)
            elif isinstance(e1,Categorical) and isinstance(e2,Categorical):
                p = 0
                for i in e1.values:
                    for j in e2.values:
                        if i == j:
                            p = p + e1.valuesToPercentages[i]*e2.valuesToPercentages[j]
                return Bernoulli(p)
            else:
                raise ScoreError("ComparisonNode: types mismatch #1")
        # >, < for real
        elif isinstance(e1,MoG) and isinstance(e2,MoG):
            if ast.relationship == "<":
                # swap
                tmp = e1
                e1 = e2
                e2 = tmp
            p = 0
            for i in range(e1.n):
                for j in range(e2.n):
                    p = p + (1 + erf(e1.mu[i],e2.mu[j],e1.sig[i],e2.sig[j])) \
                        * 0.5 * e1.w[i] * e2.w[j]
            return Bernoulli(p)
        else:
            raise ScoreError("ComparisonNode: types mismatch #2")

def erf(mu1,mu2,sig1,sig2):
    d = math.sqrt(2*(sig1**2 + sig2**2))
    if d == 0:
        return 1
    x = (mu1 - mu2)/d
    return scipy.special.erf(x)


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
        return self.val

    def visit_BoolConstant(self, ast):
        return ast

    def visit_String(self, ast):
        return ast.val

    def visit_BooleanDistribNode(self, ast):
        if self.level == "low":
            p = ast.percentTrue + (random()-0.5)/5
            if p < 0:
                p = 0.000001
            return BooleanDistribNode(p)
        else:
            return BooleanDistribNode(random())

    def visit_CategoricalDistribNode(self, ast):
        return ast
        
    def visit_GaussianDistribNode(self, ast):
        if self.level == "low":
            return GaussianDistribNode(change(ast.mu,0.2), change(ast.sig,0.2))
        else:
            return GaussianDistribNode(change(ast.mu,2), change(ast.sig,2))

    def visit_BetaDistribNode(self, ast):
        return ast

    def visit_GammaDistribNode(self, ast):
        return ast

    def visit_UniformRealDistribNode(self,ast):
        return ast

    def visit_RealDistribNode(self, ast):
        return ast
    
    def visit_VariableUseNode(self, ast):
        return VariableUseNode(ast.name)

    def visit_IfNode(self, ast):
        conditions = [self.visit(x) for x in ast.conditionNodes]
        bodies = [self.visit(x) for x in ast.bodyNodes]
        return IfNode(conditions, bodies)
    
    def visit_ComparisonNode(self, ast):
        return ComparisonNode(ast.node, ast.relationship, ast.value)
    
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


