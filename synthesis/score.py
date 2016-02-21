from astDB import *
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
        if p < 0 and p > -0.000001:
            p = 0
        if p > 1 and p < 1.000001:
            p = 1
            
        if p < 0:
            print "Bernoulli", p
            raise ScoreError("Bernoulli: p < 0")
        if p > 1:
            print "Bernoulli", p
            raise ScoreError("Bernoulli: p > 1")
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
        if not(len(w) == n) or not(len(mu) == n) or not(len(sig) == n):
            print "MoG", n, w, mu, sig
            raise ScoreError("MoG: length mismatch")
        
        neg_w = False
        for wi in w:
            if wi < 0:
                neg_w = True
                break
        if neg_w:
            print "MoG w=", w
            raise ScoreError("MoG: negative weight")
            
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
    mog = MoG(n, np.array(w), np.array(mu), np.array(sig))
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
            return self.visit_UniformRealDistribNode(ast)
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
# VariableUseNode collector
# **********************************************************************

class VarCollector():
    def __init__(self, dataset=None):
        self.vars = set()

    def visit(self, ast):
        if isinstance(ast, VariableUseNode):
            self.vars.add(ast.name)
        elif isinstance(ast, ComparisonNode):
            self.visit(ast.node)
            self.visit(ast.value)
        elif isinstance(ast, BoolBinExpNode) or isinstance(ast, BinExpNode):
            self.visit(ast.e1)
            self.visit(ast.e2)
        elif isinstance(ast, UnaryExpNode):
            self.visit(ast.e)
        elif isinstance(ast, NumberWrapper):
            return self.visit(ast.val)

    def getVars(self,ast):
        self.vars = set()
        self.visit(ast)
        return self.vars

# **********************************************************************
# Likelihood estimator
# **********************************************************************

def checkPDF(pdf):
    if pdf < 0:
        print "name", name
        print "dist", dist
        print "val", val
        print "pdf = ", pdf
        raise "Negative PDF"
    
def checkProb(p):
    if p < 0 or p > 1:
        print "name", name
        print "dist", dist
        print "val", val
        print "p = ", p
        raise "p < 0 or p > 1"

class ScoreEstimator(visitor):
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.env = {}
        self.varcollector = VarCollector()

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
        n = self.dataset.numRows
        th = 20000 # most benchmarks = 10000
        for col in self.dataset.indexesToNames:
            name = self.dataset.indexesToNames[col]
            dist = self.env[name]
            if isinstance(dist,Bernoulli):
                t = self.dataset.SQLCountCond(name + "=true")
                f = n - t
                pt = dist.at(1)
                pf = 1 - pt
                checkProb(pt)
                
                if t > 0:
                    if pt == 0:
                        return -maxint
                    loglik = loglik + log(pt)*t

                if f > 0:
                    if pf == 0:
                        return -maxint
                    loglik = loglik + log(pf)*f
                
            elif isinstance(dist,Categorical):
                for val in dist.values:
                    count = self.dataset.SQLCountCond(name + "='" + val + "'")
                    pdf = dist.at(val)
                    checkProb(pdf)
                    if count > 0:
                        if pdf == 0:
                            return -maxint
                        loglik = loglik + log(pdf)*count
            elif n < th:
                for val in self.dataset.columns[col]:
                    pdf = dist.at(val)
                    checkPDF(pdf)
                    if pdf == 0:
                        return -maxint
                    log_pdf = log(pdf)
                    loglik = loglik + log_pdf
            else:
                vmin = self.dataset.SQLFind("MIN",name)
                vmax = self.dataset.SQLFind("MAX",name)
                size = 1.0*(vmax - vmin)/th
                #print "min-max", vmin, vmax, size
                prev = 0
                
                # Exclude the last interval because upper might not cover everything because of precision issue.
                for i in xrange(th-1):
                    mid = vmin + (i + 0.5)*size
                    upper = vmin + (i + 1)*size
                    count = self.dataset.SQLCountCond(name + "<=" + str(upper))
                    pdf = dist.at(mid)
                    checkPDF(pdf)
                    if count - prev > 0:
                        if pdf == 0:
                            return -maxint
                        loglik = loglik + log(pdf)*(count - prev)
                    #print i, count, mid, pdf
                    prev = count

                # Last interval is handled here.
                mid = vmax - 0.5*size
                pdf = dist.at(mid)
                checkPDF(pdf)
                if n - prev > 0:
                    if pdf == 0:
                        return -maxint
                    loglik = loglik + log(pdf)*(n - prev)
                #print "last", n, mid, pdf
                    
        return loglik

    def visit_ASTNode(self, ast):
        for c in ast.children:
            self.visit(c)

    def visit_VariableDeclNode(self, ast):
        self.env[ast.name] = self.visit(ast.RHS)

    def visit_Constant(self, ast):
        return MoG(1,np.array([1]),np.array([ast.val]),np.array([0.0]))

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

    def ite_list(self, conds, bodies):
        (conds, bodies) = self.uniqueConditions(conds, bodies)
        if len(conds) == len(bodies):
            conds = conds[:-1]
            
        conds = [self.visit(x) for x in conds]
        bodies = [self.visit(x) for x in bodies]
        ps = [x.p for x in conds]
        ps.append(1-sum(ps))
        
        if isinstance(bodies[0],Bernoulli):
            ans = sum([p*body.p for (p,body) in zip(ps,bodies)])
            return Bernoulli(ans)
        elif isinstance(bodies[0], MoG):
            all_n = 0
            all_w = np.array([])
            all_mu = np.array([])
            all_sig = np.array([])
            for (p,body) in zip(ps,bodies):
                all_n = all_n + body.n
                all_w = np.concatenate((all_w,body.w * p), axis=0)
                all_mu = np.concatenate((all_mu,body.mu), axis=0)
                all_sig = np.concatenate((all_sig,body.sig), axis=0)
            return MoG(all_n, all_w, all_mu, all_sig)
        elif isinstance(bodies[0], Categorical):
            values = sorted(bodies[0].values)
            newmap = {}
            for val in values:
                newmap[val] = 0
            for (p,body) in zip(ps,bodies):
                if not(sorted(body.values) == values):
                    raise ScoreError("if-else: categorical distributions of branches mismatch.")
                oldmap = body.valuesToPercentages
                for val in values:
                    newmap[val] = newmap[val] + p * oldmap[val]
            return Categorical(values,newmap)
        else:
            raise ScoreError("if-else: true and false branches' types mistmatch")

    def visit_IfNode(self, ast):
        
        conditions = ast.conditionNodes
        bodies = ast.bodyNodes
        
        if self.compareCategorical(conditions):
            return self.ite_list(conditions,bodies)
        
        if len(conditions) == len(bodies):
            conditions = conditions[:-1]
            
        conditions = [self.visit(x) for x in conditions]
        bodies = [self.visit(x) for x in bodies]
            

        # if len(conditions) > 1 and self.dependent(ast.conditionNodes):
        #     print "IfNode ADJUSTMENT"
        #     print ast.strings()[0]
        #     print "conditions"
        #     for x in conditions:
        #         print x
        #     print "bodies"
        #     for x in bodies:
        #         print x
        #     not_p = 1 - conditions[0].p
        #     for b in conditions[1:]:
        #         b.p = b.p/not_p
        #         not_p = not_p * (1 - b.p)


        working = bodies[-1]
        # traverse in reversed order
        for i in range(len(conditions))[::-1]:
            working = self.ite(conditions[i],bodies[i],working)

        return working

    def compareCategorical(self, exprs):
        name = None
        for expr in exprs:
            if isinstance(expr, ComparisonNode) and \
               expr.relationship == "==":
                if isinstance(expr.value, VariableUseNode) and \
                   not(isinstance(expr.node, VariableUseNode)):
                    tmp = expr.value
                    expr.value = expr.node
                    expr.node = tmp

                if isinstance(expr.node, VariableUseNode) and \
                   isinstance(expr.value, StringValue) and \
                   (name == None or expr.node.name == name):
                    name = expr.node.name
                else:
                    return False
            else:
                return False
        return True

    def uniqueConditions(self, conds, bodies):
        ret_conds = []
        ret_bodies = []
        vals = []
        for i in range(len(conds)):
            if not(conds[i].value.val in vals):
                vals.append(conds[i].value.val)
                ret_conds.append(conds[i])
                ret_bodies.append(bodies[i])
        if len(bodies) == len(conds) + 1:
            ret_bodies.append(bodies[-1])
        return (ret_conds,ret_bodies)

    def dependent(self, exprs):
        ref = self.varcollector.getVars(exprs[0])
        for i in range(1,len(exprs)):
            vars = self.varcollector.getVars(exprs[i])
            if not(vars == ref):
                return False
        return True

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
            mu_op = lambda m1,m2,s1,s2: mu_times(m1,m2,s1,s2)
            sig_op = lambda x,y: sig_times(x,y)
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
        e1 = self.visit(ast.node)
        e2 = self.visit(ast.value)
        # print "rel = ", ast.relationship
        # print "e1 = ", e1
        # print "e2 = ", e2
        # == for boolean, categorical, real
        if ast.relationship == "==" or ast.relationship == "=":
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
            # print "ast.node = ", ast.node
            # print "ast.value = ", ast.value
            # print "e1", e1
            # print "e2", e2
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

def mu_times(m1,m2,s1,s2):
    if (s1**2 + s2**2) == 0:
        return m1*m2
    else:
        return (m1*(s2**1) + m2*(s1**2))/(s1**2 + s2**2)

def sig_times(x,y):
    if (x**2 + y**2) == 0:
        return 0
    else:
        return ((x**2) * (y**2))/(x**2 + y**2)

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

def getMoG(ast):
    print ast.strings()[0]
    estimator = ScoreEstimator()
    estimator.visit(ast)
    for name in estimator.env:
        print name, "~", estimator.env[name]

def testEstimateScore(ast, dataset):
    estimator = ScoreEstimator(dataset)
    mutator1 = Mutator("low")
    mutator2 = Mutator("high")
    
    ast1 = mutator1.visit(ast)
    ast2 = mutator2.visit(ast)

    print "score = ", estimator.evaluate(ast)
    print "score = ", estimator.evaluate(ast1)
    print "score = ", estimator.evaluate(ast2)


