from score import *
from ast import *
import matplotlib.pyplot as plt
import numpy as np

def testMoG():
    mog = MoG(2, [0.9,0.1], [25,75], [7,2])
    # mog = MoG(1, [1], [50], [7])
    xs = np.linspace(0,100,1000)
    ys = [mog.at(x) for x in xs]

    plt.plot(xs, ys)
    plt.show()

def testMoG2():
    ast = IfNode(BooleanDistribNode(percentTrue=0.9), \
                 GaussianDistribNode(mu=25,sig=7), \
                 GaussianDistribNode(mu=75,sig=2))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog
    
    xs = np.linspace(0,100,1000)
    ys = [mog.at(x) for x in xs]

    plt.plot(xs, ys)
    plt.show()

def testMoG3():
    ast = BinExpNode('+',\
                     IfNode(BooleanDistribNode(percentTrue=0.9), \
                            GaussianDistribNode(mu=25,sig=7), \
                            GaussianDistribNode(mu=75,sig=2)), \
                     IfNode(BooleanDistribNode(percentTrue=0.9), \
                            GaussianDistribNode(mu=25,sig=7), \
                            GaussianDistribNode(mu=75,sig=2)))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG4():
    ast = GaussianDistribNode(mu=GaussianDistribNode(mu=25,sig=7),sig=GaussianDistribNode(mu=1,sig=1))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG5():
    ast = BoolBinExpNode('||', BooleanDistribNode(percentTrue=0.2), \
                         UnaryExpNode('!', BooleanDistribNode(percentTrue=0.7)))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

testMoG5()
