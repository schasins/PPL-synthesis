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
    ast = BinExpNode('*',GaussianDistribNode(mu=20,sig=0.1), \
                     GaussianDistribNode(mu=30,sig=0.1))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog


def testMoG4():
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
    
def testMoG5():
    ast = GaussianDistribNode(mu=GaussianDistribNode(mu=25,sig=7),sig=GaussianDistribNode(mu=1,sig=1))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG6():
    ast = BoolBinExpNode('||', BooleanDistribNode(percentTrue=0.2), \
                         UnaryExpNode('!', BooleanDistribNode(percentTrue=0.7)))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG7():
    ast = BinExpNode('+',BetaDistribNode(alpha=2, beta=3),100)
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG8():
    x = CategoricalDistribNode(["a","b","c"],{"a":0.2,"b":0.3,"c":0.5})
    tests = [ComparisonNode(BooleanDistribNode(percentTrue=0.2), "==", BooleanDistribNode(percentTrue=0.2)), \
             ComparisonNode(BooleanDistribNode(percentTrue=0.2), "==", True), \
             ComparisonNode(True, "==", True), \
             ComparisonNode(True, "==", False), \
             ComparisonNode(GaussianDistribNode(10,0), "==", GaussianDistribNode(10,0)), \
             ComparisonNode(GaussianDistribNode(10,1), "==", GaussianDistribNode(10,0)), \
             ComparisonNode(GaussianDistribNode(10,0), "==", GaussianDistribNode(11,0)), \
             ComparisonNode(x, "==", x), \
             ComparisonNode(x, "==", "a"), \
             ComparisonNode("b", "==", "a"), \
    ]

    for ast in tests:
        print "e1 =", ast.node
        print "e2 =", ast.value
        estimator = ScoreEstimator(None)
        mog = estimator.visit(ast)
        print mog
        print "-------------------------------------------------------"

testMoG8()
