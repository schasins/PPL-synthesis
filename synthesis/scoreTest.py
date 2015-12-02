from score import *
from ast import *
import matplotlib.pyplot as plt
import numpy as np

B02 = BooleanDistribNode(None,percentTrue=0.2)
B09 = BooleanDistribNode(None,percentTrue=0.9)

def G(mu=None,sig=None):
    return GaussianDistribNode(None,mu=mu,sig=sig)

def testMoG():
    mog = MoG(2, [0.9,0.1], [25,75], [7,2])
    # mog = MoG(1, [1], [50], [7])
    xs = np.linspace(0,100,1000)
    ys = [mog.at(x) for x in xs]

    plt.plot(xs, ys)
    plt.show()

def testMoG2():
    ast = IfNode([B09,B09], \
                 [G(mu=25,sig=7), G(mu=50,sig=2), G(mu=75,sig=2)])
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog
    
    xs = np.linspace(0,100,1000)
    ys = [mog.at(x) for x in xs]

    plt.plot(xs, ys)
    plt.show()

def testMoG3():
    ast = BinExpNode('*',G(mu=20,sig=0.1), G(mu=30,sig=0.1))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog


def testMoG4():
    ast = BinExpNode('+',\
                     IfNode([B09], [G(mu=25,sig=7), G(mu=75,sig=2)]), \
                     IfNode([B09], [G(mu=25,sig=7), G(mu=75,sig=2)]))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog
    
def testMoG5():
    ast = G(mu=G(mu=25,sig=7),sig=G(mu=1,sig=1))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG6():
    ast = BoolBinExpNode('||', B02, \
                         UnaryExpNode('!', B09))
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG7():
    ast = BinExpNode('+',BetaDistribNode(alpha=2, beta=3),100)
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG8():
    x = CategoricalDistribNode(None,["a","b","c"],{"a":0.2,"b":0.3,"c":0.5})
    tests = [#ComparisonNode(B02, "==", B02), \
             #ComparisonNode(B02, "==", True), \
             #ComparisonNode(True, "==", True), \
             #ComparisonNode(True, "==", False), \
             #ComparisonNode(G(10,0), "==", G(10,0)), \
             #ComparisonNode(G(10,1), "==", G(10,0)), \
             #ComparisonNode(G(10,0), "==", G(11,0)), \
             ComparisonNode(x, "==", x), \
             ComparisonNode(x, "==", "a"), \
             ComparisonNode("b", "==", "a"), \
             ComparisonNode(G(-0.5,1), ">", G(0,1)), \
    ]

    for ast in tests:
        print "e1 =", ast.node
        print "e2 =", ast.value
        estimator = ScoreEstimator(None)
        mog = estimator.visit(ast)
        print mog
        print "-------------------------------------------------------"

testMoG8()
