from score import *
from ast import *
import matplotlib.pyplot as plt
import numpy as np

B02 = BooleanDistribNode(None,percentTrue=0.2)
B09 = BooleanDistribNode(None,percentTrue=0.9)

def G(mu=None,sig=None):
    return GaussianDistribNode(None,mu=mu,sig=sig)

def B(p):
    return BooleanDistribNode(None,percentTrue=p)

def Beta(a,b):
    return BetaDistribNode(None,alpha=a,beta=b)

def Gamma(k,l):
    return GammaDistribNode(None,k=k,l=l)

def U(a,b):
    return UniformRealDistribNode(None,a=a,b=b)

def C(v,m):
    return CategoricalDistribNode(None,v,valuesToPercentages=m)

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
    ast = BoolBinExpNode('|', B02, \
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
             ComparisonNode(G(0.1,0), ">", G(0,0)), \
    ]

    for ast in tests:
        print "e1 =", ast.node
        print "e2 =", ast.value
        estimator = ScoreEstimator(None)
        mog = estimator.visit(ast)
        print mog
        print "-------------------------------------------------------"


def testMoG9():
    ast = IfNode([B09], [C(["cat","dog","fish"],{"cat":.4,"dog":.4,"fish":.2}), \
                         C(["cat","dog","fish"],{"cat":.1,"dog":.4,"fish":.5})])
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG10():
    ast = U(0,1)
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog

def testMoG11():
    ast = Gamma(2,3)
    estimator = ScoreEstimator(None)
    mog = estimator.visit(ast)
    print mog
    
def compare():
    dataset = Dataset("../data-generation/students2.csv")
    ast = ASTNode()
    ast.addChild(VariableDeclNode("skilled","Boolean",B(.2)))
    ast.addChild(VariableDeclNode("tired","Boolean",B(.5)))
    true = IfNode([VariableUseNode("tired","Boolean")],\
                  [G(70,20),G(90,5)])
    false = IfNode([VariableUseNode("tired","Boolean")],\
                  [G(30,15),G(70,5)])
    ast.addChild(VariableDeclNode("testPerformance","Real",\
                                  IfNode([VariableUseNode("skilled","Boolean")],\
                                         [true,false])))
    print ast.strings()[0]
    print "score =", estimateScore(ast,dataset)
    
    ast = ASTNode()
    ast.addChild(VariableDeclNode("skilled","Boolean",B(0.1973)))
    ast.addChild(VariableDeclNode("tired","Boolean",B(0.4897)))
    true = IfNode([VariableUseNode("tired","Boolean")],\
                  [G(65.623311,25.306430),Beta(86.287749,0.879710)])
    false = IfNode([VariableUseNode("tired","Boolean")],\
                  [G(43.713909,33.203993),G(40.054685,83.483936)])
    ast.addChild(VariableDeclNode("testPerformance","Real",\
                                  IfNode([VariableUseNode("skilled","Boolean")],\
                                         [true,false])))
    print ast.strings()[0]
    print "score =", estimateScore(ast,dataset)

def compare2():
    dataset = Dataset("../data-generation/students.csv")
    ast = ASTNode()
    ast.addChild(VariableDeclNode("tired","Boolean",B(.5)))
    ast.addChild(VariableDeclNode("skillLevel","Real",G(10,3)))
    true = IfNode([VariableUseNode("tired","Boolean")],\
                  [G(70,20),G(90,5)])
    false = IfNode([VariableUseNode("tired","Boolean")],\
                  [G(30,15),G(70,5)])
    ast.addChild(VariableDeclNode("testPerformance","Real",\
                                  IfNode([ComparisonNode(VariableUseNode("skillLevel","Real"),">",7.0)],\
                                         [true,false])))
    
    print ast.strings()[0]
    print "score =", estimateScore(ast,dataset)
    
    ast = ASTNode()
    ast.addChild(VariableDeclNode("tired","Boolean",B(.5009)))
    ast.addChild(VariableDeclNode("skillLevel","Real",\
                                  IfNode([VariableUseNode("tired","Boolean")],\
                                         [G(13.576753,14.077953),G(11.365290,6.595882)])))
    true = IfNode([ComparisonNode(VariableUseNode("skillLevel","Real"),">",14.4610572058)],\
                  [G(35.956103,50.663757),G(48.931459,60.751408)])
    false = IfNode([ComparisonNode(VariableUseNode("skillLevel","Real"),">",10.8876738558)],\
                  [G(44.271139,62.430392),G(36.320374,25.620630)])
    ast.addChild(VariableDeclNode("testPerformance","Real",\
                                  IfNode([VariableUseNode("tired","Boolean")],\
                                         [true,false])))
    print ast.strings()[0]
    print "score =", estimateScore(ast,dataset)
    

testMoG11()
