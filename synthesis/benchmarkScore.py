#!/usr/bin/python

from parser import parse_from_file
from astDB import Dataset
from score import estimateScore, getMoG, ScoreEstimator
import sys

scores = {}

def test(ast_file,dataset_file):
    ast = parse_from_file(ast_file)
    dataset = Dataset(dataset_file)
    scoreEstimator = ScoreEstimator(dataset)
    scores[(ast_file[ast_file.rfind('/')+1:]).lower().split(".")[0]] = -1.0*estimateScore(ast, scoreEstimator)
    #print ast_file[ast_file.rfind('/')+1:], "\t", estimateScore(ast, dataset)

def run():
    if len(sys.argv) > 1:
        flag = sys.argv[1]

        if flag == "-I" or flag == "--input":
            test(sys.argv[2],sys.argv[3])
            return
        elif flag == "-A" or flag == "--all":
            test_file = sys.argv[2]
            dataset_dir_name = sys.argv[3]
            dir = test_file[:test_file.rfind('/')]
            test_list = open(test_file,'r')
            for name in test_list:
                name = name.rstrip()
                nameSuffix = name.split("/")[-1]
                nameSuffix = nameSuffix.split("-")[0] # let's get rid of whatever extra stuff we've added on to the end (to distinguish btwn diff synthesized progs)
                test(dir + "/" + name + '.blog', \
                     dir + '/'+dataset_dir_name+'/' + nameSuffix + '.csv')
            test_list.close()
            return
        elif flag == "--mog":
            ast_file = sys.argv[2]
            print getMoG(parse_from_file(ast_file))
            return
        
    print "Usage: ./benchmarkScore.py -I/--input [PROGRAM] [DATASET]"
    print "Usage: ./benchmarkScore.py -A/--all ../benchmarkSuite/tests.list datasets"
    print "Usage: ./benchmarkScore.py --mog [PROGRAM]"

run()
print scores
    
