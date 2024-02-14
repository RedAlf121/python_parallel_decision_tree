import pandas as pd
from decision_tree import Tree
from parallel_decision_tree import ParallelTree
import time as t



if __name__ == "__main__":
    test_database = "data/iris.csv"
    dataset  = pd.read_csv(test_database)
    #*Checking secuential tree
    #*Checking parallel tree
    #build_tree(Tree(dataset))
    #build_tree(ParallelTree(dataset))
    