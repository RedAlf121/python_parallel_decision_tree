import pandas as pd
import time as t
import secuential_tree.rules as rules



if __name__ == "__main__":
    test_database = "data/obessity.csv"
    data = pd.read_csv(test_database)
    #!REMEMBER TO COMMENT THESE SENTENCES TO TEST ANOTHER DATA
    data['obese'] = (data.Index >= 4).astype('int').astype('str')
    data.drop('Index', axis = 1, inplace = True)
    print(rules.gini_impurity(data.Gender))
    print(rules.entropy(data.Gender))
    print(rules.information_gain(data['obese'], data['Gender'] == 'Male'))
    #*Checking secuential tree
    #*Checking parallel tree
    #build_tree(Tree(dataset))
    #build_tree(ParallelTree(dataset))
    