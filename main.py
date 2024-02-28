import pandas as pd
import time as t
import secuential_tree.rules as rules
import parallel_tree.prules as prules
import parallel_tree.trules as trules


if __name__ == "__main__":
    test_database = "data/iris.csv"
    data = pd.read_csv(test_database)
    print(data.head())
    print(data.columns.tolist())
    max_depth = 5
    min_samples_split = 20
    min_information_gain  = 1e-5
    
    
    #*Checking secuential tree
    start = t.time()
    root,decisions = prules.train_tree(data,'class',True, max_depth,min_samples_split,min_information_gain,max_categories=30)
    #decisions = rules.train_tree(data,'class',True, max_depth,min_samples_split,min_information_gain,max_categories=30)
    end = t.time()
    print(f"Tiempo transcurrido {end-start}")
    print(decisions)
    #print(prules.predict(root,decisions,data))
    #print(rules.predict(decisions,data))
    #*Checking parallel tree
    #build_tree(Tree(dataset))
    #build_tree(ParallelTree(dataset))
    