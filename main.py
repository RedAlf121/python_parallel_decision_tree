import pandas as pd
import time as t
import secuential_tree.rules as rules



if __name__ == "__main__":
    test_database = "data/obessity.csv"
    data = pd.read_csv(test_database)
    print(data.head())
    max_depth = 5
    min_samples_split = 20
    min_information_gain  = 1e-5
    
    
    #*Checking secuential tree
    start = t.time()
    decisiones = rules.train_tree(data,'Index',True, max_depth,min_samples_split,min_information_gain)
    end = t.time()
    #print(decisiones)
    print(f"Tiempo transcurrido {end-start}")
    #*Checking parallel tree
    #build_tree(Tree(dataset))
    #build_tree(ParallelTree(dataset))
    