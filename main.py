import pandas as pd
import time as t
import secuential_tree.rules as rules
import parallel_tree.prules as prules
import parallel_tree.trules as trules


if __name__ == "__main__":
    test_database = "data/letter.csv"
    data = pd.read_csv(test_database)
    print(data.head())
    max_depth = 5
    min_samples_split = 20
    min_information_gain  = 1e-5
    
    
    #*Checking secuential tree
    start = t.time()
    decisiones = passrules.train_tree(data,'class',True, max_depth,min_samples_split,min_information_gain,max_categories=30)
    end = t.time()
    #print(decisiones)
    print(f"Tiempo transcurrido {end-start}")
    #*Checking parallel tree
    #build_tree(Tree(dataset))
    #build_tree(ParallelTree(dataset))
    