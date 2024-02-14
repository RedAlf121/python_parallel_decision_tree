import pandas as pd
from decision_tree import Tree
from parallel_decision_tree import ParallelTree
import time as t

def create_data_set():
    data_set = [[1,1,'yes'],[1,1,'yes'],[1,0,'yes'],[0,1,'no'],[0,1,'no'],[0,0,'no'],[0,0,'yes'],[0,0,'no']]
    labels = ['¿Hay un inmueble?', '¿Hay un automóvil?','class']
    return pd.DataFrame(data_set,columns=labels)


def build_tree(tree):
    start = t.time()
    #TODO construir el árbol
    end = t.time()
    print(f"Tiempo transcurrido {end-start}")

if __name__ == "__main__":
    test_database = "data/iris.csv"
    #! for real test: dataset  = pd.read_csv(test_database)
    dataset = create_data_set()
    tree = Tree(dataset)
    print(f"Attributes:\n{dataset.iloc[:,:-1]}")
    print(f"Labels:\n{dataset.iloc[:,-1]}")
    print("PROBANDO TUTORIAL:")
    print(tree.entropy(tree.dataset))
    result = tree.split_data_set(tree.dataset,0,1 )
    print(result)
    result = tree.split_data_set(tree.dataset,0,0 )
    print(result)
    result = tree.split_data_set(tree.dataset,1,1 )
    print(result)
    result = tree.split_data_set(tree.dataset,1,0 )
    print(result)
    index = tree.choose_best_features(tree.dataset)
    print(index)
    print(tree.class_num( ['yes','no','no','yes','no']))
    #*Checking secuential tree
    #*Checking parallel tree
    #build_tree(Tree(dataset))
    #build_tree(ParallelTree(dataset))
    