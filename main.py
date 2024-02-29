import pandas as pd
import time as t
import secuential_tree.rules as rules
import parallel_tree.prules as prules
import parallel_tree.trules as trules
import utils as ut
import os

def test_model(model,database: pd.DataFrame):
    train,test = ut.create_k(database,k=10)
    
    calification = ut.cross_validation_train(model,test=test,train=train)
    return calification

def test_both_trees():
    banned = ["balance-scale.csv"]
    databases = [database for database in os.listdir("data") if database.endswith(".csv") and database not in banned]
    dataframe = pd.DataFrame(columns=["NombreBaseDatos",
                                      "RecallPromedioModeloParalelo",
                                      "AccuracyPromedioModeloParalelo",
                                      "TiempoPromedioModeloParalelo",
                                      "RecallPromedioModeloSecuencial",
                                      "AccuracyPromedioModeloSecuencial",
                                      "TiempoPromedioModeloSecuencial",
                                      "Aceleracion"
                                      ])
    speedup = 0
    for database in databases:
        print(database)
        loaded_data = pd.read_csv("data/"+database)
        parallel_calification = test_model(prules,loaded_data)
        serial_calification = test_model(rules,loaded_data)
        speedup = serial_calification[-1]/parallel_calification[-1]
        dataframe.loc[len(dataframe)] = database,*parallel_calification,*serial_calification,speedup
    dataframe.to_excel("tested.xlsx", index=False)



if __name__ == "__main__":
    test_database = "data/iris.csv"
    data = pd.read_csv(test_database)
    max_depth = 5
    min_samples_split = 20
    min_information_gain  = 1e-5
    #print(test_model(prules,data))
    test_both_trees()
    #start = t.time()
    #decisions = prules.train_tree(data,'class',True, max_depth,min_samples_split,min_information_gain,max_categories=30)
    #decisions2 = rules.train_tree(data,'class',True, max_depth,min_samples_split,min_information_gain,max_categories=30)
    #print(f"Parallel Tree:\n{decisions}")
    #print(f"Secuential Tree:\n{decisions2}")
    
    #end = t.time()
    #print(f"Tiempo transcurrido {end-start}")
    #print(decisions)
    #print(prules.predict(root,decisions,data))
    #print(rules.predict(decisions,data))