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
    banned = ["balance-scale.csv", "flags_religion.csv", "letter.csv","molecular-biology_promoters.csv","splice.csv","vehicle.csv"]
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
    max_databases = 10 #testing with 10 databases
    count_databases = max_databases 
    for database in databases:
        if count_databases == 0:
            break
        print(f"Database #{max_databases-count_databases+1}: {database}")
        loaded_data = pd.read_csv("data/"+database)
        print(f"Probando modelo paralelo {database} #{max_databases-count_databases+1}")
        parallel_calification = test_model(prules,loaded_data)
        print(f"Fin de probar modelo paralelo {database} #{max_databases-count_databases+1}")
        print(f"Probando modelo secuencial {database} #{max_databases-count_databases+1}")
        serial_calification = test_model(rules,loaded_data)
        print(f"Fin de probar modelo secuencial {database} #{max_databases-count_databases+1}")
        speedup = serial_calification[-1]/parallel_calification[-1]
        dataframe.loc[len(dataframe)] = database,*parallel_calification,*serial_calification,speedup
        count_databases-=1
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