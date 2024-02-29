import time
import numpy as np
import random
import pickle as pck
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, accuracy_score, precision_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import secuential_tree.rules as rules
import parallel_tree.prules as prules


def shuffle_data(data, labels):
	if(len(data) != len(labels)):
		raise Exception("The given data and labels do NOT have the same length")
	lista=[]

	for i, j in zip(data, labels):
		lista.append(np.r_ [i, [j]])
	lista = np.array(lista)
	np.random.shuffle(lista)
	x,y = [],[]
	for i in lista:
		x.append(i[:-1])
		y.append(i[-1])

	return np.array(x), np.array(y)

def create_k(database, k=5):
	train_list=[]
	test_list=[]
	data = database
	kf = KFold(n_splits=k)
	for train_index, test_index in kf.split(data):
		train, test= data.iloc[train_index], data.iloc[test_index]

		train_list.append(train)

		test_list.append(test)

	return train_list, test_list

def cross_validation_train(model, train, test):
	avg_recall=0
	avg_presi=0
	avg_auc=0
	avg_acc=0
	avg_time = 0
	start = 0
	end = 0
	max_depth = 5
	min_samples_split = 20
	min_information_gain  = 1e-5
	a=1
	real_y=np.array([],dtype=str)
	predicc_y=np.array([],dtype=str)
	classname = train[0].columns.tolist()[-1]
	for trainn, testss in zip(train,test):
		
		y_test=testss.iloc[:,-1]
		print("Para el", a," k conjunto de prueba y entrenamiento")
		start = time.time()
		tree = model.train_tree(trainn,classname,True, max_depth,min_samples_split,min_information_gain,max_categories=30)
		end = time.time()
		predictions = model.predict(tree,testss)

		real_y = np.append(real_y, y_test)
		predicc_y = np.append(predicc_y, predictions)

		score_recll = recall_score(y_test,predictions, average = 'macro')
		score_acc = accuracy_score(y_test,predictions)

		avg_recall = score_recll + avg_recall
		avg_acc = score_acc + avg_acc
		avg_time+=(end-start)
		a+=1

	avg_recall = avg_recall/len(train)
	avg_acc = avg_acc/len(train)
	avg_time = avg_time/len(train)
	print("The final cross_val recall is", avg_recall, ", roc_auc is", avg_auc,", precision is",avg_presi,", accuracy is", avg_acc, "and time is", avg_time)
	return avg_recall, avg_acc, avg_time
