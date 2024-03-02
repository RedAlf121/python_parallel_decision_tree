from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
import itertools
import operator
from queue import Queue
import re
import time
from typing import Literal
import pandas as pd
import numpy as np
def gini_impurity(y):
    '''
    Given a Pandas Series, it calculates the Gini Impurity. 
    y: variable with which calculate Gini Impurity.
    '''
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)

    else:
        raise('Object must be a Pandas Series.')
def entropy(y):
    '''
    Given a Pandas Series, it calculates the entropy. 
    y: variable with which calculate entropy.
    '''
    if isinstance(y, pd.Series):
        a = y.value_counts()/y.shape[0]
        entropy = np.sum(-a*np.log2(a+1e-9))
        return(entropy)

    else:
        raise('Object must be a Pandas Series.')

def variance(y):
    '''
    Function to help calculate the variance avoiding nan.
    y: variable to calculate variance to. It should be a Pandas Series.
    '''
    return y.var() if(len(y) != 1) else 0

def information_gain(y, mask, func=entropy):
    '''
    It returns the Information Gain of a variable given a loss function.
    y: target variable.
    mask: split choice.
    func: function to be used to calculate Information Gain in case os classification.
    '''
    a = np.sum(mask)
    b = mask.shape[0] - a
    
    if(a == 0 or b ==0): 
        ig = 0
    
    else:
        if y.dtypes != 'O':
            
            variance_y = variance(y)
            variance_mask = variance(y[mask])
            variance_neg_mask = variance(y[-mask])
            ig = variance_y - (a/(a+b)* variance_mask) - (b/(a+b)*variance_neg_mask)
        else:
            function_y = func(y)
            function_mask = func(y[mask])
            function_neg_mask = func(y[-mask])
            ig = function_y-(a/(a+b)*function_mask)-(b/(a+b)*function_neg_mask)

        return ig
    
def categorical_options(a):
    '''
    Creates all possible combinations from a Pandas Series.
    a: Pandas Series from where to get all possible combinations. 
    '''
    a = a.unique()
    opciones = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for L in range(0, len(a)+1):
            futures.append(executor.submit(iterate_combinations, a, L))
        for future in futures:
            opciones.extend(future.result())
    return opciones[1:-1]

def iterate_combinations(a, L):
    opciones = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        opciones = pool.map(lambda subset: list(subset),itertools.combinations(a, L))
    return opciones

def max_information_gain_split(x, y, func=entropy):
    '''
    Given a predictor & target variable, returns the best split, the error and the type of variable based on a selected cost function.
    x: predictor variable as Pandas Series.
    y: target variable as Pandas Series.
    func: function to be used to calculate the best split.
    '''

    split_value = []
    ig = [] 

    numeric_variable = True if x.dtypes != 'O' else False

    # Create options according to variable type
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else: 
        options = categorical_options(x)

    # Calculate ig for all values
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        ig_value = pool.map(partial(value_ig,x, y, func, numeric_variable,split_value),options)
        ig = list(ig_value)
        
    end = time.time()
    print(f"Transcurrió: {end-start}")
    # Check if there are more than 1 results if not, return False
    if len(ig) == 0:
        return(None,None,None, False)

    else:
    # Get results with highest IG
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return(best_ig,best_split,numeric_variable, True)

def value_ig(x, y, func, numeric_variable, split_value, val):
    mask =   x < val if numeric_variable else x.isin(val)
    val_ig = information_gain(y, mask, func)
    split_value.append(val)
    return val_ig

def get_best_split(y, data):
  '''
  Given a data, select the best split and return the variable, the value, the variable type and the information gain.
  y: name of the target variable
  data: dataframe where to find the best split.
  '''
  masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y])
  if np.sum(masks.loc[3,:]) == 0:
    return(None, None, None, None)

  else:
    # Get only masks that can be splitted
    masks = masks.loc[:,masks.loc[3,:]]
    # Get the results for split with highest IG
    split_variable = masks.iloc[0].astype(np.float32).idxmax()

    #split_valid = masks[split_variable][]
    split_value = masks[split_variable][1] 
    split_ig = masks[split_variable][0]
    split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)


def make_split(variable, value, data, is_numeric):
    '''
    Given a data and a split conditions, do the split.
    variable: variable with which make the split.
    value: value of the variable to make the split.
    data: data to be splitted.
    is_numeric: boolean considering if the variable to be splitted is numeric or not.
    '''
    if is_numeric:
        data_1 = data[data[variable] < value]
        data_2 = data[(data[variable] < value) == False]

    else:
        data_1 = data[data[variable].isin(value)]
        data_2 = data[(data[variable].isin(value)) == False]

    return(data_1,data_2)

def make_prediction(data, target_factor):
    '''
    Given the target variable, make a prediction.
    data: pandas series for target variable
    target_factor: boolean considering if the variable is a factor or not
    '''
    # Make predictions
    if target_factor:
        pred = data.value_counts().idxmax()
    else:
        pred = data.mean()

    return pred
def train_tree(data, y, target_factor, max_depth=None, min_samples_split=None, min_information_gain=1e-20, counter=0, max_categories=20):
    tree = dict[str, list]()
    processes = Queue()
    _, node, yes, no = train_tree_parallel(("", data, y, target_factor, max_depth, min_samples_split, min_information_gain, counter, max_categories))
    root = node
    tree.update({node: []})
    processes.put(yes)
    processes.put(no)
    with ProcessPoolExecutor() as pool:
        while not processes.empty():
            working = pool.map(train_tree_parallel, [processes.get() for _ in range(processes.qsize())])
            for parent, node, yes, no in working:
                if yes:
                    processes.put(yes)
                    processes.put(no)
                if parent in tree.keys():
                    tree[parent].append(node)
                else:
                    tree.update({parent: [node]})
    return root, tree

"""
def train_tree(data, y, target_factor, max_depth=None, min_samples_split=None, min_information_gain=1e-20, counter=0, max_categories=20):
    tree = dict[str,list]()
    processes = Queue()
    working = Queue()
    _,node,yes,no = train_tree_parallel("",data, y, target_factor, max_depth=max_depth, min_samples_split=min_samples_split, min_information_gain=min_information_gain, counter=counter, max_categories=max_categories)
    root = node
    tree.update({node:[]})
    processes.put(yes)
    processes.put(no)
    while not processes.empty():
        with ProcessPoolExecutor() as pool:
            working = pool.map(train_tree_parallel,processes)
            #while not processes.empty():
            #    working.put(pool.submit(train_tree_parallel,*processes.get()))
        while not working.empty():
            ##Crear el árbol
            parent, node,yes,no, = working.get().result()
            if yes:
                processes.put(yes) 
                processes.put(no)
            if parent in tree.keys():
                tree[parent].append(node)
            else:
                tree.update({parent:[node]})
    return root,tree
"""
def train_tree_parallel(parameters):
    parent,data, y, target_factor, max_depth, min_samples_split, min_information_gain, counter, max_categories = parameters
    '''
    Trains a Decission Tree
    data: Data to be used to train the Decission Tree
    y: target variable column name
    target_factor: boolean to consider if target variable is factor or numeric.
    max_depth: maximum depth to stop splitting.
    min_samples_split: minimum number of observations to make a split.
    min_information_gain: minimum ig gain to consider a split to be valid.
    max_categories: maximum number of different values accepted for categorical values. High number of values will slow down learning process. R
    '''
    with ThreadPoolExecutor(3) as pool:
        pool.submit(fulfilled,data, counter, max_categories)
        depth_thread = pool.submit(checking_depth,max_depth, counter)
        sample_thread = pool.submit(checking_sample,data, min_samples_split)
        depth_cond = depth_thread.result()
        sample_cond = sample_thread.result()

    # Check for ig condition
    if depth_cond & sample_cond:
        var, val, ig, var_type = get_best_split(y, data)

        # If ig condition is fulfilled, make split
        if ig is not None and ig >= min_information_gain:
            counter += 1
            left, right = make_split(var, val, data, var_type)

            # Instantiate sub-tree
            split_type = "<=" if var_type else "in"
            question = "{} {} {}".format(var, split_type, val)
            # Find answers (recursion)
            yes_answer = (question,left, y, target_factor, max_depth, min_samples_split, min_information_gain, counter, max_categories)
            no_answer = (question,right, y, target_factor, max_depth, min_samples_split, min_information_gain, counter, max_categories)

        # If it doesn't match IG condition, make prediction
        else:
            question = make_prediction(data[y], target_factor)
            yes_answer = ()
            no_answer = ()

    # Drop dataset if doesn't match depth or sample conditions
    else:
        question = make_prediction(data[y], target_factor)
        yes_answer = ()
        no_answer = ()
    return parent,question,yes_answer,no_answer

def checking_sample(data, min_samples_split):
    if min_samples_split is None:
        sample_cond = True
    else:
        if data.shape[0] > min_samples_split:
            sample_cond = True
        else:
            sample_cond = False
    return sample_cond

def checking_depth(max_depth, counter):
    if max_depth is None:
        depth_cond = True
    else:
        if counter < max_depth:
            depth_cond = True
        else:
            depth_cond = False
    return depth_cond

def fulfilled(data, counter, max_categories):
    if counter == 0:
        types = data.dtypes
        check_columns = types[types == "object"].index
        for column in check_columns:
            var_length = len(data[column].value_counts())
            if var_length > max_categories:
                raise ValueError('The variable ' + column + ' has ' + str(var_length) + ' unique values, which is more than the accepted ones: ' + str(max_categories))

def predict_recursive(root,tree,columns,value):
    label = ""
    if tree.get(root,-1) == -1:
        label = root
    else:
        label = predict_recursive(tree[root][0],tree,columns,value) if check_condition(root,value,columns) else predict_recursive(tree[root][1] if len(tree[root])>1 else None,tree,columns,value)
    return label

def check_condition(root: str, value: list,columns: list):
    evaluation = root.split(" ")
    compare = 0
    for i in columns:
        if i == evaluation[0]:
            compare = columns.index(i)+1
            break
    return eval(f"\"{value[compare]}\" " if isinstance(value[compare],str) else f"{value[compare]} "+" ".join(evaluation[1:]))

def predict(tree,dataframe):
    root = tree[0]
    return [predict_recursive(root,tree[1],dataframe.columns.tolist(),frame) for frame in dataframe.to_records().tolist()]