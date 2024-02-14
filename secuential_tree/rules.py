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
    a = sum(mask)
    b = mask.shape[0] - a
    
    if(a == 0 or b ==0): 
        ig = 0
    
    else:
        if y.dtypes != 'O':
            ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
        else:
            ig = func(y)-a/(a+b)*func(y[mask])-b/(a+b)*func(y[-mask])
        
        return ig