__author__ = 'zhawy068'

import numpy as np
import csv

def get_freqitemsets(fname, minsupport, maxlhs):
    #minsupport is an integer percentage (e.g. 10 for 10%)
    #maxlhs is the maximum size of the lhs
    #first load the data
    data, Y = load_data(fname)
    #Now find frequent itemsets
    #Mine separately for each class
    data_pos = [x for i, x in enumerate(data) if Y[i, 0] == 0]
    data_neg = [x for i, x in enumerate(data) if Y[i, 0] == 1]
    assert len(data_pos) + len(data_neg) == len(data)
    try:
        itemsets = [r[0] for r in fpgrowth(data_pos, supp=minsupport, zmax=maxlhs)]
        itemsets.extend([r[0] for r in fpgrowth(data_neg, supp=minsupport, zmax=maxlhs)])
    except TypeError:
        itemsets = [r[0] for r in fpgrowth(data_pos, supp=minsupport, max=maxlhs)]
        itemsets.extend([r[0] for r in fpgrowth(data_neg, supp=minsupport, max=maxlhs)])
    itemsets = list(set(itemsets))
    print(len(itemsets), 'rules mined')
    #Now form the data-vs.-lhs set
    #X[j] is the set of data points that contain itemset j (that is, satisfy rule j)
    X = [set() for j in range(len(itemsets) + 1)]
    X[0] = set(range(len(data)))  #the default rule satisfies all data
    for (j, lhs) in enumerate(itemsets):
        X[j + 1] = set([i for (i, xi) in enumerate(data) if set(lhs).issubset(xi)])
    #now form lhs_len
    lhs_len = [0]
    for lhs in itemsets:
        lhs_len.append(len(lhs))
    nruleslen = Counter(lhs_len)
    lhs_len = array(lhs_len)
    itemsets_all = ['null']
    itemsets_all.extend(itemsets)
    return X, Y, nruleslen, lhs_len, itemsets_all





if __name__ == '__main__':
    translate('Annotations.csv', 'coding-11687-Cancer.txt')