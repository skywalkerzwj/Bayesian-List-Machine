__author__ = 'Jon Stark'

import numpy as np
import pandas as pd
import math
import csv
import cPickle as pickle
from sklearn import preprocessing
from BRL_code import *
from collections import defaultdict, Counter
from fim import fpgrowth  #this is PyFIM, available from http://www.borgelt.net/pyfim.html
#from matplotlib import pyplot as plt #Uncomment to use the plot_chains function


def read2(file1, file2, label, translation):
    #load label
    Y = np.genfromtxt(label)

    #load translation table
    with open(translation, 'rb') as f:
        reader1 = csv.reader(f)
        nameTable = list(reader1)

    #load and prepare data
    data1 = pd.read_csv(file1, sep='\t', header=0)
    data2 = pd.read_csv(file2, sep='\t', header=0)
    data1 = data1.T[1:]
    data2 = data2.T[1:]
    data = data1.append(data2)
    med = data.median(axis=0)

    for i in xrange(0, 144):
        print i
        for j in xrange(0, 11687):
            if math.isnan(data.iloc[i, j]):
                print j
                continue
            elif data.iloc[i, j] < med[j]:
                data.iloc[i, j] = nameTable[j][0]+'_low'
            elif data.iloc[i, j] >= med[j]:
                data.iloc[i, j] = nameTable[j][0]+'_high'
   # print data.iloc[7, 0]
    return Y, data


def translate(nameTable, dataset):
    translation = list()
    translated=list()
    with open(nameTable, 'rb') as f:
        reader1 = csv.reader(f)
        table = list(reader1)
    with open(dataset, 'rb') as f:
        reader2 = csv.reader(f, delimiter='\t')
        data = list(reader2)
    for i in xrange(1, np.shape(data)[0]):
    # for i in xrange(1, 100):
        tag = 0

        print i
        for j in xrange(1, np.shape(table)[0]):
            if data[i][0] == table[j][0]:
                print(table[j][1])
                translation.append(table[j][1])
                tag = 1
        if tag == 0:
            translation.append(table[0][1])
            #print translation
    with open(dataset+'-translated2.csv', 'wb') as f:
        for row in translation:
            f.write('%s\n' % row)

    # with open(dataset+'-translated.csv', 'wb') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerows(translation)
    return translation


def read3(file1, file2, label, translation):
    #load label
    Y = np.genfromtxt(label)

    #load translation table
    with open(translation, 'rb') as f:
        reader1 = csv.reader(f)
        nameTable = list(reader1)
    #load and prepare data
    data1 = pd.read_csv(file1, sep='\t', header=0)
    data2 = pd.read_csv(file2, sep='\t', header=0)
    data1 = data1.T[1:]
    data2 = data2.T[1:]
    df = data1.append(data2)
    #remove columns(gene) with missing values
    df = df.dropna(axis=1, how='any')
    med = df.median(axis=0)

    #for i in xrange(0, 144):
        #print i
    result = np.ndarray
    for j in xrange(0, 9713):
        print j
        binarizer = preprocessing.Binarizer().fit(df.iloc[:,j])
        binarizer.threshold = med.iloc[j]
        temp = binarizer.transform(df.iloc[:,j])
        temp = np.hstack((temp, temp.T))
    return Y, temp


def get_freqitemsets_1(dataset, Y, minsupport, maxlhs):
    #minsupport is an integer percentage (e.g. 10 for 10%)
    #maxlhs is the maximum size of the lhs
    #first load the data
    #Now find frequent itemsets
    #Mine separately for each class
    data_pos = [x for i, x in enumerate(dataset) if Y[i] == 0]
    data_neg = [x for i, x in enumerate(dataset) if Y[i] == 1]
    print 'ok'
    #data_pos = dataset[0:72]
    #data_neg = dataset[72:144]
    assert len(data_pos) + len(data_neg) == len(dataset)
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
    X[0] = set(range(len(dataset)))  #the default rule satisfies all data
    for (j, lhs) in enumerate(itemsets):
        X[j + 1] = set([i for (i, xi) in enumerate(dataset) if set(lhs).issubset(xi)])
    #now form lhs_len
    lhs_len = [0]
    for lhs in itemsets:
        lhs_len.append(len(lhs))
    nruleslen = Counter(lhs_len)
    lhs_len = np.array(lhs_len)
    itemsets_all = ['null']
    itemsets_all.extend(itemsets)
    return X, Y, nruleslen, lhs_len, itemsets_all


def topscript(dataset, Y):
    #Prior hyperparameters
    lbda = 3.  #prior hyperparameter for expected list length (excluding null rule)
    eta = 1.  #prior hyperparameter for expected list average width (excluding null rule)
    alpha = np.array([1., 1.])  #prior hyperparameter for multinomial pseudocounts

    #rule mining parameters
    maxlhs = 2  #maximum cardinality of an itemset
    minsupport = 10  #minimum support (%) of an itemset

    #mcmc parameters
    numiters = 50000  # Uncomment plot_chains in run_bdl_multichain to visually check mixing and convergence
    thinning = 1  #The thinning rate
    burnin = numiters // 2  #the number of samples to drop as burn-in in-simulation
    nchains = 3  #number of MCMC chains. These are simulated in serial, and then merged after checking for convergence.

    #End parameters

    #Now we load data and do MCMC
    permsdic = defaultdict(default_permsdic)  #We will store here the MCMC results
    Xtrain, Ytrain, nruleslen, lhs_len, itemsets = get_freqitemsets_1(dataset, Y, minsupport,
                                                                    maxlhs)  #Do frequent itemset mining from the training data
    #Xtest, Ytest, Ylabels_test = get_testdata(fname + '_test', itemsets)  #Load the test data
    print('Data loaded!')

    #Do MCMC
    res, Rhat = run_bdl_multichain_serial(numiters, thinning, alpha, lbda, eta, Xtrain, Ytrain, nruleslen, lhs_len,
                                          maxlhs, permsdic, burnin, nchains, [None] * nchains)

    #Merge the chains
    permsdic = merge_chains(res)

    ###The point estimate, BRL-point
    d_star = get_point_estimate(permsdic, lhs_len, Xtrain, Ytrain, alpha, nruleslen, maxlhs, lbda,
                                eta)  #get the point estimate

    if d_star:
        #Compute the rule consequent
        theta, ci_theta = get_rule_rhs(Xtrain, Ytrain, d_star, alpha, True)

        #Print out the point estimate rule
        print('antecedent risk (credible interval for risk)')
        for i, j in enumerate(d_star):
            print(itemsets[j], theta[i], ci_theta[i])

        #  Evaluate on the test data
        preds_d_star = preds_d_t(Xtest, Ytest, d_star, theta)  # Use d_star to make predictions on the test data
        accur_d_star = preds_to_acc(preds_d_star, Ylabels_test)  # Accuracy of the point estimate
        print('accuracy of point estimate', accur_d_star)

    ### The full posterior, BRL-post
    preds_fullpost = preds_full_posterior(Xtest, Ytest, Xtrain, Ytrain, permsdic, alpha)
    accur_fullpost = preds_to_acc(preds_fullpost, Ylabels_test)  # Accuracy of the full posterior
    print('accuracy of full posterior', accur_fullpost)

    return permsdic, d_star, itemsets, theta, ci_theta, preds_d_star, accur_d_star, preds_fullpost, accur_fullpost


if __name__ == '__main__':
    # file1 = "coding-11687-Cancer.txt"
    # file2 = "coding-11687-Normal.txt"
    # label = "coding-11687-label.txt"
    # translation = "coding-11687-Cancer-translation2.csv"
    #
    # Y, data = read3(file1, file2, label, translation)
    # print data.shape
    # pickle.dump(data, open("save.p", "wb"))
    pkl_file = open('save.p', 'rb')
    dataset = pickle.load(pkl_file)
    dataset = dataset.values.tolist()
   # dataset = dataset.as_matrix
    Y = np.genfromtxt("coding-11687-label.txt")
    topscript(dataset, Y)