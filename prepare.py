__author__ = 'Jon Stark'
import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
import os, time, json, traceback, sys
from scipy.special import gammaln
from scipy.stats import poisson, beta
from BRL_code import *
try:
    import cPickle as Pickle
except:
    import Pickle
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
    #load attributes
    data1 = np.genfromtxt(file1, skip_header=1)
    data1 = data1[:, 1:]
    data2 = np.genfromtxt(file2, skip_header=1)
    data2 = data2[:, 1:]
    data = np.hstack((data1, data2))
    data.dtype.names = nameTable
    # std_data = preprocessing.scale(data, axis=0, with_mean=True, with_std=True, copy=True)
    # b_data = preprocessing.binarize(std_data, threshold=0.0, copy=True)
    # for i in xrange(0, np.shape(b_data)[0]):
    #     for j in xrange(0, np.shape(b_data)[1]):
    #         if b_data[i, j] == 0:
    #             b_data[i, j] = nameTable[j]+'_low'
    #         if b_data[i, j] == 1:
    #             b_data[i, j] = nameTable[j]+'_high'
    #
    #
    #
    # print b_data
    # #remove features with missing values
    # data_wo_nan = data[~np.isnan(data).any(axis=1)]
    # data_wo_nan = np.transpose(data_wo_nan)


    return data, Y


def get_freqitemsets_1(file1, file2, label, minsupport, maxlhs):
    #minsupport is an integer percentage (e.g. 10 for 10%)
    #maxlhs is the maximum size of the lhs
    #first load the data
    data, Y = read2(file1, file2, label)
    #Now find frequent itemsets
    #Mine separately for each class
    data_pos = [x for i, x in enumerate(data) if Y[i] == 0]
    data_neg = [x for i, x in enumerate(data) if Y[i] == 1]
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
    lhs_len = np.array(lhs_len)
    itemsets_all = ['null']
    itemsets_all.extend(itemsets)
    return X, Y, nruleslen, lhs_len, itemsets_all


def topscript():
    file1 = "coding-11687-Cancer.txt"
    file2 = "coding-11687-Normal.txt"
    label = "coding-11687-label.txt"
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
    Xtrain, Ytrain, nruleslen, lhs_len, itemsets = get_freqitemsets_1(file1, file2, label, minsupport,
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
    file1 = "coding-11687-Cancer.txt"
    file2 = "coding-11687-Normal.txt"
    label = "coding-11687-label.txt"
    data, Y = read2(file1, file2, label,"coding-11687-Cancer-translation.csv")

    data.dtype.names
    #topscript()

    #get_freqitemsets_1(data1,data2,label,minsupport,maxlhs)
        #b_data.shape