import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
import pickle
import cPickle
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def conv(i):
	s = float(i)
	if s>0:
		return 1
	else:
		return s

def function1(train,test):
    train = np.delete(train, 0, 0)
    train = np.delete(train, 0, 1)
    test = np.delete(test, 0, 0)
    test = np.delete(test, 0, 1)
    np.random.shuffle(train)
    t_result = train[:,0]
    m_result = test[:,0]
    train = np.delete(train, 0, 1)
    train = np.delete(train, 0, 1)
    test = np.delete(test, 0, 1)
    test = np.delete(test, 0, 1)
    m_res = m_result.tolist()[:]
    #t_data = train.tolist()[:]
    #m_data = test.tolist()[:]
    t_res = t_result.tolist()[:]
    m_res = map(conv, m_res)
    t_res = map(conv, t_res)
    #m_data = [map(conv, x) for x in m_data] 
    #t_data = [map(conv, x) for x in t_data]
    return (test,m_res, train,t_res)


def function2(train,test):
    print train.shape
    m,n = train.shape
    p,q = test.shape
#    t1= np.random.rand(m,9616)
#    m1= np.random.rand(p,9616)
#    for i in range(9616):
#        if chiscore[i] >2.4:
#            t1.append(train[:,i])
#            m1.append(test[:,i])
    t_data = train.tolist()[:]
    m_data = test.tolist()[:]
    m_data = [map(conv, x) for x in m_data]
    t_data = [map(conv, x) for x in t_data]
    return (m_data, t_data)


def function1(train1,test1):
    fea = {}
    m, n = train1.shape 
    for i in range(0,n):
        sn = 0
        sp = 0
        for j in range(0,m):
            if int(label[j])== -1:
                sn = sn + int(train1[j,i])
            elif int(label[j])== 1:
                sp= sp + int(train1[j,i])
        if sp == 0 or sn == 0:
            fea[feature_set[i]] = (sn,sp)
        print i
#    with open("test_data", 'wb') as f:
#        np.save(f,test)

    return fea

train = np.load("detecting-stance-in-tweets-pre/csv.pickle1")
#train1 = np.load("train_data")
#test1 = np.load("test_data")

test = np.load("detecting-stance-in-tweets-post/testcsv.pickle1")
scaler = StandardScaler()

#feature_set = train[0,:]
#label = train[:,1]

#feature_set = feature_set[3:]
#label = label[1:]
#print label
#fea = function1(train1,test1)
#text_file = open("featureperclass-+", "w")
#for key, value in sorted(fea.iteritems(), key=lambda (k,v): (v,k)):
#    text_file.write("%s: %s\n" % (key, value))
#text_file.close()
