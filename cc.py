import numpy as np
from sklearn import svm
from sklearn.feature_selection import chi2
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


def function3(train,test):
    print train.shape
    m,n = train.shape
    p,q = test.shape
    t1= np.random.rand(m,k)
    m1= np.random.rand(p,k)
    for i in range(k):
        if chiscore[i] >=2:
            t1[:,i] = train[:,i]
            m1[:,i] = test[:,i]
    t_data = t1.tolist()[:]
    m_data = m1.tolist()[:]
    m_data = [map(conv, x) for x in m_data]
    t_data = [map(conv, x) for x in t_data]
    return (m_data, t_data)


train = np.load("detecting-stance-in-tweets-pre/csv.pickle1")
#train1 = np.load("train_data1")
#test1 = np.load("test_data1")

#for i in range(2, 11):
#    t = np.load("detecting-stance-in-tweets-pre/csv.pickle" + str(i))
#    np.delete(t, 0, 0)
#    train = np.concatenate((train, t), axis=0)
#
test = np.load("detecting-stance-in-tweets-post/testcsv.pickle1")
scaler = StandardScaler()


#for i in range(2, 3):
#    t = np.load("detecting-stance-in-tweets-post/testcsv.pickle" + str(i))
#    np.delete(t,0,0
#
#        )
#    test = np.concatenate((test, t), axis=0)
#
#print train.shape, test.shape	
# train  = pickle.load(open('csv.pickle', 'rb'))
# test  = pickle.load(open('testcsv.pickle', 'rb'))


#function1(train1,test1)
#print "saved"
(test,m_res, train, t_res) = function1(train,test)
(m_data, t_data) = function2(train,test)
chiscore = chi2(t_data,t_res)[0]
k = 0
for i in range(len(chiscore)):
    if chiscore[i] >=2:
        k = k+1
print k
(m_data, t_data) = function3(train, test)
#(m_data, t_data) = function2(train1,test1)
#print t_data
#one = []
#neg = []
#zero = []
#for i in range(len(t_res)):
#	if t_res[i]<0:
#		neg.append(i)
#	elif t_res[i] == 0:
#		zero.append(i)
#	else:
#		one.append(i)
#
#temp_l = min(len(zero), len(one), len(neg))
#one = one[:temp_l]
#neg = neg[:temp_l]
#zero = zero[:temp_l]
#t_resp = []
#t_datap = []
#for i in range(temp_l):
#	t_resp.append(t_res[one[i]])
#	t_resp.append(t_res[zero[i]])
#	t_resp.append(t_res[neg[i]])
#	t_datap.append(t_data[one[i]])
#	t_datap.append(t_data[zero[i]])
#	t_datap.append(t_data[neg[i]])
#
#t_res = t_resp
#t_data = t_datap
#
#t_data = t_data[:500]
#t_res = t_res[:500]
#t_data = preprocessing.scale(t_data)
#m_data = preprocessing.scale(m_data)
#m_data = scaler.fit_transform(m_data)
#t_data = scaler.fit_transform(t_data)

#t_res = map(conv, t_res)
#m_res = map(conv, m_res)
#print t_res
print len(m_data)
print len(m_res)
print len(t_data)
print len(t_res)

#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#grid.fit(t_data, t_res)

#print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))
#
clf1 = tree.DecisionTreeClassifier()
print "Training\n"
clf2 = svm.LinearSVC()
clf3 = GaussianNB()
clf4 = NearestCentroid()
clf1.fit(t_data,t_res)
clf2.fit(t_data,t_res)
clf3.fit(t_data,t_res)
clf4.fit(t_data,t_res)

#count = 0
#for el in m_data:
#	clf1.predict([el])
#        clf2.predict([el])
#        clf3.predict([el])
#        clf4.predict([el])
p = 0
n = 0
neg = 0

print m_res.count(1), m_res.count(-1), m_res.count(0), "  ++ \n\n"

for i in range(len(m_data)):
	t = clf4.predict([m_data[i]])
	if t == 1 and m_res[i]==1:
		p += 1
	elif t == -1 and m_res[i] == -1:
		neg += 1
	elif m_res[i] == 0:
		 n += 1
print p, neg, n


print  "desicion tree" , clf1.score(m_data, m_res)
print  "svm",clf2.score(m_data, m_res)
print  "bayesian", clf3.score(m_data, m_res)
print  "centroid", clf4.score(m_data, m_res)
