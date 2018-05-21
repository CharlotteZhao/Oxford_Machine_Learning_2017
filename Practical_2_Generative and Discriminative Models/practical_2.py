import _pickle as cp
import numpy as np
import plotly
import matplotlib.pyplot as plt
import plotly.plotly as py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import operator
from sklearn.model_selection import KFold
import math
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
#data
'''
iris = load_iris()
X, y = iris['data'], iris['target']
'''
'''
X = np.array([[0,3,1], [1,3,1],[0,1,0],[2,3,1]])
y = np.array([0,1,2,3])
x_predict = np.array([2,3,1])
'''



#X, y = cp.load(open('voting-full.pickle', 'rb'))
#with open("voting_ucl.pickle", "rb") as f:
   #X,y= pickle.load(f)

with open("pima.pickle", "rb") as f:
    X,y= pickle.load(f)

#class

class NBC():
    num_classes = 0
    feature_types = []
    K = 4 # number of categories
    num_y_c = 0 #number of data points that have the same label
    N = 0
    D = 0
    def __init__(self, feature_types, num_classes):
        self.num_classes = num_classes
        self.feature_types = feature_types
        self.N = Xtrain.shape[0]
        self.D = Xtrain.shape[1]
        self.theta = np.zeros((num_classes,self.D,self.K))  #parameter
        self.first = np.zeros((self.num_classes))
        self.num_y_c = np.zeros((num_classes))
        self.pi_c = np.zeros((num_classes))

    def fit(self, X,y):
        num_xi = np.zeros(self.D)  # number of same feature within these data points
        #first part of log p (D|self.theta, pi)
        self.N = X.shape[0]
        self.D = X.shape[1]
        for y_predict in range(0, self.num_classes):  # for every possible class label

            for i in range(0, self.N):
                if (y_predict == y[i]):  # y = c -> pi_c
                    self.num_y_c[y_predict] = self.num_y_c[y_predict] + 1
            self.pi_c[y_predict] = self.num_y_c[y_predict] / self.N

            if self.pi_c[y_predict] != 0:
                self.first[y_predict] =  math.log(self.pi_c[y_predict])
            else:
                self.first[y_predict] =  math.log(10**(-6))



        #second part of log p (D|self.theta, pi)
        for y_predict in range (0,self.num_classes): #for every possible class label
            mu = [0] * (self.D)
            sig = [0] * (self.D)
            #for every feature
            for feature in range (0,self.D):
                #find the number of data points in the training set that has the same label

            #compute the probability of this feature in those data points
                if self.feature_types[feature] == 'r': #real number
                    temp = []

                    #compute mean and variance
                    for data in range(0, self.N):
                        #if y[data] == y_predict & bool(X[data,feature] != np.float64(2)): #if not empty
                        if y[data] == y_predict:  # not consider empty

                            temp.append(X[data, feature])

                    if (len(temp) == 0):
                        mu[feature] = 9999
                    else:
                        mu[feature] = np.mean(temp)
                        sig[feature] = np.var(temp)
                        if sig[feature] < 10 ** (-6):
                            sig[feature] = 10 ** (-6)


                    self.theta[y_predict][feature][0] = mu[feature]
                    self.theta[y_predict][feature][1] = sig[feature]

                alpha = 1

                if  self.feature_types[feature] == 'b': #binary
                    #compute the probability of 1 given class label
                    for data in range(0, self.N):
                        #if y[data] == y_predict & bool(X[data,feature] != np.float64(2)): #if not empty
                        if y[data] == y_predict: #not consider empty

                            if X[data][feature] == 1:
                                self.theta[y_predict][feature][0] = self.theta[y_predict][feature][0] + 1
                if self.feature_types[feature] == 'c': #categorical
                    #compute the probability of each category given class label
                    for data in range(0, self.N):

                        #if y[data] == y_predict & bool(X[data,feature] != np.float64(2)):#if not empty
                        if y[data] == y_predict: # not consider empty

                            for category in range (0,self.K):
                                if X[data][feature] == category:
                                     self.theta[y_predict][feature][category] = self.theta[y_predict][feature][category] + 1
                if self.feature_types[feature] == 'b':  # binary

                    self.theta[y_predict][feature][0] = (self.theta[y_predict][feature][0]  + alpha )/ (self.num_y_c[y_predict] + 2*alpha)
                #print (self.num_y_c[y_predict])
                if self.feature_types[feature] == 'c':  # categorical
                    for category in range(0, self.K):

                        self.theta[y_predict][feature][category] = (self.theta[y_predict][feature][category] + alpha) / (self.num_y_c[y_predict] + self.K * alpha)
    def Predict(self,X_test):

        y_test_predict = np.zeros((N-Ntrain))
        num_error = 0
        for data in range (0,N-Ntrain):
            y_test_predict[data] = self.predict(X_test[data])
        return y_test_predict

    def predict(self, x_predict):
        N = X.shape[0]
        D = X.shape[1]
        second = np.zeros(self.num_classes)
        prob = np.zeros(self.num_classes)  # probability of each class label
        prob_xi = np.zeros(D) # prob of same feature within these data points
        for y_predict in range(0, self.num_classes):  # for every possible class label

            for feature in range (0,self.D):
                #if x_predict[feature] != 2: #if not empty
                if (True): #not consider empty
                    if self.feature_types[feature] == 'r':
                        #print(self.theta[y_predict][feature][1])
                        #print(math.exp(-np.power(x_predict[feature] - self.theta[y_predict][feature][0], 2.)/ (2 * self.theta[y_predict][feature][1])))
                        if self.theta[y_predict][feature][0] == 9999:
                            prob_xi[feature] = 0
                        else:
                            prob_xi[feature] = math.log(1 / np.sqrt(2 * math.pi * self.theta[y_predict][feature][1])) + (-np.power(x_predict[feature] - self.theta[y_predict][feature][0], 2.) / (2 * self.theta[y_predict][feature][1]))
                    if self.feature_types[feature] == 'c':
                        if (self.theta[y_predict][feature][x_predict[feature]] != 0):
                            prob_xi[feature] = math.log(self.theta[y_predict][feature][x_predict[feature]])
                        else:
                            prob_xi[feature] = math.log(10 ** (-6))


                    if self.feature_types[feature] == 'b':
                        #print('theta',y_predict,self.theta[y_predict][feature][0])
                        t = self.theta[y_predict][feature][0] *(x_predict[feature]) + (1-self.theta[y_predict][feature][0]) * (1-x_predict[feature])
                        if (t != 0):
                            prob_xi[feature] = math.log(t)
                        else:
                            prob_xi[feature] = math.log(10 ** (-6))

                second[y_predict] = second[y_predict] + prob_xi[feature]



         # sum of the first part and the second part
        for y_predict in range(0, self.num_classes):  # for every possible class label
            #print(second)
            prob[y_predict] = self.first[y_predict] + second[y_predict]
        #print (self.first)

        # find maximum log likelihood
        mll = prob[0]
        class_label = 0
        for y_predict in range(0, self.num_classes):  # for every possible class label
            if mll < prob[y_predict]:
                mll = prob[y_predict]
                class_label = y_predict

        return class_label


def find_mean(X,y,p,q,N,D):
    temp = []
    for i in range (0,N):
        if (y[i] == y[p]): #data of the same label in the whole data set
            temp.append(X[i,q])

    return np.mean(temp)




#NBC test
# nbc.fit(Xtrain,ytrain)


N, D = X.shape
Ntrain = int(0.8 * N)
num_iteration = 200
num_classifier = 10
classification_error_nbc = np.zeros((num_classifier))
classification_error_lr = np.zeros((num_classifier))

m_nbc = np.zeros((num_classifier,num_iteration))
m_lr = np.zeros((num_classifier,num_iteration))
for i in range (0,num_iteration):
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]

    #print(ytrain[:12])
    #print('sh',shuffler[:12])
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    for classifier in range(0,num_classifier):

        # iris
        #nbc = NBC(feature_types=['r','r','r','r'],num_classes=3)
        #voting
        #nbc = NBC(feature_types=['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'], num_classes=2)
        #pima
        nbc = NBC(feature_types=['r', 'r', 'r', 'r','r','r','r','r'], num_classes=2)

        nbc.fit(Xtrain[:int(Ntrain*0.1*(classifier+1))],ytrain[:int(Ntrain*0.1*(classifier+1))])
        yhat = nbc.Predict(Xtest)
        m_nbc[classifier][i] = np.mean(yhat != ytest)


    #full data set
    # preprocessing

    '''
    for p in range (0,Ntrain):
        for q in range (0,D):
            if Xtrain[p][q] == 2: #empty
                #Xtrain[p][q] = find_mean(X,y,p,q,N,D) #1: mean
                Xtrain[p][q] = 0 #2: use 0

    for p in range(0,N-Ntrain):
        for q in range(0, D):
            if Xtest[p][q] == 2:  # empty
                #Xtest[p][q] = find_mean(X,y,p,q,N,D)
                Xtest[p][q] = 0 #2: use 0
    '''
    #a hot coding
    '''
    enc = OneHotEncoder()
    enc.fit(X[shuffler])
    Xtrain = enc.transform(Xtrain).toarray()
    Xtest = enc.transform(Xtest).toarray()
    '''

    for classifier in range(0, num_classifier):
        # logistic regression
        # iris
        #LR = LogisticRegression(C=5,multi_class='multinomial',solver = 'newton-cg')
        # voting
        #LR = LogisticRegression(C=5, multi_class='multinomial', solver='newton-cg')
        # pima
        LR = LogisticRegression(C=5, multi_class='ovr')

        LR.fit(Xtrain[:int(Ntrain * 0.1 * (classifier + 1))], ytrain[:int(Ntrain * 0.1 * (classifier + 1))])
        yhat = LR.predict(Xtest)
        m_lr[classifier][i] = np.mean(yhat != ytest)





for classifier in range(0,num_classifier):
    classification_error_nbc[classifier] = np.mean(m_nbc[classifier])
    classification_error_lr[classifier] = np.mean(m_lr[classifier])

x = np.arange(0,num_classifier)
plt.figure()
plt.plot(x,classification_error_nbc,'--',label='NBC_classification_error')
plt.plot(x,classification_error_lr,label='LR_classification_error')

plt.xlabel('num_classifier')
plt.ylabel('classification_error')
plt.ylim([0.2,0.4])
plt.title('pima')
plt.legend()
plt.show()


#read data#

import numpy as np
import pickle

#pima

data = np.loadtxt("pima.txt", delimiter=',', skiprows=0,dtype=float)
X = data[:,0:7]
y = data[:,8]
'''
N,D = a.shape
X = np.zeros((N,D))
y = np.zeros(N)
for i in range(0,N):
    for j in range (0,D):
        if a[i][j] == 'y':
            X[i][j] = 1
        if a[i][j] == 'n':
            X[i][j] = 0
        if a[i][j] == '?':
            X[i][j] = 2


for i in range (0,N):
    if b[i] == 'republican':
        y[i] = 0
    if b[i] == 'democrat':
        y[i] = 1
'''



with open("pima.pickle", "wb") as f:
    pickle.dump((X,y), f)

print(X)
print(y.shape)