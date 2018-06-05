
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


from sklearn import datasets


# In[3]:


digits = datasets.load_digits() # loading dataset we wish to work with


# In[4]:


dir(digits) # check to see the directory the the "digits" dataset


# In[5]:


digits['data']


# In[6]:


digits['images'] # pixles stored as an image 
for i in range(10):
    plt.matshow(digits['images'][i]) # a for loop that plots each image 


# In[7]:


digits['target'] # the label of each row 


# ### Part 1

# ### Transformation 1

# In[8]:


def transform(y,digit_number=0):
    return np.array([x == digit_number for x in y]) 


# In[9]:


def trans_1_0(digits):
    digits_0_1 = digits['data']
    digits_0_1 = np.where(digits_0_1[:] > 0, 1,0)
    return digits_0_1


# In[10]:


def LogistcalProcess(digits, digit_number, start = 0):
    y = digits['target']
    index = list(range(len(y))) # specific to python 3 this returns a list object. 
    import random
    random.shuffle(index)
    #X = digits['data']
    #X = trans_1_0(digits) # Transformation 1 to use select rows 7 and 8
    #X = X[:] 
    X = transform_3(digits,start = start) # Transformation 3 to use select rows 9 and 10
    X = X.as_matrix()
    X = np.array([X[index[i]] for i in range(len(y))])
    y = np.array([y[index[i]] for i in range(len(y))])
    n = len(y)/2 
    n = int(n)
    X_train = X[:n]
    X_test = X[n:]
    y_train = transform(y[:n], digit_number)
    y_test = transform(y[n:], digit_number) 
    #allows for great flexability becasue it changes with the number you input
    assert len(X_train) == len(y_train) 
    assert len(X_test) == len(y_test) # makes sure that are the same with assert  
    model = LogisticRegression()
    model.fit(X_train,y_train) 
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    recall = float(tp)/(tp+fn) # recall 
    tnr = float(tn)/(tn+fp) # true negatie rate       
    precision = float(tp)/(tp+fp) # precision
    npv = float(tn)/(tn+fn) # negative predictive power       
    F1 = 2.0/(1.0/precision+1.0/recall)
    F1 = "%1.4f" % F1
    recall ="%1.4f" % recall
    precision ="%1.4f" % precision
    return F1


# In[14]:


def KnnProcess(digits, digit_number, start= 0):
    y = digits['target']
    index = list(range(len(y))) # specific to python 3 this returns a list object. 
    import random
    random.shuffle(index)
    X = digits['data']
    #X = trans_1_0(digits) # Transformation 1 to use select rows 7 and 8 comment out the rest
    #X = X[:] 
    #X = transform_3(digits,start = start) # Transformation 3 to use select rows 9 and 10 comment out the rest
    #X = X.as_matrix()
    X = np.array([X[index[i]] for i in range(len(y))])
    y = np.array([y[index[i]] for i in range(len(y))])
    n = len(y)/2 
    n = int(n)
    X_train = X[:n]
    X_test = X[n:]
    y_train = transform(y[:n], digit_number)
    y_test = transform(y[n:], digit_number)
    return X_train, X_test, y_train, y_test
    #allows for great flexability becasue it changes with the number you input
    assert len(X_train) == len(y_train) 
    assert len(X_test) == len(y_test) # makes sure that are the same with assert  
    model = KNeighborsClassifier()
    model.fit(X_train,y_train) 
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    recall = float(tp)/(tp+fn) # recall 
    tnr = float(tn)/(tn+fp) # true negatie rate       
    precision = float(tp)/(tp+fp) # precision
    npv = float(tn)/(tn+fn) # negative predictive power       
    F1 = 2.0/(1.0/precision+1.0/recall)
    F1 = "%1.4f" % F1
    recall ="%1.4f" % recall
    precision ="%1.4f" % precision
    return F1
    


# In[19]:


X_train, X_test, y_train, y_test = KnnProcess(digits, digit_number = 1, start= 0)
print(len(y_test))


# In[52]:


lst = []
for i in range(10):
    a = LogistcalProcess(digits, digit_number = i, start = 15) 
    a = float(a)
    totals = (a,i)
    lst.append(totals)
print("Transformation 1: This is the minimum Logistical F1 score with its accompanying digit ",str(min(lst)))


# In[53]:


lst = []
for i in range(10):
    a = KnnProcess(digits, digit_number = i, start = 0) 
    a = float(a)
    totals = (a,i)
    lst.append(totals)
print("Transformation 1: This is the minimum KNN F1 score with its accompanying digit ",str(min(lst)))


# ### Transformation 2

# In[26]:


def transform_3(digits, start):
    'This function is meant to subset the data by columns you specify by start and counts up by 8'
    digits_pick = pd.DataFrame(digits['data'])
    digits_pick = digits_pick.iloc[:,start:start+8]
    return digits_pick   


# In[53]:


lst = []
for i in range(10):
    a = LogistcalProcess(digits, digit_number = i, start = 15) 
    a = float(a)
    totals = (a,i)
    lst.append(totals)
print("Transformation 3: This is the minimum Logistical F1 score with its accompanying digit ",str(min(lst)))


# In[52]:


lst = []
for i in range(10):
    a = KnnProcess(digits, digit_number = i, start = 30) 
    a = float(a)
    totals = (a,i)
    lst.append(totals)
print("Transformation 2: This is the minimum KNN F1 score with its accompanying digit ",str(min(lst)))

