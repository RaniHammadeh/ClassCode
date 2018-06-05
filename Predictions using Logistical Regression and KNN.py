
# coding: utf-8

# In[1]:


# Assignment 1 - Rani Hammadeh


# In[3]:


import pandas as pd
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


# In[4]:


df = pd.read_csv("Desktop/CSC478/assignment1.csv")


# In[5]:


df.head(10)


# In[6]:


df.drop("Unnamed: 0", axis = 1, inplace = True)


# In[7]:


df.head(10)


# In[8]:


C


# In[9]:


df.hist(column = "feature-1")


# In[10]:


df.hist(column = "feature-2")


# In[11]:


df.hist(column = "feature-3")


# In[12]:


df.hist(column = "feature-4")


# In[12]:


# There appears to be outliers for each feature


# In[13]:


df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


df.info()


# In[16]:


df[df['feature-0'] == 1000]


# In[17]:


df[df['feature-1'] == 1000]


# In[18]:


df[df['feature-2'] == 1000]


# In[19]:


df[df['feature-3'] == 1000]


# In[20]:


df[df['feature-4'] == 1000]


# In[21]:


'''There is a max value of 1000 for each feature I will drop values of 1000
and rerun the disribution, mean, and standard deviation''' 


# In[15]:


df2 = df[df[:] < 1000]


# In[23]:


# check to see the spread after having dropped outliers


# In[ ]:


df2.hist()


# In[24]:


df2.hist(column = "feature-0")


# In[25]:


df2.hist(column = "feature-1")


# In[26]:


df2.hist(column = "feature-2")


# In[27]:


df2.hist(column = "feature-3")


# In[28]:


df2.hist(column = "feature-4")


# In[29]:


df2.describe()


# In[30]:


df2.mean() # The mean of each feature can be found below


# In[31]:


df2.std() # The standard deviation of each feature can be found below


# In[32]:


df2.corr() 


# In[33]:


''' Question 1: The data above does appear to 
have a weak to small correlation with the target variable "y" but it nonexsist between features  '''
''' Question 2: A small correlation does not imply the features are indpendent, 
it just indicates that they do not
have a liner relationship and points to a non-linear relationship'''


# In[34]:


# Imputing missing values


# In[35]:


df.isnull().any() # there are missing values for each feature


# In[36]:


df2 = df2.sample(frac=1) # shuffle the data 


# In[37]:


df2.head(10)


# In[38]:


# Train/Test split


# In[39]:


df_features = pd.DataFrame(df2.drop("y", axis = 1)) 


# In[40]:


df_features.head(10)


# In[41]:


df_y = pd.DataFrame(df2["y"]) 


# In[42]:


df_y.head(10)


# In[43]:


X, y = df_features, df_y


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .50, random_state = 101)


# # KNN Classifier 

# In[45]:


impute = Imputer()
impute.fit(X_train)
X_train = impute.transform(X_train)
X_test = impute.transform(X_test)


# In[46]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[97]:


KNNmodel = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNNmodel.fit(X_train, y_train.values.ravel())
y_pred = KNNmodel.predict(X_test)


# In[48]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[100]:


KNNmodel.metric()


# In[49]:


tn


# In[50]:


fp


# In[51]:


fn


# In[52]:


tp


# In[53]:


print(confusion_matrix(y_test,y_pred)) 


# In[54]:


TruePostiveRate = float(tp)/(tp + fn)
TruePostiveRate


# In[55]:


TrueNegativeRate = float(tn)/(tn + fp)
TrueNegativeRate


# In[56]:


Precision = float(tp)/(tp + fp)
Precision


# In[57]:


NegativePredictivePower = float(tn)/(tn + fn)
NegativePredictivePower 


# In[58]:


F_1_Score = 2.0/(1.0/Precision+1.0/TruePostiveRate)
F_1_Score 


# In[59]:


print(classification_report(y_test,y_pred))


# # Logistic Regression Classifier

# In[60]:


impute = Imputer()
impute.fit(X_train)
X_train = impute.transform(X_train)
X_test = impute.transform(X_test)


# In[61]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[62]:


Logmodel = LogisticRegression()
Logmodel.fit(X_train, y_train.values.ravel())
y_pred = Logmodel.predict(X_test)


# In[63]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[64]:


tn


# In[65]:


fp


# In[66]:


fn


# In[67]:


tp


# In[68]:


print(confusion_matrix(y_test,y_pred)) 


# In[69]:


TruePostiveRate = float(tp)/(tp + fn)
TruePostiveRate


# In[70]:


TrueNegativeRate = float(tn)/(tn + fp)
TrueNegativeRate


# In[71]:


Precision = float(tp)/(tp + fp)
Precision


# In[72]:


NegativePredictivePower = float(tn)/(tn + fn)
NegativePredictivePower


# In[73]:


F_1_Score = 2.0/(1.0/Precision+1.0/TruePostiveRate)
F_1_Score


# In[74]:


print(classification_report(y_test,y_pred))


# # Logistc Regression without removing outliers

# In[75]:


df3 = df.sample(frac=1) # shuffle the data 


# In[76]:


df3.head(10)


# In[77]:


df3_features = pd.DataFrame(df3.drop("y", axis = 1)) 


# In[78]:


df3_y = pd.DataFrame(df3["y"]) 


# In[79]:


X, y = df3_features, df3_y


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .50, random_state = 101)


# In[81]:


impute = Imputer()
impute.fit(X_train)
X_train = impute.transform(X_train)
X_test = impute.transform(X_test)


# In[82]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[83]:


Log2model = LogisticRegression()
Log2model.fit(X_train, y_train.values.ravel())
y_pred2 = Logmodel.predict(X_test)


# In[84]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred2).ravel()


# In[85]:


tn


# In[86]:


fp


# In[87]:


fn


# In[88]:


tp


# In[89]:


print(confusion_matrix(y_test,y_pred2)) 


# In[90]:


TruePostiveRate = float(tp)/(tp + fn)
TruePostiveRate


# In[91]:


TrueNegativeRate = float(tn)/(tn + fp)
TrueNegativeRate


# In[92]:


Precision = float(tp)/(tp + fp)
Precision


# In[93]:


NegativePredictivePower = float(tn)/(tn + fn)
NegativePredictivePower


# In[94]:


F_1_Score = 2.0/(1.0/Precision+1.0/TruePostiveRate)
F_1_Score


# In[95]:


print(classification_report(y_test,y_pred))


# In[96]:


'''Question 3: The results are better if we remove outliers. 
This is becasue the outliers have an impact on the 
decision boundry of how a given data point is sorted; leading to 
a poor model'''

