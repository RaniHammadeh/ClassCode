
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import random
import math as math
import sklearn
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


df = pd.read_csv("Desktop/CSC478/jobs.csv")


# In[3]:


finder = re.compile('\w\w\w\w+') # split it into 2 parts 


# In[4]:


def corpus_prep(df): 
    X = df['Title']
    corpus = []
    for words in range(len(X)): 
        words = str(X[words].lower())
        corpus.append(finder.findall(words))
    X_train_corpus, X_test_corpus = corpus[:1200], corpus[1200:]
    return X_train_corpus, X_test_corpus, corpus


# In[5]:


X_train_corpus, X_test_corpus, corpus = corpus_prep(df)
#len(X_train_corpus) #1200
#type(X_train_corpus) # list
#print(X_train_corpus)
#len(X_test_corpus) #514
#type(X_test_corpus) # list
#print(X_test_corpus)
#print(corpus)


# In[9]:


len(corpus)


# In[13]:


def flaten_list(corpus):
    flat_list = []
    for sublist in corpus:
        for item in sublist:
            flat_list.append(item.lower())
    return flat_list


# In[20]:


corpus_flat = flaten_list(corpus)
#corpus_flat


# In[21]:


X_test_corpus_flat = flaten_list(X_test_corpus)
#X_test_corpus_flat


# In[22]:


X_train_corpus_flat = flaten_list(X_train_corpus)
#X_train_corpus_flat


# In[17]:


def list_2_dic(doc): # given a document which is a list of words
    d = {}
    words = set(doc)
    for word in words:
        d[word] = doc.count(word)
    
    return d


# In[24]:


d = list_2_dic(X_train_corpus_flat)
#d


# In[26]:


from functools import reduce
corpus_2 = [list_2_dic(doc) for doc in corpus]
words_final = reduce(lambda a,b: a|b, [set(doc.keys()) for doc in corpus_2]) 
#words_final


# In[12]:


from functools import reduce
train_corpus_1 = [list_2_dic(doc) for doc in X_train_corpus]
train_words_final = reduce(lambda a,b: a|b, [set(doc.keys()) for doc in train_corpus_1]) 
#train_words_final


# In[27]:


print("There are", len(train_words_final), "distinct words.")


# In[14]:


from functools import reduce
test_corpus_1 = [list_2_dic(doc) for doc in X_test_corpus]
test_words_final = reduce(lambda a,b: a|b, [set(doc.keys()) for doc in test_corpus_1]) 
#test_words_final


# In[15]:


def words_set(corpus):
    word_set = []  # put in the train/test split here? shuffle too? 
    for sublist in corpus:
        for item in sublist:
            word_set.append(item)
            words = set(word_set)
            words = list(words)
    return words


# In[16]:


test_words = words_set(X_test_corpus)
#len(test_words) # distinct words


# In[17]:


def compute_IDF(corpus, words, min_term_freq= 5, n = 10, m = 50 ):
    idf = {}
    for k, word in enumerate(words):
        d = 0 
        for doc in corpus:
            if word in doc:
                d = d +1      
        if d >=  min_term_freq:
            idf[word] =  math.log(float(len(corpus))/(1+d))
            print(k,word,idf[word]) 
    
    numbers = [[idf[word],word] for word in idf] 
    numbers.sort() 
    Top_10_Words = numbers[-n:]
    Lower_10_Words = numbers[:n]
    Top_50_Words_Num = numbers[-m:]
    Top_10_Words = [word for freq,word in numbers[-n:]]
    Lower_10_Words = [word for freq,word in numbers[:n]]
    Top_50_Words = [word for freq,word in numbers[-m:]]
    return idf, Top_10_Words, Lower_10_Words, Top_50_Words, Top_50_Words_Num 


# In[18]:


idf, Top_10_Words, Lower_10_Words, Top_50_Words, Top_50_Words_Num = compute_IDF(test_corpus_1, train_words_final, min_term_freq= 5, n = 10, m = 50 )
print()
print("Break")
print()
print("These are the top 10 words: ",Top_10_Words)
print()
print("Break")
print()
print("These are the lower 10 words:", Lower_10_Words)
print()
print("Break")
print()
print("These are the top 50 words:", Top_50_Words)


# ## Training Set

# In[19]:


X_train = np.ndarray((len(train_corpus_1),len(Top_50_Words))) # remove cities now?


# In[20]:


X_train.shape


# In[21]:


for i in range(len(train_corpus_1)): # test set here? 
    
    doc1 = train_corpus_1[i]
    
    for j in range(len(Top_50_Words)):
        word = Top_50_Words[j]
        idf_word = idf[word]
        tf = doc1.get(word, 0)
        feature = tf * idf_word
        X_train[i,j] = feature
#print(X_train)


# In[22]:


y_train = df['Salary'][:1200]
#y_train


# In[23]:


df_Train_Features = pd.DataFrame(X_train, columns= ['resources', 'rouge', 'software', 'superintendent', 'support', 'tampa', 'therapist', 'upscale', 'west', 'wireless', 'alto', 'audit', 'bloomington', 'clearwater', 'clinical', 'company', 'computer', 'construction', 'control', 'corporate', 'database', 'english', 'estimator', 'facilities', 'group', 'growth', 'inside', 'insurance', 'japanese', 'medical', 'miami', 'molding', 'newport', 'office', 'packaging', 'palo', 'philadelphia', 'physical', 'physician', 'plastic', 'private', 'professional', 'program', 'programmer', 'recruiter', 'regional', 'services', 'shift', 'territory', 'with'])


# In[24]:


#df_Train_Features


# In[25]:


df_Train_Title = pd.DataFrame(df['Title'][:1200]) 


# In[26]:


#df_Train_Title


# In[27]:


df_Train_Salary = df['Salary']
df_Train_Salary = pd.DataFrame(df_Train_Salary[:1200])
#df_Train_Salary


# In[28]:


df_Training = pd.concat([df_Train_Features, df_Train_Salary], axis=1)


# In[29]:


X_train = df_Training.drop(['Salary'], axis = 1)
X_train
y_train = df_Training['Salary']
#y_train


# ### Test Set

# In[30]:


X = np.ndarray((len(test_corpus_1),len(Top_50_Words))) # remove cities now?


# In[31]:


X.shape


# In[32]:


for i in range(len(test_corpus_1)): # test set here? 
    
    doc = test_corpus_1[i]
    
    for j in range(len(Top_50_Words)):
        word = Top_50_Words[j]
        idf_word = idf[word]
        tf = doc.get(word, 0)
        feature = tf * idf_word
        X[i,j] = feature
#print(X)


# In[33]:


df2_features = pd.DataFrame(X, columns= ['resources', 'rouge', 'software', 'superintendent', 'support', 'tampa', 'therapist', 'upscale', 'west', 'wireless', 'alto', 'audit', 'bloomington', 'clearwater', 'clinical', 'company', 'computer', 'construction', 'control', 'corporate', 'database', 'english', 'estimator', 'facilities', 'group', 'growth', 'inside', 'insurance', 'japanese', 'medical', 'miami', 'molding', 'newport', 'office', 'packaging', 'palo', 'philadelphia', 'physical', 'physician', 'plastic', 'private', 'professional', 'program', 'programmer', 'recruiter', 'regional', 'services', 'shift', 'territory', 'with'])


# In[34]:


#df2_features 


# In[35]:


df3 = df['Salary']
df3 = pd.DataFrame(df3[1200:])
df3
df3 = df3.reset_index()
df3
df3_salary = df3.drop(['index'], axis= 1 )
#df3_salary


# In[36]:


df_predicions = pd.concat([df2_features, df3_salary], axis=1)
#df_predicions


# In[37]:


X_test = df_predicions.drop(['Salary'], axis = 1)
X_test
y_test = df_predicions['Salary']


# ### Linear Model

# In[38]:


lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)


# In[39]:


plt.scatter(y_test,y_pred)


# In[40]:


from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test,y_pred)
MSE = metrics.mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
RMSE


# In[41]:


cor_df = df_predicions.corr() 

cor_df #upscale, computer programmer 
cor_df_words = cor_df.index
cor_df_words = pd.DataFrame(cor_df_words, columns= ['Words'])
type(cor_df_words)


# In[42]:


#cor_df.index


# In[43]:


top_cor = np.where(cor_df['Salary'] > .05, cor_df['Salary'], 0)
top_cor = pd.DataFrame(top_cor, columns= ['Correlation'])
top_cor = pd.concat([cor_df_words,top_cor], axis =1)
top_cor = top_cor.loc[top_cor['Correlation'] > .05]
top_cor # words associated with the highest salary 


# In[44]:


bottom_cor = np.where(cor_df['Salary'] < .05, cor_df['Salary'], 0)
bottom_cor = pd.DataFrame(bottom_cor, columns= ['Correlation'])
bottom_cor = pd.concat([cor_df_words,bottom_cor], axis =1)
bottom_cor = bottom_cor.loc[bottom_cor['Correlation'] < 0]
bottom_cor # words asscoiated with a lower salary 

