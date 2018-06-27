
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from scipy import stats
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Desktop/CSC478/hour.csv")


# In[3]:


df2 = df[['season', 'yr', 'mnth', 'hr',  'weekday', 'workingday', 
          'weathersit', 'atemp', 'hum', 'windspeed', 'cnt']]


# In[4]:


df.head()


# In[9]:


df.hist('atemp')


# In[10]:


plt.scatter(df['atemp'],df['cnt'])


# In[4]:


df2['hr'] = (df2['hr'] + 1) # I am adding by 1 to avoide the divide by zero error
df2['Cnt/Hr'] = df2['cnt']/df2['hr']


# In[13]:


reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')


# In[22]:


df3 = df[df['cnt'] > 80].sample(100)


# In[24]:


plt.scatter(df3['cnt'],df3['atemp'])


# In[5]:


df2.head()


# In[7]:


df2.describe()


# In[ ]:


sns.distplot(df2['Cnt/Hr']) # Cnt/Hr skewes heavily to the right 


# In[ ]:


'it does not appear the data correlates much'
sns.heatmap(df2.corr())


# In[ ]:


def week_graph(df, day = 0):
    lst = []
    lst2 = []
    x = df
    for i in range(1,25):
        weekday = x.loc[(x['workingday'] == day) & 
                        (x['hr'] == [i])] 
        cnthr = weekday['Cnt/Hr']
        avg = cnthr.mean()
        lst.append(avg)
        avg = pd.DataFrame(lst, columns= ['Averages'])
    for i in range(24):
        lst2.append(i)
        num = pd.DataFrame(lst2, columns= ['Hour'])
        num['Hour'] = (num['Hour'] + 1)    
    avgbyhour = pd.concat([num,avg], axis = 1)
    avgbyhour
    if day == 1:
        plt.title("Cnt/Hr for Rainy Day")
        plt.xlabel("Hour")
        plt.ylabel("Cnt/Hr")
        plt.title("Cnt/Hr for Weekday Day")
        return plt.bar(avgbyhour['Hour'], avgbyhour['Averages'])
    else:
        plt.title("Cnt/Hr for Rainy Day")
        plt.xlabel("Hour")
        plt.ylabel("Cnt/Hr")
        plt.title("Cnt/Hr for Non-WeekdayDay" )
        return plt.bar(avgbyhour['Hour'], avgbyhour['Averages'])


# In[ ]:


def rain_day(df, option_1 = None, option_2 = None):
    rainlist1 = []
    rainlist2 = []
    x = df
    for i in range(1,25):
        rain = x.loc[(x['weathersit'] == option_1) | 
                     (x['weathersit'] == option_2)]
        rain = rain.loc[rain['hr'] == [i]]
        cnthr = rain['Cnt/Hr']
        avg = cnthr.mean()
        rainlist1.append(avg)
        avg = pd.DataFrame(rainlist1, columns= ['Averages'])
    for i in range(24):
        rainlist2.append(i)
        num = pd.DataFrame(rainlist2, columns= ['Hour'])
        num['Hour'] = (num['Hour'] + 1)
    avgbyhour = pd.concat([num,avg], axis = 1)
    
    if option_1 == 3 and option_2 == 4:
        plt.title("Cnt/Hr for Rainy Day")
        plt.xlabel("Hour")
        plt.ylabel("Cnt/Hr")
        return plt.bar(avgbyhour['Hour'], avgbyhour['Averages'])
    else:
        plt.title("Cnt/Hr for Rainy Day")
        plt.xlabel("Hour")
        plt.ylabel("Cnt/Hr")
        plt.title("Cnt/Hr for No Rain Day" )
        return plt.bar(avgbyhour['Hour'], avgbyhour['Averages'])


# In[ ]:


weekday = week_graph(df2, day = 1)


# In[ ]:


nonweekday = week_graph(df2, day = 0)


# In[ ]:


rainday = rain_day(df2,option_1= 3, option_2 = 4)


# In[ ]:


nonrainday = rain_day(df2,option_1= 1, option_2 = 2)


# In[ ]:


'''
The spread of the data above suggest that there are outliers
in the early morning. Before we run a model we must determine what
an outlier is, how we caculate it, and finally remove it.
'''


# In[ ]:


# How I removed outliers 

'''
To see how many outliters there were in the dataset I first ran the data through
boxplots by hr and weekday or not and hour and rain or no rain day to get a 
visulization of the outtiers. Then,given that we were asked to plot bike rentals 
by hour and then by whether it was a weekday or notand a rainy day or not I decide 
to remove outliters in a conditonal manner. To avoid complications I created 2 
dataframes below. One that if it was a weekday and the a second one if it was not. 
To remove the outliters I caclualted the z-score of each hour of the weekday and 
nonweekday dataframes. I then set a threshold of 3 and removed and row whos hour 
exceeded that threshold. I followed the same process with the rain and no rain 
day dataframes. If a row had a score of a 1 or 2 in the 'weatherits' column it 
was conisdered a non rainy day and if a row had 3 or 4 it was scored as a rainy day.
For additional analysis I also caclulated a general z-score where I ran a z-score
on the entire dataframe and removed rows that had a higher z-score than 3. 

'''


# In[ ]:


def boxplot_week(df, day = 0):
    a = 0
    lst = []
    x = df
    for i in range(1,25):
        a += 1
        weekday = x.loc[(x['workingday'] == day) &
                        (x['hr'] == [i])]
        hour_1 = weekday['hr']
        cnt_1 = weekday['cnt']
        hrdf = pd.DataFrame(hour_1, columns= ['Hour'])
        cntdf = pd.DataFrame(cnt_1, columns= ['Cnt'])
        cnt_1 = cnt_1/hour_1
        lst.append(cnt_1)
    return plt.boxplot(lst)
print()


# In[ ]:


def rainday_boxplot(df, opt1 = None, opt2 = None):
    a = 0
    lst = []
    x = df
    for i in range(1,25): 
        a += 1
        rain = x.loc[(x['weathersit'] == opt1) | 
                     (x['weathersit'] == opt2)]
        hour = rain[rain['hr'] == a ]
        hour_1 = hour['hr'] 
        cnt_1 = hour['cnt']
        hrdf = pd.DataFrame(hour_1, columns= ['Hour'])
        cntdf = pd.DataFrame(cnt_1, columns= ['Cnt'])
        cnt_1 = cnt_1/hour_1
        lst.append(cnt_1)
        cnt_1df = pd.DataFrame(cnt_1, columns= ['Cnt/Hr'])
    return plt.boxplot(lst)
print()  


# In[ ]:


weekdaybox = boxplot_week(df2, day = 1)


# In[ ]:


nonweekdaybox = boxplot_week(df2, day = 0)


# In[ ]:


nonraindayboxplot = rainday_boxplot(df2, opt1 = 1, opt2 = 2)


# In[ ]:


raindayboxplot = rainday_boxplot(df2, opt1 = 3, opt2 = 4)


# In[ ]:


'''
As you can see from the boxplots above, a number of outlier need to be 
removed per hour.
'''


# In[8]:


#below is a for loop for removing outliers by hour for weekdays
lst = [] 
lstw = []
for i in range(1,25):
    workingday_1 = df2[(df2['workingday'] == 1) & (df2['hr'] == [i])]
    mean_frame_wrkday = np.mean(workingday_1['Cnt/Hr'])
    std_frame_wrkday = np.std(workingday_1['Cnt/Hr'])
    x_wrk = workingday_1['cnt']
    z_score_wrk = (workingday_1['Cnt/Hr'] - 
                   workingday_1['Cnt/Hr'].mean())/workingday_1['Cnt/Hr'].std(ddof=0)
    workingday_1['z_score'] = z_score_wrk
    lstw.append(workingday_1)
    trial1 = pd.concat(lstw)
    workingday_1_no_out = workingday_1[~(workingday_1['z_score'] >= 3)]
    lst.append(workingday_1_no_out)
    workingday_1_no_out = pd.concat(lst)


# In[9]:


#below is a for loop for removing outliers by hour for nonweekdays
lst2 = []
lst2w = []
for i in range(1,25):
    workingday_0 = df2[(df2['workingday'] == 0) & (df2['hr'] == [i])]
    mean_frame_Nwrkday = np.mean(workingday_0['Cnt/Hr'])
    std_frame_Nwrkday = np.std(workingday_0['Cnt/Hr'])
    x_Nwrk = workingday_0['cnt']
    z_score_Nwrk = (workingday_0['Cnt/Hr'] - 
                    workingday_0['Cnt/Hr'].mean())/workingday_0['Cnt/Hr'].std(ddof=0)
    workingday_0['z_score'] = z_score_Nwrk
    lst2w.append(workingday_0)
    trial2 = pd.concat(lst2w)
    workingday_0_no_out = workingday_0[~(workingday_0['z_score'] >= 3)]
    lst2.append(workingday_0_no_out)
    workingday_0_no_out = pd.concat(lst2)


# In[10]:


# below is the final dataframe for outliers removed by hour and by weekday or nonweekday
weekday_df = pd.concat([workingday_1_no_out,workingday_0_no_out])
weekday_df.head() 


# In[11]:


#below is a for loop for removing outliers by hour for no rain days.
norainlst = []
norainlstw = []
for i in range(1,25):
    norain = df2.loc[(df2['weathersit'] == 1) | (df2['weathersit'] == 2)] 
    norain = norain[norain['hr'] == i]
    z_score_wrk = (norain['Cnt/Hr'] - 
                   norain['Cnt/Hr'].mean())/norain['Cnt/Hr'].std(ddof=0)
    norain['z_score'] = z_score_wrk
    norainlstw.append(norain)
    trial1 = pd.concat(norainlstw)
    norain_no_out = norain[~(norain['z_score'] >= 3)]
    norainlst.append(norain_no_out)
    norain_noout = pd.concat(norainlst)


# In[12]:


#below is a for loop for removing outliers by hour for rain days.
rainlst = []
rainlstw = []
for i in range(1,25):
    rain = df2.loc[(df2['weathersit'] == 3) | (df2['weathersit'] == 4)] 
    rain = rain[rain['hr'] == i ]
    z_score_wrk = (rain['Cnt/Hr'] - 
                   rain['Cnt/Hr'].mean())/rain['Cnt/Hr'].std(ddof=0)
    rain['z_score'] = z_score_wrk
    rainlstw.append(rain)
    trial1 = pd.concat(rainlstw)
    rain_no_out = rain[~(rain['z_score'] >= 3)]
    rainlst.append(rain_no_out)
    rain_noout = pd.concat(rainlst)


# In[13]:


# below is the final dataframe for outliers removed by hour and by rainy day or non rainy day
rain_df = pd.concat([norain_noout,rain_noout])
rain_df.head()


# In[14]:


#  General outlier remover 
df3 = df2
z_score = (df3['Cnt/Hr'] - 
           df3['Cnt/Hr'].mean())/df3['Cnt/Hr'].std(ddof=0)
df3['z_score'] = z_score
general_out = df3[~(df3['z_score'] >= 3)]


# In[ ]:


general_out.head()


# ### Linear Model: Cnt

# In[ ]:


Linear_Df = rain_df
Linear_Df = Linear_Df.drop(['Cnt/Hr','z_score'], axis = 1)


# In[ ]:


df_X = Linear_Df[['season', 'yr', 'mnth', 'hr', 'weekday', 'workingday', 
                  'weathersit','atemp', 'hum', 'windspeed',]]
df_Y = Linear_Df['cnt']


# In[ ]:


df_X2 = Linear_Df[['season', 'yr', 'hr', 'weekday', 
                   'workingday', 'atemp', 'hum', 'windspeed',]]
df_Y2 = Linear_Df['cnt'] # removing non signifcant features: mnth, and weathersit


# In[ ]:


# note: to view model that have removed non-significant featuers please swith the comments out
X , y = df_X, df_Y
#X_2, y_2 = df_X2, df_Y2 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size=0.30, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.30, random_state=42)


# In[ ]:


LinearPipeline = Pipeline([
    ('imputer', Imputer()),
    ('scaler',StandardScaler()),
    ('LinearModel',LinearRegression())  
])


# In[ ]:


LinearPipeline.fit(X_test,y_test)


# In[ ]:


y_pred = LinearPipeline.predict(X_test)


# In[ ]:


LRmodel = LinearPipeline.get_params('LinearModel')['LinearModel'].coef_ 
X.columns = 'season', 'yr', 'mnth', 'hr', 'weekday', 'workingday', 
'weathersit','atemp', 'hum', 'windspeed'
coef = pd.DataFrame(LRmodel, X.columns, columns = ['Coeff']) 
coef


# In[ ]:


'''
From coeffient scores, the model is telling us that "atemp" (55.91) 
the temperature of a given day,'hr' (52.99) the time of day,'yr', 
(38.8) and 'season' (20.3) have the greatest impact on the prediction 
of cnt with temperature, a coeffient of 55.91, having the single greatest 
impact on the cnt prediction.For example, a 1 unit increase in temperature 
leads to an additonal 55 bikes being rented out for that specific hour. 
'''


# In[ ]:


plt.scatter(y_test,y_pred)


# In[ ]:


MAE = metrics.mean_absolute_error(y_test,y_pred)
MSE = metrics.mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
RMSE # This is a score of how much error there is in the predicton, lower is better


# In[ ]:


#source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est = est.fit()
print(est.summary())


# In[ ]:


'''
From coeffient scores, the model is telling us that "atemp" (55.91) 
the temperature of a given day,'hr' (52.99) the time of day,'yr', 
(38.8) and 'season' (20.3) have the greatest impact on the prediction 
of cnt with temperature, a coeffient of 55.91, having the single greatest 
impact on the cnt prediction.For example, a 1 unit increase in temperature 
leads to an additonal 55 bikes being rented out for that specific hour. 
'''


# In[ ]:


'''
For the linear regression model it appears that this model may not be the 
optimal choice to derive insights from. An RMSE score of 132 and an R^2 of 
.40 was the best I could atain. This is after I removed outliers in a general 
sense. Meaning that I did not use any conditions to calulate any specific z_score. 
However, I have concerns over this data and large outiers in hour 1 would effect 
the z_scores of hours during the working day 9-5 when large numbers are more common. 
This made me think that too much important data was being removed and thus decided 
to go with the next best model which removed outliers condtionally by whether or not 
it was a rainy day. This model returned an RMSE of 137 and have an R^2 score of .30. 

I ran 2 aditonal models as well. One model was run after I conditonally 
removed outliters by hour and whether it was a weekday or not this yield a 
RMSE of 139. The second I removed non-signifcant features from the model, 
those they had a higher p-value than .05, (mnths and weathersit) and this 
showed little to no improvment in model peformence yielding an RMSE of 137 
and an R^2 of .38.  
'''


# ### KNN: Rainy Day

# In[15]:


# Below is the 1st KNN model that predicts rain or no rain based off of bike rental activity and other factors. 
df_KNN = rain_df
df_KNN = df_KNN.drop(['Cnt/Hr','z_score'], axis = 1)
df_KNN.head()


# In[16]:


# series map method
'''
Since the KNN model predicts binary values 0 or 1 I
am transforming the 'weathersit' feature to one. A value
of 1 or 2 in this feature means it was not a rainy day
and one of 3 or 4 means it was a rainy day. I recoded
this feature and created a new feature call 'Rain' to reflect this.
'''
df_KNN['Rain'] = df_KNN.weathersit.map({1:0,2:0,3:1,4:1}) # no rain = 0 rain = 1
df_KNN.head()


# In[17]:


dfKNN_X = df_KNN[['season', 'yr', 'mnth', 'hr', 'weekday','workingday','cnt']]
dfKNN_Y = df_KNN['Rain']


# In[18]:


'''
Below we are splitting the data into train and test splits. The train data 
will be used to train the KNN model so it could learn about patterns within
the data that will later be used to form predictions on the test data.
'''
Knn_X , Knn_y = dfKNN_X, dfKNN_Y
X_train, X_test, y_train, y_test = train_test_split(Knn_X,  Knn_y, test_size=0.30, random_state=42)


# In[19]:


KNNPipeline = Pipeline([
    ('imputer', Imputer()),
    ('scaler',StandardScaler()),
    ('KNNModel',KNeighborsClassifier(n_neighbors= 3,
                                     metric='manhattan'))  
])


# In[20]:


KNNPipeline.fit(X_test,y_test)
y_pred = KNNPipeline.predict(X_test)


# In[21]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[22]:


tn, fp, fn, tp


# In[23]:


print(classification_report(y_test,y_pred)) # with no encoding f for 1 is .55


# In[ ]:


'''
I ran a k-nearest neighbor model to see if I could predict it was 
a rainy day or not based on bike rental rates. To get an prediction score 
of 0 (no rain) or 1 (rain) I compared each row of the test data to the
3 closests data points in the data that was used to train the model.
The ouput of 0 or 1 was generated based on how much the row was similar/closer 
by a majority vote of the 3 points it was compared to. For example, if row 1 
of the test data was being compared to 3 data points in the training data 
and 2 of those points had a class of 0 and 1 had a class of 1 row 1 would
be given the class 0. Each row of the test data went through this process to 
produce the final result. 

We can evaluate the model above by looking at its precision, recall, and f-1scores.
precision is a score of the models ability to correctly predict cases that are 
predicted to be positive, recall is a score of positive classifications from ones
that are actually positive, and f1 is an average of precision and recall and is the
accuracy of the model. While this model did a great job of predicting no rain days
with a precision score of .94, a recall score of .98, and a f1-score of .96.
However, we are moreintersted in predicting if it was a rainy data based 
on bike rental rates.When it came to predicting rainy days there was dip 
in model performance.With an precision score of .85 a recall score of .47 
and a f1-score of .61.I would conclude this model is fine to use to determine
if it was raining based on bike rental rates. However, better alternatives should
be explored as well. 

There could be a number of factors that conrtibue to this. One issue 
impacting this could be the features we are using. While there were 
some features that may have captured "seasonal" data such as "season" and 
"month" the features were mostly composted of human acitivity data, 
specifically bike rental. Thus, because this can vary so widely it 
could be hard to determine if it was raining or not based on bike rental acitivty.
Another factor could be the data set is too small.
'''


# ### KNN: Working Day

# In[ ]:


'''
The model below uses the same KNN process as the one above to predict
if it was a weekday or not a weekday.
'''


# In[ ]:


df_workingday = rain_df
df_workingday = df_workingday.drop(['Cnt/Hr','z_score'], axis = 1)
df_workingday.head()


# In[ ]:


#Here we are splitting our data into features in one dataframe and the target calss in another
df_workingday_KNN_X = df_workingday[['season', 'yr', 'mnth', 'hr', 
                                     'weekday', 'weathersit','atemp', 'hum', 'windspeed', 'cnt']]
df_workingday_KNN_Y = df_workingday['workingday']


# In[ ]:


# Once again we are splitting the data into train test split
Knn_X , Knn_y = df_workingday_KNN_X, df_workingday_KNN_Y
X_train, X_test, y_train, y_test = train_test_split(Knn_X,  
                                                    Knn_y, test_size=0.30, random_state=42)


# In[ ]:


KNNPipeline = Pipeline([
    ('imputer', Imputer()),
    ('scaler',StandardScaler()),
    ('KNNModel',KNeighborsClassifier(n_neighbors= 3, metric='manhattan'))  
])


# In[ ]:


KNNPipeline.fit(X_test,y_test)
y_pred = KNNPipeline.predict(X_test)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[ ]:


tn, fp, fn, tp


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


'''
Like the model before it each row of the test data was compred to 3 
data points of the training data that was used to buld the model. 
The new row or 'test' data was then given a ouput of a 0 or 1 based on 
the majority vote of the pieces of data that it was most similar/closest too. 
I used a manhattan distance measurement to get this distances since city blocks 
tend to run parallal to each other.

This KNN model produced better results than the rain/no rain model.
For non workdays days (0) the model had a precision score of .95, 
recall score of .93, and f1-score of .94. For working days (1) the model 
had a precision score of .97, recall score of .98, and f1-score of .97. 

I would feel more comfortable using this model that predicts weekday or not 
versus the model that predicted rain/no rain days. I believe this model preformed
better becasue the features used were much more applicable to what was being predicted.
'''


# ### K-Means: Rainy Day

# In[ ]:


cluster_df1 = rain_df.drop(['Cnt/Hr','z_score'], axis = 1)


# In[ ]:


# source: https://www.youtube.com/watch?v=dWy_VVepxMQ&t=2162s
sumofsq = []
for i in range(1,5):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(cluster_df1)
    sumofsq.append(kmeans.inertia_)   
plt.plot(range(1,5),sumofsq) 
plt.title("Elbow Graph")
plt.xlabel("Number of cluster")
plt.ylabel("Sum of Squares")
plt.show()


# In[ ]:


'''
The graph above shows the optimal number of clusters to use
is 2 with a sum of squares of approximatly 1.8 
'''


# In[ ]:


cluster_df1.head()


# In[ ]:


cluster_df1['Rain'] = cluster_df1.weathersit.map({1:0,2:0,3:1,4:1}) # no rain = 0 rain = 1


# In[ ]:


cluster_df1 = cluster_df1.drop(['weathersit'], axis = 1)


# In[ ]:


cluster_df = cluster_df1[['season', 'yr', 'mnth', 'hr', 
                          'weekday', 'workingday', 'cnt', 'Rain']]


# In[ ]:


cluster_df.head()


# In[ ]:


# shuffle the data before scaling it. 
from sklearn.utils import shuffle
cluster_df = shuffle(cluster_df)


# In[ ]:


# creating the target rain class to cross reference
# with the formed clusters by K-means 
y = pd.DataFrame(cluster_df['Rain'])


# In[ ]:


KMeansPipeline = Pipeline([
    ('imputer', Imputer()),
    ('scaler',StandardScaler()),
    ('KMeansModel',KMeans(n_clusters = 2, random_state= 7))  
])


# In[ ]:


KMeansPipeline.fit(cluster_df)
cluster_df['cluster'] = KMeansPipeline.predict(cluster_df)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y,cluster_df['cluster']).ravel()


# In[ ]:


tn, fp, fn, tp 


# In[ ]:


print(classification_report(y,cluster_df['cluster'] ))


# In[ ]:


'''
When comparing the 2 clusters fromed by the K-means model to rain/no rain days 
once againthe performence is called into question. Both no rain (0) and rain days
(1) performed poorly.

For no rain days the model had a precision score of .90, recall score of .40, 
and f1-score of .55. For rain days (1) the model had a precision score of .07, 
recall score of .52, and f1-score of .15. Thus, this would not be an 
ideal model to use to determine if it was raining based on bike rental activity. 
'''

