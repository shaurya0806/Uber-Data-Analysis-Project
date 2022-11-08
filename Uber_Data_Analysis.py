#!/usr/bin/env python
# coding: utf-8

# # Uber Data Analysis
# -----------------------------------------------------------------
# 
# - Defining the problem statement
# 
# 
# - Collecting the data
#     * Kaggle
#     
#     
# - Exploratory data analysis
# 
# 
# - Feature engineering
# 
# 
# - Modelling
# 
# 
# - Testing

# ## 1. Defining the problem statement
# ____________________________________________________
# 
# In this project, we study the data of Uber which is present in tabular format in which we use different libraries like numpy, pandas and matplotlib and different machine learning algorithms. 
# 
# We study different columns of the table and try to co-relate them with others and find a relation between those two. 
# 
# We try to find and analyze those key factors like date, month etc which helps Uber Company to enhance their business by focusing on those services and make required changes. 

# In[9]:


from IPython.display import Image
Image(url= "Uber_image.jpg")


# ## 2. Collecting the data
# ___________________________________

# In[1]:


import pandas as pd

uber_dataset = pd.read_csv(r"new_uber.csv")


# ## 3. Exploratory data analysis
# __________________________________________________
# 
# Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
# 
# It is a good practice to understand the data first and try to gather as many insights from it.
# 
# EDA is all about making sense of data in hand.
# 

# In[2]:


uber_dataset.head()


# In[3]:


uber_dataset.shape


# In[4]:


uber_dataset.info()


# In[5]:


uber_dataset.describe()


# In[6]:


uber_dataset.isnull().sum()


# ## 4. Feature Engineering
# -----------------------------------------------------
# 
# What is a feature and why we need the engineering of it? Basically, all machine learning algorithms use some input data to create outputs. This input data comprise features, which are usually in the form of structured columns. Algorithms require features with some specific characteristic to work properly. Here, the need for feature engineering arises. 
# 
# I think feature engineering efforts mainly have two goals:
# 
# 1) Preparing the proper input dataset, compatible with the machine learning algorithm requirements.
# 
# 2) Improving the performance of machine learning models.
# 
# **According to a survey in Forbes, data scientists spend 80% of their time on data preparation:**

# ### Ploting

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd


# #### Strip Plots

# In[8]:


sns.stripplot(data=uber_dataset, x='price', y='name')


# In[9]:


sns.stripplot(data=uber_dataset, x='price', y='icon')


# In[10]:


sns.stripplot(data=uber_dataset, x='surge_multiplier', y='source')


# In[11]:


sns.stripplot(data=uber_dataset, x='surge_multiplier', y='hour')


# #### Converting Timestamp to Datetime value

# In[8]:


uber_dataset['timestamp'].head()


# In[16]:


from datetime import datetime
timestamp1 = 1544952608
timestamp2 = 1543284024
timestamp3 = 1543818483
timestamp4 = 1543594384
timestamp5 = 1544728504
dt_object1 = datetime.fromtimestamp(timestamp1)
dt_object2 = datetime.fromtimestamp(timestamp2)
dt_object3 = datetime.fromtimestamp(timestamp3)
dt_object4 = datetime.fromtimestamp(timestamp4)
dt_object5 = datetime.fromtimestamp(timestamp5)

print("dt_object =", dt_object1)
print("dt_object =", dt_object2)
print("dt_object =", dt_object3)
print("dt_object =", dt_object4)
print("dt_object =", dt_object5)


# - So by this timestamp to datetime conversion we get to know that, our data is of the year 2018 and in the month of november and december only

# #### Bar plots

# In[26]:


uber_dataset['month'].value_counts().plot(kind='bar', figsize=(10,5), color='blue')


# In[27]:


uber_dataset['source'].value_counts().plot(kind='bar', figsize=(10,5), color='green')


# In[28]:


uber_dataset['name'].value_counts().plot(kind='bar', figsize=(10,5), color='orange')


# In[29]:


uber_dataset['icon'].value_counts().plot(kind='bar', figsize=(10,5), color='red')


# In[30]:


uber_dataset['uvIndex'].value_counts().plot(kind='bar', figsize=(10,5), color='brown')


# In[31]:


uber_dataset['moonPhase'].value_counts().plot(kind='bar', figsize=(10,5), color='orange')


# In[32]:


uber_dataset['precipProbability'].value_counts().plot(kind='bar', figsize=(10,5), color='blue')


# ### Label Encoding

# In[9]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 


# In[10]:


uber_dataset.dtypes


# In[11]:


uber_dataset['id']= label_encoder.fit_transform(uber_dataset['id']) 
uber_dataset['datetime']= label_encoder.fit_transform(uber_dataset['datetime']) 
uber_dataset['timezone']= label_encoder.fit_transform(uber_dataset['timezone'])
uber_dataset['destination']= label_encoder.fit_transform(uber_dataset['destination']) 
uber_dataset['product_id']= label_encoder.fit_transform(uber_dataset['product_id'])
uber_dataset['short_summary']= label_encoder.fit_transform(uber_dataset['short_summary'])
uber_dataset['long_summary']= label_encoder.fit_transform(uber_dataset['long_summary'])


# In[12]:


uber_dataset['name']= label_encoder.fit_transform(uber_dataset['name'])

print("Class mapping of Name: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)


# In[13]:


uber_dataset['source']= label_encoder.fit_transform(uber_dataset['source'])

print("Class mapping of Source: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)


# In[14]:


uber_dataset['icon']= label_encoder.fit_transform(uber_dataset['icon'])

print("Class mapping of Icon: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)


# In[15]:


uber_dataset.dtypes


# In[16]:


uber_dataset.head()


# ### Filling NAN Values

# In[17]:


uber_dataset.isnull().sum()


# In[18]:


uber_dataset['price'].median()


# In[19]:


uber_dataset["price"].fillna(10.5, inplace = True) 


# In[20]:


uber_dataset.isnull().sum()


# In[21]:


uber_dataset['price'].dtype


# In[22]:


uber_dataset['price'] = uber_dataset['price'].astype(int)


# In[23]:


uber_dataset['price'].head()


# ### RFE (Recursive Feature Elimination)

# In[24]:


import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[27]:


from sklearn.feature_selection import RFE


# In[28]:


X = uber_dataset.drop('price', axis = 1)
y = uber_dataset['price']


# In[29]:


X.head()


# In[30]:


y.head()


# In[31]:


X.shape


# In[32]:


y.shape


# In[33]:


y.value_counts().plot(kind='bar',figsize=(30,8),color='red')


# #### Training accuracy in 56 features

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[35]:


X_train.shape


# In[36]:


X_test.shape


# In[37]:


y_train.shape


# In[38]:


y_test.shape


# In[39]:


#Creating model
reg = LinearRegression()
#Fitting training data
reg = reg.fit(X_train, y_train)


# In[40]:


reg.score(X_train, y_train)


# #### Training accuracy in 40 features

# In[167]:


rfe = RFE(reg, 40, verbose=1)
rfe = rfe.fit(X, y)


# In[168]:


rfe.support_


# In[169]:


XX = X[X.columns[rfe.support_]]


# In[170]:


XX.head()


# In[171]:


X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size = 0.3, random_state = 10)


# In[172]:


X_train.shape


# In[173]:


#Creating model
reg1 = LinearRegression()
#Fitting training data
reg1 = reg1.fit(X_train, y_train)


# In[174]:


reg1.score(X_train, y_train)


# #### Training accuracy in 15 features

# In[175]:


rfe = RFE(reg, 15, verbose=1)
rfe = rfe.fit(X, y)


# In[176]:


XX = X[X.columns[rfe.support_]]


# In[177]:


XX.head()


# In[178]:


X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size = 0.3, random_state = 10,)


# In[179]:


X_train.shape


# In[180]:


#Creating model
reg1 = LinearRegression()
#Fitting training data
reg1 = reg1.fit(X_train, y_train)


# In[181]:


reg1.score(X_train, y_train)


# #### Training accuracy in 25 features

# In[41]:


rfe = RFE(reg, 25, verbose=1)
rfe = rfe.fit(X, y)


# In[42]:


XX = X[X.columns[rfe.support_]]


# In[43]:


XX.head()


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size = 0.3, random_state = 20,)


# In[45]:


X_train.shape


# In[46]:


#Creating model
reg1 = LinearRegression()
#Fitting training data
reg1 = reg1.fit(X_train, y_train)
#Y prediction
Y_pred = reg1.predict(X_test)


# In[47]:


reg1.score(X_train, y_train)


# - Since we find the accuracy for  k = 56 , 40 , 25 and 15. 
# - Hence we noticed that the when k = 25 we get the maximum training accuracy in Linear Regression Model

# ### 25 Columns After RFE

# In[48]:


XX.columns


# In[49]:


XX.shape


# In[50]:


XX.head()


# ### Drop Useless Features

# In[51]:


features_drop = ['latitude', 'longitude', 'apparentTemperature',
       'long_summary', 'precipIntensity', 'humidity', 'windSpeed', 'windGust',
       'temperatureHigh', 'apparentTemperatureHigh', 'dewPoint','precipIntensityMax',
       'temperatureMax', 'apparentTemperatureMax', 'distance', 'cloudCover', 'moonPhase']
new_uber = XX.drop(features_drop, axis=1)


# In[52]:


new_uber.head()


# ### Binning

# In[53]:


month_mapping = {11: 0, 12: 1}
new_uber['month'] = new_uber['month'].map(month_mapping)


# In[54]:


surge_multiplier_mapping = {1.: 0, 1.25: 1, 1.5: 2, 1.75: 3, 2.:4}
new_uber['surge_multiplier'] = new_uber['surge_multiplier'].map(surge_multiplier_mapping)


# ### Final Dataset

# In[55]:


new_uber.head()


# In[56]:


y.head()


# ## 5. Modeling
# -----------------------------------------------------

# In[57]:


new_uber.shape


# In[58]:


y.shape


# In[59]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xx_train, xx_test, yy_train, yy_test = train_test_split(new_uber, y, test_size = 0.2, random_state = 42)


# In[60]:


xx_train.shape


# In[61]:


xx_test.shape


# In[62]:


yy_train.shape


# In[63]:


yy_test.shape


# In[64]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 


# #### 5.1 Linear regression

# In[65]:


linear = LinearRegression()
linear.fit(xx_train, yy_train)
linear.score(xx_test, yy_test)


# #### 5.2 Decision Tree

# In[66]:


decision = DecisionTreeRegressor(random_state = 0)  
decision.fit(xx_train , yy_train) 
decision.score(xx_test, yy_test)


# #### 5.3 Random Forest

# In[67]:


random = RandomForestRegressor(n_estimators = 100, random_state = 0) 
random.fit(xx_train , yy_train)  
random.score(xx_test, yy_test)


# #### 5.4 Gradient Boosting Regressor

# In[ ]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5)
clf.fit(xx_train, yy_train)


# In[69]:


clf.score(xx_test, yy_test)


# #### K fold Crossvalidation

# In[70]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
cross_val_score(LinearRegression(),xx_test,yy_test,cv=cv)


# ## 6. Testing
# -----------------------------------------------------

# #### Linear regression

# In[71]:


linear.coef_


# In[72]:


prediction = linear.predict(xx_test)
prediction


# In[74]:


prediction=  prediction.astype(int)

plt.scatter(yy_test,prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
# In[77]:


from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(yy_test,prediction))
print('MSE :'," ", metrics.mean_squared_error(yy_test,prediction))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(yy_test,prediction)))


# In[78]:


sns.distplot(yy_test - prediction,bins=50)


# #### Random Forest

# In[79]:


predictions = random.predict(xx_test)


# In[83]:


sns.regplot(yy_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[84]:


from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(yy_test,predictions))
print('MSE :'," ", metrics.mean_squared_error(yy_test,predictions))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(yy_test,predictions)))


# In[85]:


sns.distplot(yy_test - predictions,bins=50)


# #### Price prediction function

# In[86]:


new_uber.head()


# In[87]:


def predict_price(name,source,surge_multiplier,icon):    
    loc_index = np.where(new_uber.columns==name)[0]

    x = np.zeros(len(new_uber.columns))
    x[0] = source
    x[1] = surge_multiplier
    x[2] = icon
    if loc_index >= 0:
        x[loc_index] = 1

    return random.predict([x])[0]


# In[88]:


pre= random.predict(xx_test)


# ####  <span style='background:yellow'>Follow  these instructions before predicting the price:</span> 
# <hr>
# 
# -  **For cab_name**:  <font color = 'red'>Black SUV --> 0 , Lux --> 1 , Shared --> 2 , Taxi --> 3 , UberPool --> 4 , UberX --> 5</font>
# 
# 
# - **For Source**:  <font color = 'blue'>Back Bay --> 0 , Beacon Hill --> 1 , Boston University --> 2 , Fenway --> 3 , Financial District --> 4 , Haymarket Square --> 5 , North End --> 6 , North Station --> 7 , Northeastern University --> 8 , South Station --> 9 , Theatre District --> 10 , West End --> 11</font>
# 
# 
# - **For Surge_multiplier** : <font color = 'red'>Enter Surge Multiplier value from 0 to 4</font>
# 
# 
# - **for Icon**:  <font color = 'blue'>clear-day  --> 0 , clear-night  --> 1 , cloudy  --> 2 , fog  --> 3 , partly-cloudy-day  --> 4 , partly-cloudy-night  --> 5 , rain  --> 6</font>
# 

# <span style='background:yellow'>**predict_price(cab_name , source , surge_multiplier , icon)**</span>

# In[90]:


predict_price(1 , 3, 2, 0)


# In[ ]:




