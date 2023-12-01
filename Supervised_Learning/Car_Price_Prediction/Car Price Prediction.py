#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# In[1]:


# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


# In[2]:


# Import Dataset
car = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Oasis Infobyte\Car Price Prediction\CarPrice_Assignment.csv")


# In[3]:


car


# # Data Preprocessing

# In[4]:


# Check Data types and Information
car.info()


# In[5]:


# Describe Data and Statistics
car.describe().T


# In[6]:


# Check Null Values
car.isnull().sum()


# In[7]:


car.shape


# In[8]:


# Check Duplicate Entry
car.duplicated().sum()


# # Visualization

# In[9]:


sns.histplot(car, x='symboling')


# In[10]:


sns.histplot(car, x='fueltype')


# In[11]:


sns.histplot(car, x='aspiration')


# In[12]:


sns.histplot(car, x='doornumber')


# In[13]:


sns.histplot(car, x='carbody')


# In[14]:


sns.histplot(car, x='drivewheel')


# In[15]:


sns.histplot(car, x='enginelocation')


# In[16]:


sns.histplot(car, x='fuelsystem')


# In[17]:


# Compare Price with wheelbase, enginesize, boreration, horsepower, citympg, highwaympg
sns.pairplot(car,x_vars=('wheelbase', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg'),y_vars='price')


# # Model Preprocessing

# In[18]:


# Convert Categorical to Numerical Data
car['fueltype'] = car['fueltype'].map({'gas':1,'diesel':0})
car['aspiration'] = car['aspiration'].map({'std':1,'turbo':0})
car['doornumber'] = car['doornumber'].map({'two':1,'four':0})
car['drivewheel'] = car['drivewheel'].map({'4wd':2,'rwd':1,'fwd':0})
car['enginelocation'] = car['enginelocation'].map({'front':1,'rear':0})
car.head()


# In[19]:


# Using Dummy Methods are Convert Categorical to Numerical data
encoded_car = pd.get_dummies(car, columns = ['symboling', 'carbody', 'cylindernumber', 'enginetype', 'fuelsystem'])
encoded_car


# In[20]:


# Drop Uncertain Columns
encoded_car = encoded_car.drop(encoded_car.iloc[:,[0,1]], axis=1)
encoded_car


# In[21]:


encoded_car.info()


# # Model Preparation

# In[22]:


# Create Independent and Dependent Variable
X = encoded_car.drop('price',axis=1)
y = encoded_car.loc[:,'price']


# In[23]:


# Spliting the Dataset for Traing and Testing data 
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=42)


# # Linear Regression Model

# In[24]:


# Build and Fit Linear Regression Model
regression = LinearRegression()
regression.fit(X_train,y_train)
regression_pred = regression.predict(X_test)
regression_pred


# In[25]:


# Check Evalution Metrices Error
rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=regression_pred))
mape = mean_absolute_percentage_error(y_true=y_test, y_pred=regression_pred)
Adj_sqaure = r2_score(y_true=y_test, y_pred=regression_pred)
print('RMSE :', rmse)
print('MAPE :', mape)
print('Adj Square :', Adj_sqaure)


# # Random Forest Regressor Model

# In[26]:


# Build and fit Random Forest Regressor Model
RForest = RandomForestRegressor()
RForest.fit(X_train , y_train)
RForest_pred = RForest.predict(X_test)
RForest_pred


# In[27]:


# Check Evalution Metrices Error
rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=RForest_pred))
mape = mean_absolute_percentage_error(y_true=y_test, y_pred=RForest_pred)
Adj_sqaure = r2_score(y_true=y_test, y_pred=RForest_pred)
print('RMSE :', rmse)
print('MAPE :', mape)
print('Adj Square :', Adj_sqaure)

