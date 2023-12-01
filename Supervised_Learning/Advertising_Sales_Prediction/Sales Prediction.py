#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction based On Advertising Amount 
In this project, we used Python to create predictive models for sales based on advertising data.We explored the dataset, 
built two models (Linear Regression and Random Forest Regression), and evaluated their performance. These models offer businesses valuable insights into how advertising budgets impact sales.
# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import Dataset
sale = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Oasis Infobyte\Sales Prediction\Advertising.csv")


# In[3]:


sale


# # Data Preprocessing

# In[4]:


sale.columns


# In[5]:


# Drop unimporatnat columns
sale.drop('Unnamed: 0', axis=1, inplace=True)


# In[6]:


sale.shape


# In[7]:


# Check datatype and information
sale.info()


# In[8]:


# Check some statistics
sale.describe().T


# In[9]:


# Check any Null Values
sale.isnull().sum()


# In[10]:


# Check Correlation between sales and product
sale.corr()


# # Data Visualization

# In[11]:


sns.heatmap(sale.corr(), annot=True)


# In[12]:


sale.hist(bins=20, figsize=(10,7))


# In[13]:


sns.lmplot(data=sale, x='TV', y='Sales')
sns.lmplot(data=sale, x='Radio', y='Sales')
sns.lmplot(data=sale, x='Newspaper', y='Sales')


# In[14]:


fig = plt.figure(figsize=(12,4))
sale['Sales'].plot()


# # Model Preparation

# In[15]:


# Create Independent and Dependent Varaible
X = sale.drop('Sales', axis=1).values
y = sale.loc[:,'Sales'].values


# In[16]:


# Spliting the Dataset for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Linear Regression Model

# In[17]:


# Build and Fit model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
lin_pred


# In[18]:


# Check Evaluation Metrics Error
rmse = np.sqrt(mean_squared_error(y_true= y_test, y_pred= lin_pred))
mape = mean_absolute_percentage_error(y_true= y_test, y_pred= lin_pred)
Adj_Sqaure = r2_score(y_true= y_test, y_pred= lin_pred)
print('RMSE :', rmse)
print('MAPE :', mape)
print('Adj Square :', Adj_Sqaure)


# ### Random Forest Regressor

# In[19]:


# Build and Fit Model
ran_model = RandomForestRegressor()
ran_model.fit(X_train, y_train)
ran_pred = ran_model.predict(X_test)
ran_pred


# In[20]:


# Check Evaluation Metrics Error
rmse = np.sqrt(mean_squared_error(y_true= y_test, y_pred= ran_pred))
mape = mean_absolute_percentage_error(y_true= y_test, y_pred= ran_pred)
Adj_Sqaure = r2_score(y_true= y_test, y_pred= ran_pred)
print('RMSE :', rmse)
print('MAPE :', mape)
print('Adj Square :', Adj_Sqaure)


# ## Predict New Data

# In[21]:


# New Data for Prediction
Predict = pd.DataFrame({'TV': [147.04,230.1,232.1], 'Radio' : [23.26,37.8,8.6], 'Newspaper' : [30.55,69.2,8.7]})


# In[22]:


# Predict the data using linear regression model
lin_pred = lin_model.predict(Predict)
lin_pred


# In[23]:


# Predict the data using Random Forest Regressor
ran_pred = ran_model.predict(Predict)
ran_pred


# In[24]:


Predict["Linear Regression Predict"] = lin_pred
Predict["Random Forest Predict"] = ran_pred


# In[25]:


# Predicated DataFrame
Predict

