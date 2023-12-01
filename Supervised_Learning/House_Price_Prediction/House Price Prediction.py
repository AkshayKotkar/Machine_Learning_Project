#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction 
# ### Using Linear Regression Model

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# In[2]:


# Import Training DataSet
Train = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Bharat Intern\House Price Prediction\Train.csv")


# In[3]:


Train


# # Data Preprocessing 

# In[4]:


Train.shape


# In[5]:


# Column Names
Train.columns


# In[6]:


# Change Column Name
Train.columns = ['POSTED BY', 'UNDER CONSTRUCTION', 'RERA', 'BHK NO', 'BHK OR RK',
       'SQUARE FT', 'READY TO MOVE', 'RESALE', 'ADDRESS', 'LONGITUDE',
       'LATITUDE', 'TARGET']


# In[7]:


# Check DataType and Information
Train.info()


# In[8]:


# Description of Data
Train.describe().T


# In[9]:


# Check Null Values in DataSet
Train.isnull().sum()


# In[10]:


Train['UNDER CONSTRUCTION'].value_counts()


# In[11]:


Train['RERA'].value_counts()


# In[12]:


Train['BHK NO'].value_counts().sort_index()


# In[13]:


Train['BHK OR RK'].value_counts()


# In[14]:


Train['READY TO MOVE'].value_counts().sort_index()


# In[15]:


Train['RESALE'].value_counts().sort_index()


# In[16]:


# Correlation Between Features
Train.corr()


# # Visualization

# In[17]:


sns.heatmap(Train.corr(), annot=True, cmap='cool', annot_kws={"size":8}, fmt='.2f')


# In[18]:


plt.figure(figsize=(20,14))
plt.subplot(3,2,1)
sns.countplot(data=Train, x="UNDER CONSTRUCTION")
plt.title('Count of Under Construction Building')

plt.subplot(3,2,2)
sns.countplot(data=Train, x="RERA")
plt.title('Count of RERA')

plt.subplot(3,2,3)
sns.countplot(data=Train, x="BHK NO")
plt.title('Count of Bedrooms')

plt.subplot(3,2,4)
sns.countplot(data=Train, x="BHK OR RK")
plt.title('Count of BHK or RK')

plt.subplot(3,2,5)
sns.countplot(data=Train, x="READY TO MOVE")
plt.title('Count of Ready to MOve')

plt.subplot(3,2,6)
sns.countplot(data=Train, x="RESALE")
plt.title('Count of Flat Resale')

plt.tight_layout()


# In[19]:


plt.figure(figsize=(10, 6)) 

bin_edges = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]

sns.histplot(data=Train, x="SQUARE FT", bins=bin_edges)
plt.xlabel("Square Feet")
plt.ylabel("Count")
plt.title("Distribution of Square Feet")


# In[20]:


Train.boxplot(column='TARGET', by='BHK NO')


# In[21]:


plt.scatter(Train['LONGITUDE'], Train['LATITUDE'], c=Train['TARGET'])
plt.colorbar(label='Price (in Lacs)')


# # Model Preprocessing

# ### Train Data Preprocessing

# In[22]:


# Label Encoding for Categorical Columns
le = LabelEncoder()
Train['BHK OR RK'] = le.fit_transform(Train['BHK OR RK'])
Train['POSTED BY'] = le.fit_transform(Train['POSTED BY'])


# In[23]:


# Drop unnesseary column
Train = Train.drop(['ADDRESS'], axis=1)


# In[24]:


# Standard Scaling for Scale a column in one unit.
sc = StandardScaler()
Train[['SQUARE FT', 'LONGITUDE', 'LATITUDE']] = sc.fit_transform(Train[['SQUARE FT', 'LONGITUDE', 'LATITUDE']])


# In[25]:


Train


# In[26]:


# Split Dataset for Training Model
X_train = Train.drop(['TARGET'], axis=1)
y_train = Train['TARGET']


# ### Test Data for Preprocessing

# In[27]:


# Import Test DataSet
Test = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Bharat Intern\House Price Prediction\Test.csv")


# In[28]:


Test.columns = ['POSTED BY', 'UNDER CONSTRUCTION', 'RERA', 'BHK NO', 'BHK OR RK',
       'SQUARE FT', 'READY TO MOVE', 'RESALE', 'ADDRESS', 'LONGITUDE',
       'LATITUDE']


# In[29]:


X_Test = Test


# In[30]:


# Transform Column using Label Encoding and Standard Scaling.
le = LabelEncoder()
X_Test['BHK OR RK'] = le.fit_transform(X_Test['BHK OR RK'])
X_Test['POSTED BY'] = le.fit_transform(X_Test['POSTED BY'])
X_Test = X_Test.drop(['ADDRESS'], axis=1)
sc = StandardScaler()
X_Test[['SQUARE FT', 'LONGITUDE', 'LATITUDE']] = sc.fit_transform(X_Test[['SQUARE FT', 'LONGITUDE', 'LATITUDE']])


# In[31]:


X_Test


# # Model Building

# In[32]:


# Build Linear Regression Model
model = LinearRegression()
fit = model.fit(X_train, y_train)


# In[33]:


# Predict Target value of Test Data
y_Test = fit.predict(X_Test)


# In[34]:


X_Test['TARGET'] = y_Test


# In[35]:


X_Test


# ### Predict Train Values Data for Accuracy

# In[36]:


# Predict Train Values and Check Accuracy.
y_pred = fit.predict(X_train)


# In[37]:


print("Training Error:",mean_squared_error(y_train, y_pred))
print('\n')
print("Training Error:",mean_absolute_percentage_error(y_train, y_pred))

