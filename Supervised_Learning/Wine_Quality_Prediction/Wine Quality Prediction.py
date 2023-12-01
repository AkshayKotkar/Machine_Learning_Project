#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction
# ### Using Linear Regression Model

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# In[2]:


# Import Wine Quality Dataset
df = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Bharat Intern\Wine Quality Prediction\winequality-red.csv")


# In[3]:


df


# # Data Preprocessing

# In[4]:


df.shape


# In[5]:


# Check DataType and Information
df.info()


# In[6]:


# Description of DataSet
df.describe().T


# In[7]:


# Find any Null values available in dataset
df.isnull().sum()


# In[8]:


# Check Duplicated Values
df.duplicated().sum()


# In[9]:


# Using drop_duplicates drop 240 duplicated values.
df.drop_duplicates(inplace=True)


# In[10]:


# Then Remain 1359 Rows
df.shape


# In[11]:


# Check Unique No of Wine Quality
df['quality'].value_counts().sort_index()


# In[12]:


# Correlation between all features
df.corr()


# # Visualization

# In[13]:


sns.heatmap(df.corr(), annot=True, cmap='PRGn', fmt='.2f',annot_kws={"size":8})


# In[14]:


sns.countplot(data=df, x='quality')
plt.title("Count of Wine Quality")


# In[15]:


plt.figure(figsize=(18,12))
plt.subplot(4,4,1)
sns.histplot(data=df, x='fixed acidity', kde=True)

plt.subplot(4,4,2)
sns.histplot(data=df, x='volatile acidity', kde=True)

plt.subplot(4,4,3)
sns.histplot(data=df, x='citric acid', kde=True)

plt.subplot(4,4,4)
sns.histplot(data=df, x='residual sugar', kde=True)

plt.subplot(4,4,5)
sns.histplot(data=df, x='chlorides', kde=True)

plt.subplot(4,4,6)
sns.histplot(data=df, x='free sulfur dioxide', kde=True)

plt.subplot(4,4,7)
sns.histplot(data=df, x='total sulfur dioxide', kde=True)

plt.subplot(4,4,8)
sns.histplot(data=df, x='density', kde=True)

plt.subplot(4,4,9)
sns.histplot(data=df, x='pH', kde=True)

plt.subplot(4,4,10)
sns.histplot(data=df, x='sulphates', kde=True)

plt.subplot(4,4,11)
sns.histplot(data=df, x='alcohol', kde=True)

plt.subplot(4,4,12)
sns.histplot(data=df, x='quality', kde=True)

plt.tight_layout()


# In[16]:


plt.figure(figsize=(18,12))
plt.subplot(4,4,1)
sns.boxplot(data=df, y='fixed acidity')

plt.subplot(4,4,2)
sns.boxplot(data=df, y='volatile acidity')

plt.subplot(4,4,3)
sns.boxplot(data=df, y='citric acid')

plt.subplot(4,4,4)
sns.boxplot(data=df, y='residual sugar')

plt.subplot(4,4,5)
sns.boxplot(data=df, y='chlorides')

plt.subplot(4,4,6)
sns.boxplot(data=df, y='free sulfur dioxide')

plt.subplot(4,4,7)
sns.boxplot(data=df, y='total sulfur dioxide')

plt.subplot(4,4,8)
sns.boxplot(data=df, y='density')

plt.subplot(4,4,9)
sns.boxplot(data=df, y='pH')

plt.subplot(4,4,10)
sns.boxplot(data=df, y='sulphates')

plt.subplot(4,4,11)
sns.boxplot(data=df, y='alcohol')

plt.subplot(4,4,12)
sns.boxplot(data=df, y='quality')

plt.tight_layout()


# In[17]:


plt.scatter(x=df['free sulfur dioxide'], y=df['total sulfur dioxide'], c=df['quality'])
plt.colorbar(label='Wine Quality')
plt.title("Compare Free and Total Sulfur Dioxide")
plt.xlabel('Free Sulfur Dioxide')
plt.ylabel('Total Sulfur Dioxide')


# In[18]:


plt.scatter(x=df['citric acid'], y=df['fixed acidity'], c=df['quality'])
plt.colorbar(label='Wine Quality')
plt.title("Compare Citric Acid and Fixed Acidity")
plt.xlabel('Citric Acid')
plt.ylabel('Fixed Acidity')


# # Model Preprocessing

# In[19]:


# Create Independent and Dependent Varaible.
X = df.drop(['quality'], axis=1)
y = df['quality']


# In[21]:


# Standard Scaling for all data transform in one scale.
sc = StandardScaler()
X = sc.fit_transform(X)


# In[22]:


# Split Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, random_state=42)


# # Model Building

# In[23]:


# Build Linear Regression Model
model = LinearRegression()
fit = model.fit(X_train, y_train)
y_pred = fit.predict(X_test)
t_pred = fit.predict(X_train)


# In[24]:


# Check Error Rate
print("Training Error:",mean_squared_error(y_train, t_pred))
print("Testing Error:" ,mean_squared_error(y_test, y_pred))
print('\n')
print("Training Error:",mean_absolute_percentage_error(y_train, t_pred))
print("Testing Error:" ,mean_absolute_percentage_error(y_test, y_pred))

