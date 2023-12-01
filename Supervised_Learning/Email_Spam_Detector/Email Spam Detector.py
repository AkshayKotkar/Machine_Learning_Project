#!/usr/bin/env python
# coding: utf-8

# # Email Spam Detector

# In this project, the objective is to develop an email spam detector using Python and machine learning techniques. The primary goal is to create a model that can classify emails as either spam or non-spam (ham).

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


# Import Dataset
data = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Oasis Infobyte\Email Spam Detector\spam.csv", encoding='windows-1252')


# In[3]:


# Data Observation
data


# ## Data Preprocessing

# In[4]:


# Check Datatype and Information
data.info()


# In[5]:


# Check some Statistics
data.describe().T


# In[6]:


# Column Name
data.columns


# In[7]:


# Drop Unimportant Column, In this column Ham Reply Data Stored
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)


# In[8]:


# Shape of Dataset After Drop some Columns
data.shape


# In[9]:


# Check Null Values
data.isnull().sum()


# In[10]:


# Check Duplicated Values
data.duplicated().sum()


# In[11]:


# Remove Duolicates value from Dataset
data = data.drop_duplicates().reset_index(drop=True)


# In[12]:


# Label Encoding
# Our Target Data Change to O and 1
encoding = LabelEncoder()
data['v1'] = encoding.fit_transform(data['v1'])
data
# 0 is Human mail
# 1 is Spam mail


# In[13]:


# Value Counts for Check Spam and Ham mail counting
data['v1'].value_counts()


# ## Data Visualization

# In[14]:


plt.pie(data['v1'].value_counts(), labels=['Ham','Spam'], autopct='%0.2f')
plt.show()


# # Model Preparation

# In[15]:


# Create Independant and Dependant Varaible
X = data['v2']
y = data['v1']


# In[16]:


# Spliting the Dataset for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[17]:


# Vectorize the Mail Data
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
vectorizer.get_feature_names_out()


# In[18]:


# Build and Fit the Logistic Regression Model
Logistic_Regression_Model = LogisticRegression()
Logistic_Regression_Model = Logistic_Regression_Model.fit(X_train,y_train)


# In[19]:


# Predict Train Dataset 
train_pred = Logistic_Regression_Model.predict(X_train)
accuracy_score(y_train,train_pred)


# In[20]:


# Predict Test Dataset 
test_pred = Logistic_Regression_Model.predict(X_test)
accuracy_score(y_test,test_pred)


# In[21]:


# Check Confusion Matrix
matrix = confusion_matrix(y_test,test_pred)
print(matrix)


# In[22]:


# Check Classification report
report = classification_report(y_test,test_pred, target_names=['Spam', 'Ham'])
print(report)


# In[23]:


# Build and fit Random Forest Classifier Model
Random_Forest_Model = RandomForestClassifier(n_estimators=100)
Random_Forest_Model = Random_Forest_Model.fit(X_train,y_train)


# In[24]:


# Predict Train Dataset
train_pred = Random_Forest_Model.predict(X_train)
accuracy_score(y_train,train_pred)


# In[25]:


# Predict Test Dataset 
test_pred = Random_Forest_Model.predict(X_test)
accuracy_score(y_test,test_pred)


# In[26]:


# Check Confusion Matrix
matrix = confusion_matrix(y_test,test_pred)
print(matrix)


# In[27]:


# Check Classification report
report = classification_report(y_test,test_pred, target_names=['Spam', 'Ham'])
print(report)


# In[28]:


# Predict the New mail Using Random Forest Classifier Model
Mail = "Nah I don't think he goes to usf, he lives around here though"
Mail_Vectorizer = vectorizer.transform([Mail])
Prediction = Random_Forest_Model.predict(Mail_Vectorizer)
# If Prediction is 0 Means Ham and 1 is Spam
if Prediction == 0:
    print('Ham Mail')
elif Prediction == 1:
    print('Spam Mail')

