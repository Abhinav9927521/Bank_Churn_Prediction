#!/usr/bin/env python
# coding: utf-8

# # Bank Churn Prediction

# In[2]:


# Import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Load the dataset from sklearn.
df = pd.read_csv("Churn_Modelling.csv")


# In[4]:


df


# In[5]:


# Display the first few rows of the dataset.
df.head()


# In[6]:


# Display the last few rows of the dataset.
df.tail()


# In[7]:


# Checking the shape of the dataset.
df.shape


# In[8]:


df.info()


# In[9]:


# Checking any null value of the dataset
df.isnull().sum()


# In[10]:


# Checking the description of the dataset.
df.describe()


# In[11]:


df = df.dropna(axis=1)


# In[12]:


df.isnull().sum()


# In[13]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
label_encoder = LabelEncoder()
df["Surname"] = label_encoder.fit_transform(df["Surname"])
df["Gender"] = label_encoder.fit_transform(df["Gender"])


# In[14]:


df.corr()


# In[15]:


# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df.corr(), annot=True)
plt.title("Corelation Matrix HeatMap")
plt.show()


# In[16]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.title("Boxplot of Each Feature in the iris dataset")
# plt.title(rotation=45)
plt.show()


# In[17]:


# Plot the pairplots of all the datapoints
sns.pairplot(df)
plt.show()


# In[18]:


x = df.drop('Exited', axis=1)  # Features
y = df['Exited'] 


# In[19]:


# Split the dataset into training and testing sets.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[20]:


# Import the Logistic Regression model from sklearn.
model=LogisticRegression()
model.fit(x_train,y_train)


# In[21]:


model.score(x_test, y_test)


# In[59]:





# In[62]:


# model.score(x_test, y_test)


# In[22]:


# Import the support vector machine class from sklearn.svm
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(x_train,y_train)


# In[24]:


svm_model.score(x_test, y_test)


# In[ ]:




