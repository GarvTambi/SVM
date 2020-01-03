#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Simple SVM


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


# The following code reads bank currency note data into pandas dataframe:
bankdata = pd.read_csv("/home/garv/Desktop/bill_authentication.csv")


# In[21]:


bankdata.shape


# In[22]:


bankdata.head()


# In[23]:


bankdata.describe()


# In[25]:


# Data Processing
# Data preprocessing involves (1) Dividing the data into attributes and labels and
# (2) dividing the data into training and testing sets.

x = bankdata.iloc[:,0:4].values
y = bankdata.iloc[:,4].values


# In[26]:


#  the model_selection library of the Scikit-Learn library contains the train_test_split
# method that allows us to seamlessly divide data into training and test sets.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


# In[28]:


# Training the Algorithm

from sklearn.svm import SVC   # Support Vector Classifier
svclassifier = SVC(kernel='linear')#This class takes one parameter,which is the kernel type
svclassifier.fit(x_train, y_train)


# In[29]:


y_pred = svclassifier.predict(x_test)


# In[30]:


print(y_pred)


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:




