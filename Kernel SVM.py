#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Kernel SVM


# In[18]:


# In case of non-linearly separable data, the simple SVM algorithm cannot be used. Rather,
# a modified version of SVM, called Kernel SVM, is used.


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[20]:


irisdata = pd.read_csv("/home/garv/Desktop/IRIS.csv")


# In[21]:


# PreProcessing

x = irisdata.iloc[:,0:4].values
y = irisdata.iloc[:,4].values


# In[22]:


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


# In[ ]:





# In[23]:


# Polynomial Kernel


# In[24]:


# Polynomial Kernel
from sklearn.svm import SVC   # Support Vector Classifier
svclassifier = SVC(kernel ='poly', degree= 8)
svclassifier.fit(x_train, y_train)


# In[25]:


y_pred = svclassifier.predict(x_test)


# In[26]:


# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:





# In[27]:


# Gaussian Kernel


# In[28]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)


# In[29]:


y_pred = svclassifier.predict(x_test)


# In[30]:


# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:





# In[31]:


# Sigmoid Kernel


# In[33]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(x_train, y_train)


# In[34]:


y_pred = svclassifier.predict(x_test)


# In[35]:


# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:


# If we compare the performance of the different types of kernels we can clearly
# see that the sigmoid kernel performs the worst. This is due to the reason that sigmoid
# function returns two values, 0 and 1, therefore it is more suitable for binary
# classification problems. However, in our case we had three output classes

