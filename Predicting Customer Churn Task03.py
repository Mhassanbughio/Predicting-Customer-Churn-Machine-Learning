#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[66]:


ds=pd.read_csv("Customer_Churn.csv")


# In[67]:


ds.head()


# ## Data cleaning and preprocessing

# In[68]:


ds.info()


# In[69]:


ds.describe()


# In[70]:


##Drop unwanted Columns from the dataset


# ## Feature engineering

# In[71]:


ds=ds.drop(columns=["RowNumber","Surname"])


# In[72]:


ds.info()


# In[73]:


ds["Geography"].unique


# In[74]:


ds["Gender"].unique

# fyi : One-hot encoding is a popular technique used in machine learning and data processing to represent categorical variables or features as binary vectors.
# In[75]:


ds=pd.get_dummies(data=ds,drop_first=True)


# In[76]:


ds

## If two or more independent variables have an exact linear relationship between them then it is perfect multicollinearity
# In[ ]:


Exploratory data analysis (EDA) to identify key features


# In[77]:


ds.Exited.plot.hist()


# In[78]:


(ds.Exited==0).sum() #Total customer stying at the company "0"


# In[79]:


(ds.Exited==1).sum() #Total customer exit the company "1" 


# In[80]:


ds_2=ds.drop(columns='Exited')


# In[41]:


ds_2.corrwith(ds['Exited']).plot.bar(figsize=(16,9), title='Correlated with Exited Column', rot = 45,grid = True)


# In[42]:


corr=ds.corr()


# In[43]:


plt.figure(figsize=(16,9))
sns.heatmap(corr,annot=True)


# In[81]:


X= ds.drop(columns='Exited')
y= ds['Exited']


# ## Model selection and training

# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[83]:


X_test.shape


# In[84]:


X_train.shape

## : StandardScaler is a commonly used technique in machine learning for standardizing or scaling numerical features before fitting a model. It transforms the data by subtracting the mean and dividing by the standard deviation, resulting in a distribution with a mean of 0 and a standard deviation of 1.
# In[85]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# In[86]:


X_train


# In[87]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
     


# In[88]:


y_pred= clf.predict(X_test)


# ## Model evaluation and fine-tuning

# In[89]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[90]:


acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)


# In[91]:


results=pd.DataFrame([['Logistic regression',acc,f1,prec,rec]],columns=['Model','Accuracy','F1','Precision','Recall'])
results


# In[92]:


print(confusion_matrix(y_test,y_pred))


# ## Model selection and training

# In[93]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)


# In[94]:


y_pred= clf.predict(X_test)


# In[95]:


acc=accuracy_score(y_test,y_pred)


# In[96]:


f1=f1_score(y_test,y_pred)


# In[97]:


prec=precision_score(y_test,y_pred)


# In[98]:


rec=recall_score(y_test,y_pred)


# ## Model evaluation and fine-tuning

# In[99]:


RF_results=pd.DataFrame([['Random Forest Classifier',acc,f1,prec,rec]],columns=['Model','Accuracy','F1','Precision','Recall'])
results.append(RF_results,ignore_index=True)


# In[100]:


print(confusion_matrix(y_test,y_pred))


# In[101]:


ds.head()


# In[ ]:


#Observe tha data 


# In[104]:


single_obs=[[647,40,3,85000.45,2,0,0,92012.45,0,1,1,1]]
clf.predict(scaler.fit_transform(single_obs))
     


# In[ ]:




