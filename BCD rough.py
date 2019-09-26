#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 14
fig_size[1] = 9.5
plt.rcParams["figure.figsize"] = fig_size


# In[59]:


df = pd.read_csv('bcd_data.csv')


# In[60]:


df1 = df


# In[61]:


df.head()


# In[62]:


df.shape


# In[63]:


df.columns


# In[64]:


df.info()


# In[65]:


df.describe()


# In[66]:


df.isna().sum()


# In[67]:


df['diagnosis'].value_counts()/df['diagnosis'].value_counts().sum() * 100


# In[68]:


df['diagnosis'].value_counts().plot(kind = 'bar',color = ['c','r'])


# In[69]:


corr = df.corr().abs()
corr.head()


# In[70]:


ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
); 


# In[71]:


upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))


# In[72]:


drop_col = [col for col in upper.columns if any(upper[col]>.90)]


# In[73]:


df_c = df1.drop(df1[drop_col],axis=1)


# In[74]:


df_c.head()


# In[75]:


df_c.shape


# ## Normalize the Data for PCA

# In[76]:


sc = StandardScaler()
ss = sc.fit_transform(df_c.iloc[:,2:])
df_std = pd.DataFrame(ss,columns=df_c.columns[2:])
df_std_copy = df_std.copy()


# In[77]:


df_std.head()


# ## Implementing PCA to show Variance of the Data

# In[87]:


pca = PCA(.95)
pca.fit(df_std_copy)


# In[88]:


print(pca.explained_variance_ratio_) 


# In[89]:


plt.style.use('ggplot')
plt.plot(np.cumsum(pca.explained_variance_ratio_),color='grey')
plt.axhline(y=0.95,color='r',linestyle = '--',lw=1.5)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[90]:


x_train,x_test,y_train,y_test = train_test_split(df_std,df_c.iloc[:,1],test_size = 0.3)


# In[97]:


pca.fit(x_train)
pca.transform(x_train)


# In[98]:


pca = PCA(.95)
principalComponents = pca.fit_transform(x_train)
#pca.fit(x_train)


# In[99]:


principalDataframe = pd.DataFrame(data = principalComponents)
principalDataframe = principalDataframe.add_prefix('PCA_')


# In[ ]:





# ## Making a data frame of Principal components

# In[100]:


principalDataframe = pd.DataFrame(principalComponents)
principalDataframe = principalDataframe.add_prefix('PCA_')


# In[101]:


principalDataframe.head()


# In[102]:


plt.scatter(principalDataframe.PCA_0, principalDataframe.PCA_1)
plt.title('PCA_0 against PCA_1')
plt.xlabel('PCA_0')
plt.ylabel('PCA_1')


# In[103]:


fig = plt.figure(figsize = (14,9.5))
plt.style.use('ggplot')
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PCA_0')
ax.set_ylabel('PCA_1')

ax.set_title('Plot of PCA_0 vs PCA_1', fontsize = 20)

targets = ['B', 'M']

colors = ['r', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = df_c.iloc[:,1] == target
    ax.scatter(principalDataframe.loc[indicesToKeep, 'PCA_0']
               , principalDataframe.loc[indicesToKeep, 'PCA_1']
               , c = color
               , s = 50)
    
ax.legend(targets)
ax.grid()


# In[104]:


x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


# ## Logistic Regression w/o PCA

# In[105]:


x_train_wopca,x_test_wopca,y_train_wopca,y_test_wopca = train_test_split(df_std,df_c.iloc[:,1],test_size = 0.3)


# In[106]:


clf_wopca = LogisticRegression(solver = 'lbfgs')
clf_wopca.fit(x_train_wopca,y_train_wopca)


# In[107]:


y_pred_log_wopca = clf_wopca.predict(x_test_wopca)


# In[108]:


confusion_matrix(y_test_wopca, y_pred_log_wopca)


# In[109]:


acc_log_wopca = round(clf_wopca.score(x_train_wopca,y_train_wopca) * 100, 2)
acc_log_wopca


# In[ ]:





# ## Logistic Regression with PCA

# In[110]:


clf = LogisticRegression(solver = 'lbfgs')
clf.fit(x_train,y_train)


# In[111]:


y_pred_log = clf.predict(x_test)


# In[112]:


confusion_matrix(y_test, y_pred_log)


# In[113]:


acc_log = round(clf.score(x_train, y_train) * 100, 2)
acc_log


# ## Decision Tree with PCA

# In[114]:


dt_e = DecisionTreeClassifier(criterion='entropy',min_samples_split=20)
dt_e.fit(x_train,y_train)


# In[115]:


y_pred_dt = dt_e.predict(x_test)


# In[116]:


confusion_matrix(y_test, y_pred_dt)


# In[117]:


acc_dt = round(dt_e.score(x_train, y_train) * 100, 2)
acc_dt


# In[118]:


with open('dtree.dot','w') as dotfile:
    export_graphviz(dt_e,out_file=dotfile,feature_names=principalDataframe.columns)
dotfile.close()


# In[119]:


principalDataframe.columns


# ## Random Forest with PCA

# In[120]:


rf= RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)


# In[121]:


y_pred_rf = rf.predict(x_test)


# In[122]:


confusion_matrix(y_test, y_pred_rf)


# In[123]:


acc_rf = round(rf.score(x_train, y_train) * 100, 2)
acc_rf


# In[ ]:





# In[ ]:





# In[ ]:




