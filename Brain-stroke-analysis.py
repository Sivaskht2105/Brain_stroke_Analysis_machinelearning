#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Importing Dataset 

# In[2]:


df=pd.read_csv("stroke.csv")
df


# In[3]:


df.shape


# In[4]:


df.size


# In[5]:


df.isnull().sum()


# In[6]:


df.bmi.replace(to_replace=np.nan, value=df.bmi.mean(), inplace=True)


# In[7]:


df.isnull().sum()


# In[8]:


df.nunique()


# In[9]:


df.duplicated().sum()


# In[10]:


df.info()


# In[11]:


df.columns


# In[12]:


df.describe()


# In[13]:


df.corr()


# # Data Visualization

# # Heat Map Correlation 

# In[14]:


sns.heatmap(df.corr(),cmap='magma',annot= True)
plt.show()


# # Count Plot
# 

# Gender

# In[15]:


print(df.gender.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="gender")
plt.show()


# hypertension

# In[16]:


print(df.hypertension.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="hypertension")
plt.show()


# Marriage Status

# In[17]:


print(df.ever_married.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="ever_married")
plt.show()


# In[18]:


print(df.work_type.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="work_type")
plt.show()


# Residence Type

# In[19]:


print(df.Residence_type.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="Residence_type")
plt.show()


# Smoking status

# In[20]:


print(df.smoking_status.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="smoking_status")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()


# Stroke

# In[21]:


print(df.stroke.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=df, x="stroke")
plt.show()


# In[22]:


fig = plt.figure(figsize=(7,7))
graph = sns.scatterplot(data=df, x="age", y="bmi", hue='gender')
graph.axhline(y= 25, linewidth=4, color='r', linestyle= '--')
plt.show()


# In[23]:


fig = plt.figure(figsize=(7,7))
graph = sns.scatterplot(data=df, x="age", y="avg_glucose_level", hue='gender')
graph.axhline(y= 150, linewidth=4, color='r', linestyle= '--')
plt.show()


# pair plot

# In[24]:


fig = plt.figure(figsize=(10,10))
sns.pairplot(df)
plt.show()


# # Data Preprocessing

# In[25]:


df.info()


# In[26]:


x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values


# In[27]:


x


# In[28]:


y


# Encoding

# In[29]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0,5,9])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))


# In[30]:


x


# Label Encoder

# In[31]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 15] = le.fit_transform(x[:, 15])
x[:, 16] = le.fit_transform(x[:, 16])


# In[32]:


x


# In[33]:


y


# In[34]:


print('Shape of X: ', x.shape)
print('Shape of Y: ', y.shape)


# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.7, random_state= 42)


# In[36]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


model = KNeighborsClassifier(n_neighbors=5)


# In[39]:


model.fit(x_train,y_train)


# In[40]:


model.score(x_train,y_train)*100


# In[41]:


model.score(x_test,y_test)*100


# In[42]:


y_predict = model.predict(x_test)


# In[43]:


y_predict


# In[44]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model1=KNeighborsClassifier()
model1.fit(x_train,y_train)

y_predict1=model1.predict(x_test)
print(classification_report(y_test,y_predict1))


# In[45]:


from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

model2=KNeighborsClassifier()
model2.fit(x_train,y_train)

y_predict2=model2.predict(x_test)
print(classification_report(y_test,y_predict2))


# In[46]:


from sklearn.metrics import classification_report
from sklearn.svm import SVC

model3=SVC(random_state=1)
model3.fit(x_train,y_train)

y_predict3=model3.predict(x_test)
print(classification_report(y_test,y_predict3))


# In[47]:


from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

model4=GaussianNB()
model4.fit(x_train,y_train)

y_predict4=model4.predict(x_test)
print(classification_report(y_test,y_predict4))


# In[48]:


from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

model5=DecisionTreeClassifier(random_state=1)
model5.fit(x_train,y_train)

y_predict5=model5.predict(x_test)
print(classification_report(y_test,y_predict5))


# In[49]:


from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

model6=RandomForestClassifier(random_state=1)
model6.fit(x_train,y_train)

y_predict6=model6.predict(x_test)
print(classification_report(y_test,y_predict6))


# In[50]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_predict6)
print(cm)
accuracy_score(y_test,y_predict6)


# # Feature Importance

# In[51]:


#get importance
importance=model6.feature_importances_
#Summerize feature imporatance
for i,v in enumerate(importance):
    print('Feature:%0d,score:%.5f'%(i,v))


# # Conclusion

# # Final observation:
#     

# In[52]:


cm = confusion_matrix(y_test, y_predict)
print('Confusion Matrix:\n',cm)

# Calculate the Accuracy
accuracy = accuracy_score(y_predict,y_test)
print('Accuracy: ',accuracy)

#Visualizing Confusion Matrix
plt.figure(figsize = (6, 3))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['No stroke', 'Stroke'], xticklabels = ['Predicted no stroke', 'Predicted stroke'])
plt.yticks(rotation = 0)
plt.show()


# In[53]:


results=pd.DataFrame({'Model':['LogisticRegression','KNeighborsClassifier','GaussianNB','DecisionTreeClassifier',
                               'RandomForestClassifier'],'Score':[94,94,36,91,95]})
output_df=results.sort_values(by='Score',ascending=False)
output_df=output_df.reset_index(drop=True)
output_df


# In[54]:


plt.figure(figsize=(5,5))
plt.bar(output_df['Model'], output_df['Score'], color='red')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Comparison of Model Scores')
plt.ylim(0, 100)
plt.xticks(rotation=90)
plt.show()


# In[ ]:




