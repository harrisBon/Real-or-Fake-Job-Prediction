#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('fake_job_postings.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe().transpose()


# In[6]:


sns.countplot(x='fraudulent',data=df)


# In[7]:


# We are dealing with imbalanced data
df.corr()


# In[8]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')


# In[9]:


df.corr()['fraudulent'].sort_values().drop(['fraudulent','job_id']).plot(kind='bar')


# In[10]:


df.isnull().sum()


# In[11]:


100*df.isnull().sum()/len(df)


# In[12]:


#There are too many missing values in the above 2 columns
df=df.drop(['location','salary_range'],axis=1)


# In[13]:


df.head()


# In[14]:


df['title'].nunique()


# In[15]:


df['department'].nunique()


# In[16]:


# There are too many unique elements in the above 2 columns
df=df.drop(['title','department'],axis=1)


# In[17]:


df.info()


# In[18]:


df.fillna('Not Specified',inplace=True)


# In[19]:


df.isnull().sum()


# In[20]:


df['required_education'].value_counts()


# In[22]:


plt.figure(figsize=(15,14))
ax=sns.countplot(y='required_education',hue='fraudulent',data=df)
plt.legend(bbox_to_anchor=(1, 1.1),title='Fraudulent', loc=2, borderaxespad=0.)
plt.tight_layout()
for p in ax.patches:
    width = p.get_width()
    x, y = p.get_xy()
    ax.annotate(float(width),
                ((x + width), y), 
                xytext = (40, -15),
                fontsize = 16,
                color = '#000000',
                textcoords = 'offset points',
                ha = 'center',
                va = 'center')


# In[23]:


req_edu_dummies = pd.get_dummies(df['required_education'],drop_first=True)


# In[24]:


df = pd.concat([df.drop('required_education',axis=1),req_edu_dummies],axis=1)


# In[25]:


df.columns


# In[26]:


df.corr()['fraudulent'].sort_values()


# In[27]:


df['required_experience'].value_counts()


# In[28]:


req_exp_dummies = pd.get_dummies(df['required_experience'],drop_first=True)


# In[29]:


df = pd.concat([df.drop('required_experience',axis=1),req_exp_dummies],axis=1)


# In[30]:


df.columns


# In[31]:


df.corr()['fraudulent'].sort_values()


# In[32]:


df['employment_type'].value_counts()


# In[33]:


plt.figure(figsize=(12,10))
ax=sns.countplot(x='employment_type',hue='fraudulent',data=df)
plt.legend(bbox_to_anchor=(1, 1.1),title='Fraudulent', loc=2, borderaxespad=0.)
plt.tight_layout()


# In[34]:


emp_type_dummies = pd.get_dummies(df['employment_type'],drop_first=True)


# In[35]:


df = pd.concat([df.drop('employment_type',axis=1),emp_type_dummies],axis=1)


# In[36]:


df.columns


# In[37]:


df.corr()


# In[38]:


df.head()


# In[110]:


#We are not going to deal with text columns in this project
#df=df.drop(['job_id','company_profile','description','requirements','benefits'],axis=1)


# In[112]:


df.head()


# In[113]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[114]:


industry_new=pd.DataFrame(df[['industry']])


# In[115]:


for i in industry_new:
    if industry_new[i].dtype=='object':
        industry_new[i]=le.fit_transform(industry_new[i])
        
industry_new.head()


# In[116]:


function_new=pd.DataFrame(df
                          [['function']])


# In[117]:


for i in function_new:
    if function_new[i].dtype=='object':
        function_new[i]=le.fit_transform(function_new[i])


# In[118]:


function_new.head()


# In[119]:


df=df.drop(['industry','function'],axis=1)


# In[120]:


df=pd.concat([df,industry_new,function_new],axis=1)


# In[121]:


df.columns


# In[123]:


df.head()


# In[56]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(14, 6), dpi=100)

length1=df[df["fraudulent"]==1]['company_profile'].str.len()


ax1.hist(length1,bins = 30,color='orangered')
ax1.set_title('Fake')
ax1.set_xlabel('Num of Char')

length0=df[df["fraudulent"]==0]['company_profile'].str.len()


ax2.hist(length0, bins = 30)
ax2.set_title('Real')
ax2.set_xlabel('Num of Char')
fig.suptitle('Characters in Company Profile')
plt.show()


# In[57]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(14, 6), dpi=100)

length1=df[df["fraudulent"]==1]['description'].str.len()


ax1.hist(length1,bins = 30,color='orangered')
ax1.set_title('Fake')
ax1.set_xlabel('Num of Char')

length0=df[df["fraudulent"]==0]['description'].str.len()


ax2.hist(length0, bins = 30)
ax2.set_title('Real')
ax2.set_xlabel('Num of Char')
fig.suptitle('Characters in Description')
plt.show()


# In[58]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(14, 6), dpi=100)

length1=df[df["fraudulent"]==1]['requirements'].str.len()


ax1.hist(length1,bins = 30,color='orangered')
ax1.set_title('Fake')
ax1.set_xlabel('Num of Char')

length0=df[df["fraudulent"]==0]['requirements'].str.len()


ax2.hist(length0, bins = 30)
ax2.set_title('Real')
ax2.set_xlabel('Num of Char')
fig.suptitle('Characters in Requirements')
plt.show()


# In[59]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(14, 6), dpi=100)

length1=df[df["fraudulent"]==1]['benefits'].str.len()


ax1.hist(length1,bins = 30,color='orangered')
ax1.set_title('Fake')
ax1.set_xlabel('Num of Char')

length0=df[df["fraudulent"]==0]['benefits'].str.len()


ax2.hist(length0, bins = 30)
ax2.set_title('Real')
ax2.set_xlabel('Num of Char')
fig.suptitle('Characters in Benefits')
plt.show()


# In[60]:


plt.figure(figsize=(10,7))
sns.countplot(x ="telecommuting", hue="fraudulent", data=df)


# In[61]:


plt.figure(figsize=(10,7))
sns.countplot(x ="has_company_logo", hue="fraudulent", data=df)


# In[62]:


plt.figure(figsize=(10,7))
sns.countplot(x ="has_questions", hue="fraudulent", data=df)


# In[ ]:


#We are not going to deal with text columns in this project
df=df.drop(['job_id','company_profile','description','requirements','benefits'],axis=1)


# In[136]:


#Applying different machine learnng algorithms
from sklearn.model_selection import train_test_split

x=df.drop('fraudulent',axis=1)
y=df['fraudulent']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[137]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[139]:


predictions=rfc.predict(x_test)


# In[140]:


from sklearn.metrics import classification_report,confusion_matrix


# In[141]:


print(classification_report(y_test,predictions))


# In[142]:


print(confusion_matrix(y_test,predictions))


# In[143]:


from sklearn.tree import DecisionTreeClassifier


# In[144]:


dtree = DecisionTreeClassifier()


# In[145]:


dtree.fit(x_train,y_train)


# In[147]:


predictions_tree = dtree.predict(x_test)


# In[148]:


print(classification_report(y_test,predictions_tree))


# In[149]:


print(confusion_matrix(y_test,predictions_tree))


# In[150]:


from sklearn.linear_model import LogisticRegression


# In[162]:


logmodel = LogisticRegression(max_iter=700)
logmodel.fit(x_train,y_train)


# In[163]:


predictions_log = logmodel.predict(x_test)


# In[164]:


print(classification_report(y_test,predictions_log))


# In[165]:


print(confusion_matrix(y_test,predictions))


# In[166]:


from sklearn.neighbors import KNeighborsClassifier


# In[167]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[168]:


knn.fit(x_train,y_train)


# In[169]:


predictions_knn = knn.predict(x_test)


# In[170]:


print(classification_report(y_test,predictions_knn))


# In[171]:


print(confusion_matrix(y_test,predictions_knn))


# In[172]:


error_rate = []


for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[173]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[174]:


knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)
predictions_knn = knn.predict(x_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,predictions_knn))
print('\n')
print(classification_report(y_test,predictions_knn))


# In[175]:


from sklearn.preprocessing import MinMaxScaler


# In[176]:


scaler = MinMaxScaler()


# In[177]:


x_train = scaler.fit_transform(x_train)


# In[178]:


x_test = scaler.transform(x_test)


# In[179]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout


# In[181]:


x_train.shape


# In[182]:


model = Sequential()



model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))


model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[184]:


model.fit(x=x_train, 
          y=y_train, 
          epochs=600,
          validation_data=(x_test, y_test), verbose=1
          )


# In[185]:


model_loss = pd.DataFrame(model.history.history)


# In[186]:


model_loss.plot()


# In[251]:


# trained too much so now we will use early stopping
from tensorflow.keras.callbacks import EarlyStopping


# In[252]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[253]:


model.fit(x=x_train, 
          y=y_train, 
          epochs=600,
          validation_data=(x_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[254]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[255]:


predictions_ann =  (model.predict(x_test) > 0.2).astype("int32")


# In[256]:


print(classification_report(y_test,predictions_ann))


# In[240]:


from tensorflow.keras.layers import Dropout


# In[241]:


model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[242]:


model.fit(x=x_train, 
          y=y_train, 
          epochs=600,
          validation_data=(x_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[244]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[249]:


predictions_drop = (model.predict(x_test) > 0.1).astype("int32")


# In[250]:


print(classification_report(y_test,predictions_drop))


# In[257]:


# Now, we are going to resample our data
# under-sample
class_0=df[df['fraudulent']==0]
class_1=df[df['fraudulent']==1]


# In[258]:


class_0.shape


# In[259]:


class_1.shape


# In[260]:


class_0_under=class_0.sample(866)


# In[261]:


test_under=pd.concat([class_0_under,class_1],axis=0)


# In[262]:


x_under=pd.DataFrame(test_under.drop('fraudulent',axis=1))
y_under=pd.DataFrame(test_under['fraudulent'])


# In[264]:


x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, test_size=0.3, random_state=101,stratify=y_under)


# In[265]:


rfc_under=RandomForestClassifier()
rfc_under.fit(x_train,y_train.values.ravel())


# In[266]:


pred_under=rfc_under.predict(x_test)


# In[267]:


print(classification_report(y_test,pred_under))


# In[268]:


print(confusion_matrix(y_test,pred_under))


# In[285]:


# over-sample
class_1_over=class_1.sample(17014,replace=True)


# In[286]:


test_over=pd.concat([class_1_over,class_0],axis=0)


# In[287]:


x_over=pd.DataFrame(test_over.drop('fraudulent',axis=1))
y_over=pd.DataFrame(test_over['fraudulent'])


# In[288]:


x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.3, random_state=101,stratify=y_over)


# In[289]:


rfc_over=RandomForestClassifier()
rfc_over.fit(x_train,y_train.values.ravel())


# In[290]:


pred_over=rfc_over.predict(x_test)


# In[291]:


print(classification_report(y_test,pred_over))


# In[292]:


print(confusion_matrix(y_test,pred_over))


# In[ ]:


#The last algorithm is the most effective

