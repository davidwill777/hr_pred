#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier    
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

train = pd.read_csv('./sa_pre_model.csv',header=0)

#windows = 100 #for other
#steps = 35
windows = 1500 #for sa
steps = 35

df = []

train = train.drop(['第二個月未帶卡次數','第三個月病假時數','第二個月遺失卡次數','第二個月遲到次數'
            ,'第三個月曠職時數','第二個月曠職時數','第三個月遲到次數'],axis=1)

#print(train)
cols = ['年齡','姓別代號','台成清交(最高)','理工科系(最高)','台成清交(次高)','理工科系(次高)','外語專長種類數','個人專長種類數','體育專長種類數'
       ,'婚姻代號','撫養人數','第三個月事假時數','第三個月特休時數'
       ,'第三個月未帶卡次數','第三個月遺失卡次數','第三個月忘刷卡次數','第二個月事假時數','第二個月病假時數','第二個月特休時數'
       ,'第二個月忘刷卡次數','近一年考績','第三個月案件催辦次數','第二個月案件催辦次數'
       ,'第三個月評核分數','第二個月評核分數','是否離職']



classifier = AdaBoostClassifier(
    DecisionTreeClassifier(criterion='gini',class_weight='balanced',max_depth=10),
    n_estimators=50
)

st = RandomOverSampler(random_state=1)

data = np.arange(len(train))

def rolling(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

a = rolling(data,windows)
#print(a)

scores_roc = []
scores_accuracy = []
k = 0

for i in range(0,a.shape[0],steps): 
    for j in range(0,a.shape[1],1): #100

       df.append(train.iloc[a[i][j],:])
       #print(i,j,a[i][j])
       k = k + 1

       if k == windows:
            
            test_1=pd.DataFrame(columns=cols,data=df)
            
            x = test_1[cols]
            x = x.drop(['是否離職'],axis=1)
            x_train_pos_data = x.iloc[:int(len(x) * 0.7)].copy()        
            x_test_pos_data = x.iloc[int(len(x) * 0.7):].copy()
         
            
            y = test_1['是否離職']
            y_train_pos_data = y.iloc[:int(len(y) * 0.7)].copy()            
            y_test_pos_data = y.iloc[int(len(x) * 0.7):].copy()   
       
            
            train_resample,train_resample_target = st.fit_sample(x_train_pos_data,y_train_pos_data)
            classifier.fit(train_resample,train_resample_target)
            #classifier.fit(x_train_pos_data,y_train_pos_data)
            test_pred = classifier.predict(x_test_pos_data)
            
            try:
                scores_accuracy.append(accuracy_score(y_test_pos_data,test_pred))
            except ValueError:
                pass   
            
            try:
                scores_roc.append(roc_auc_score(y_test_pos_data,test_pred))
            except ValueError:
                pass       
            
            k = 0
            df = []


# In[33]:


plt.figure(figsize=(10,5))
plt.title('sa rolling model roc_scores')
plt.xlabel('times')
plt.ylabel('Test_roc_scores')
plt.plot(scores_roc)


# In[34]:


plt.figure(figsize=(10,5))
plt.title('sa rolling model accuracy')
plt.xlabel('times')
plt.ylabel('Test_acc')
plt.plot(scores_accuracy)

