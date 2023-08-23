#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


main = pd.read_csv("questions.csv")
main.shape


# In[5]:


df = main.sample(30000,random_state=2)


# In[6]:


df.sample(7)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


nan_in_col  = df[df['question1'].isna()]
nan_in_col


# In[11]:


null_rows = df[df.isnull().any(axis=1)]
print(null_rows)


# In[12]:


df.duplicated().sum()


# In[13]:


print(df['is_duplicate'].value_counts())
df['is_duplicate'].value_counts().plot(kind='bar')


# In[14]:


qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print("no of unique questions:", np.unique(qid).shape[0])
x = qid.value_counts() > 1
print("no of repeated questions", x[x].shape[0])


# In[15]:


plt.hist(qid.value_counts().values,bins=100)
plt.yscale('log')
plt.show()


# In[16]:


df['q1_len'] = df['question1'].str.len() 
df['q2_len'] = df['question2'].str.len()


# In[17]:


df.tail()


# In[18]:


df['q1_num_words'] = df['question1'].apply(lambda x: len(str(x).split()))
df['q2_num_words'] = df['question2'].apply(lambda x: len(str(x).split()))


# In[19]:


df.tail()


# In[20]:


df['common_word_count'] = df.apply(lambda row: len(set(str(row['question1']).split()) & set(str(row['question2']).split())), axis=1)


# In[21]:


df.head()


# In[22]:


df['total_word_count'] = df.apply(lambda row: len(set(str(row['question1']).split(" "))), axis=1)+df.apply(lambda row:len(set(str(row['question2']).split(" "))), axis=1)


# In[23]:


df.head()


# In[24]:


df["word_share"] = round(df["common_word_count"]/df["total_word_count"],4)
df.tail()


# In[25]:


ques_df = df[['question1', 'question2']]
ques_df.tail()


# In[26]:


final_df = df.drop(columns = ['id', 'qid1', 'qid2', 'question1', 'question2'])
final_df.shape


# In[27]:


final_df.head()


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer

# Merge texts
questions = list(ques_df['question1'].fillna('')) + list(ques_df['question2'].fillna(''))

# Apply Bag of Words model
cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)

# Create DataFrame for both question features
temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)

# Concatenate vectorized DataFrame with the newly added feature DataFrame
final_df = pd.concat([final_df, temp_df], axis=1)

print(final_df.shape)
final_df.head()


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)
print("trained")


# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:




