#!/usr/bin/env python
# coding: utf-8

# ## Data

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[2]:


#Loading Dataset
df = pd.read_csv('C:/Users/17pol/Downloads/book (2).csv')


# ## EDA

# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


df.shape


# In[ ]:


#Unique Users


# In[8]:


df['User.ID'].nunique()


# In[ ]:


#Unique books


# In[9]:


df['Book.Title'].nunique()


# In[ ]:


#rating values


# In[10]:


df['Book.Rating'].nunique()


# In[11]:


sns.countplot(df['Book.Rating'])


# In[13]:


#Average Book Rating

Avg_rating = round(df['Book.Rating'].mean(), 2)
print('Average rating for all books is ', Avg_rating)


# In[14]:


# Drop Unnamed: 0 column
df = df.drop('Unnamed: 0', axis = 1)


# In[16]:


# Renaming columns
df = df.rename({'User.ID':'User_id', 'Book.Title':'Book_title', 'Book.Rating':'Book_rating'}, axis = 1)


# In[22]:


books_df = df.pivot_table(index='User_id',
                                 columns='Book_title',
                                 values='Book_rating').reset_index(drop=True)
books_df


# In[24]:


books_df.index = df['User_id'].unique()
books_df


# In[25]:


#Impute those NaNs with 0 values
books_df.fillna(0, inplace=True)
books_df


# In[26]:


#Calculating Cosine Similarity between Users
user_sim = 1 - pairwise_distances(books_df.values,metric='cosine')


# In[27]:


user_sim


# In[28]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
#Set the index and column names to user ids 
user_sim_df.index = df.User_id.unique()
user_sim_df.columns = df.User_id.unique()
user_sim_df.iloc[0:5, 0:5]


# In[29]:


# Fillimg diagonal values with 0
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[30]:


#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]


# In[32]:


df[(df['User_id']==276729) | (df['User_id']==276726)]


# In[34]:


user_1=df[df['User_id']==276726]
user_2=df[df['User_id']==276729]
pd.merge(user_1,user_2,on='Book_title',how='outer')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




