#!/usr/bin/env python
# coding: utf-8

# ## Prepare rules for the all the data sets 
# 1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values
# 2) Change the minimum length in apriori algorithm
# 3) Visulize the obtained rules using different plots 
# 
# 
# 

# In[1]:


import pandas as pd
movies = pd.read_csv('C:/Users/17pol/Downloads/my_movies (1).csv')
movies.head()


# In[3]:


pip install mlxtend


# ## EDA

# In[2]:


movies.sample(10)


# In[3]:


movies.shape


# In[4]:


movies.info()


# In[5]:


movies.isna().sum()


# In[6]:


movies.columns


# ## Data Preprocessing

# In[7]:


movies = movies.drop(['V1','V2','V3','V4','V5'], axis = 1)


# In[8]:


movies.head()


# In[9]:


movie_count = []
movie_names = movies.columns
for name in movie_names:
    movie_count.append(movies[name].value_counts()[1])
movie_count


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15,5))
plt.title('Movie Counts')
plt.xlabel('Movies')
plt.bar(movie_names, movie_count)


# ## Apriori Algorithm

# In[14]:


from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ### Apriori for min_support 0.1

# In[15]:


frequent_itemsets = apriori(movies, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[17]:


frequent_itemsets1 = apriori(movies, min_support = 0.1, use_colnames=True)
frequent_itemsets1['length'] = frequent_itemsets1['itemsets'].apply(lambda x: len(x))
frequent_itemsets1


# ### Rules when min_support = 0.1 and min_threshold for lift is 1

# In[20]:


rules1 = association_rules(frequent_itemsets1, metric='lift', min_threshold=1)
rules1


# In[21]:


rules1 = rules1.sort_values(['confidence', 'lift'], ascending =[False, False])
rules1


# #### Visualizing

# In[22]:


plt.scatter(rules1['support'], rules1['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# ### Rules when min_support = 0.1 and min_threshold for confidence is 0.5

# In[24]:


rules2 = association_rules(frequent_itemsets1, metric='confidence', min_threshold=0.5)
rules2


# In[25]:


rules2 = rules2.sort_values(['confidence', 'lift'], ascending =[False, False])
rules2


# #### Visualizing

# In[26]:


plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# ## Apriori Algorithm for min_support = 0.2

# In[27]:


frequent_itemsets2 = apriori(movies, min_support= 0.2, use_colnames=True)
frequent_itemsets2


# ### Rules when min_support = 0.2 and min_threshold for lift is 0.5

# In[28]:


rules3 = association_rules(frequent_itemsets2, metric= 'lift', min_threshold=0.5)
rules3


# In[29]:


rules3 = rules3.sort_values(['confidence','lift'], ascending=[False, False])
rules3


# #### Visualizing

# In[30]:


plt.scatter(rules3['support'], rules3['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[ ]:




