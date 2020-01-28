#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages needed
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


NN = yf.Ticker('NVO')


# In[5]:


NN.info


# In[8]:


historical_price_NN = NN.history(period = '10y')


# In[9]:


historical_price_NN


# In[10]:


NN_last_10y = historical_price_NN['Close']


# In[11]:


NN_last_10y


# In[12]:


# 10 years
Test_price_chart = NN_last_10y.plot(figsize=(10, 6))


# In[13]:


log_returns = np.log(1 + NN_last_10y.pct_change())
# quick overview
log_returns.head()


# In[14]:


log_returns.plot(figsize = (10, 6))


# In[15]:


mean = log_returns.mean()
mean


# In[16]:


var = log_returns.var()
var


# In[17]:


stdev = log_returns.std()
stdev


# In[18]:


drift = mean - (0.5 * var)
drift


# In[19]:


# !!for conceptual clearness!!

# equivalent to 95% confidence interval
norm.ppf(0.95)

# in other words this event is 1.65 std away from the mean


# In[20]:


# n = last closing price available
n = 0 # since last day is today's day
days = n + 100 # gives share price in 100 days
iterations = 100 # gives 100 different paths


# In[21]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, iterations)))


# In[22]:


daily_returns
# is a numpy array


# In[23]:


# now get our last share price data point
Last_closing_share_price = NN_last_10y[-1]
Last_closing_share_price


# In[24]:


# create empty matrix
price_list = np.zeros_like(daily_returns)


# In[25]:


# need to set each starting share price to closing price of last available day
price_list[0] = Last_closing_share_price
price_list


# In[26]:


# Calculate share prices
for d in range(1,days):
    price_list[d] = price_list[d - 1] * daily_returns[d]


# In[27]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# In[28]:


price_list[-1]


# In[29]:


meanprice = np.mean(price_list[-1])


# In[30]:


medianprice = np.median(price_list[-1])


# In[31]:


returnmean = (meanprice - Last_closing_share_price)/Last_closing_share_price


# In[33]:


returnmedian = (medianprice - Last_closing_share_price)/Last_closing_share_price


# In[34]:


returnmean


# In[35]:


returnmedian


# In[ ]:


# 1000 paths


# In[36]:


# n = last closing price available
n = 0 # since last day is today's day
days = n + 100 # gives share price in 100 days
iterations = 1000 # gives 100 different paths


# In[37]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, iterations)))


# In[38]:


daily_returns
# is a numpy array


# In[39]:


# now get our last share price data point
Last_closing_share_price = NN_last_10y[-1]
Last_closing_share_price


# In[40]:


# create empty matrix
price_list = np.zeros_like(daily_returns)


# In[41]:


# need to set each starting share price to closing price of last available day
price_list[0] = Last_closing_share_price
price_list


# In[42]:


# Calculate share prices
for d in range(1,days):
    price_list[d] = price_list[d - 1] * daily_returns[d]


# In[43]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# In[44]:


price_list[-1]


# In[45]:


meanprice = np.mean(price_list[-1])


# In[46]:


medianprice = np.median(price_list[-1])


# In[47]:


returnmean = (meanprice - Last_closing_share_price)/Last_closing_share_price


# In[48]:


returnmedian = (medianprice - Last_closing_share_price)/Last_closing_share_price


# In[49]:


returnmean


# In[50]:


returnmedian


# In[ ]:


# 10000 paths


# In[51]:


# n = last closing price available
n = 0 # since last day is today's day
days = n + 100 # gives share price in 100 days
iterations = 10000 # gives 100 different paths


# In[52]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, iterations)))


# In[53]:


daily_returns
# is a numpy array


# In[54]:


# now get our last share price data point
Last_closing_share_price = NN_last_10y[-1]
Last_closing_share_price


# In[55]:


# create empty matrix
price_list = np.zeros_like(daily_returns)


# In[56]:


# need to set each starting share price to closing price of last available day
price_list[0] = Last_closing_share_price
price_list


# In[57]:


# Calculate share prices
for d in range(1,days):
    price_list[d] = price_list[d - 1] * daily_returns[d]


# In[58]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# In[59]:


price_list[-1]


# In[60]:


meanprice = np.mean(price_list[-1])


# In[61]:


medianprice = np.median(price_list[-1])


# In[62]:


returnmean = (meanprice - Last_closing_share_price)/Last_closing_share_price


# In[63]:


returnmedian = (medianprice - Last_closing_share_price)/Last_closing_share_price


# In[64]:


returnmean


# In[65]:


returnmedian


# In[ ]:


# 100000 paths


# In[66]:


# n = last closing price available
n = 0 # since last day is today's day
days = n + 100 # gives share price in 100 days
iterations = 100000 # gives 100 different paths


# In[67]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, iterations)))


# In[68]:


daily_returns
# is a numpy array


# In[69]:


# now get our last share price data point
Last_closing_share_price = NN_last_10y[-1]
Last_closing_share_price


# In[70]:


# create empty matrix
price_list = np.zeros_like(daily_returns)


# In[71]:


# need to set each starting share price to closing price of last available day
price_list[0] = Last_closing_share_price
price_list


# In[72]:


# Calculate share prices
for d in range(1,days):
    price_list[d] = price_list[d - 1] * daily_returns[d]


# In[73]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# In[74]:


price_list[-1]


# In[75]:


meanprice = np.mean(price_list[-1])


# In[76]:


medianprice = np.median(price_list[-1])


# In[77]:


returnmean = (meanprice - Last_closing_share_price)/Last_closing_share_price


# In[78]:


returnmedian = (medianprice - Last_closing_share_price)/Last_closing_share_price


# In[79]:


returnmean


# In[80]:


returnmedian


# In[ ]:




