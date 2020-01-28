#!/usr/bin/env python
# coding: utf-8

# # Tesla Monte Carlo share price simulation

# ### References: https://www.investopedia.com/articles/07/montecarlo.asp , https://datascienceplus.com/how-to-apply-monte-carlo-simulation-to-forecast-stock-prices-using-python/
# ### YFinance: https://pypi.org/project/yfinance/

# In[304]:


# download yahoo finance API (for share price data quite good - less so for fundamentals)
pip install yfinance


# In[305]:


# download quandl contains everything: FX, options, fundamentals, macroeconmic indicators, forecasts, ....
# BUT MONTHLY PAYMENT
pip install quandl


# In[306]:


# import packages needed
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[307]:


# separate package to be used for fundamentals
import quandl
quandl.ApiConfig.api_key = 'Kevin'


# In[308]:


Tesla = yf.Ticker('TSLA')
print(Tesla)


# In[309]:


Tesla.info


# In[310]:


historical_price_Tesla = Tesla.history(period = 'max')


# In[311]:


historical_price_Tesla


# ### Historical share price charts

# In[312]:


Tesla_shr_prc = historical_price_Tesla['Close']


# In[313]:


Tesla_shr_prc.head()


# #### Long-term

# In[314]:


# 10 years
Test_price_chart = Tesla_shr_prc.plot(figsize=(10, 6))


# #### Short-term

# In[315]:


tesla_short_term = tesla.history(period = '1y',interval = '5d')
tesla_short_term.head()


# In[316]:


# 1 year
Tesla_price_chart_1y = tesla_short_term['Close'].plot(figsize=(10, 6))


# ### Calculate returns

# In[317]:


log_returns = np.log(1 + Tesla_shr_prc.pct_change())
# quick overview
log_returns.head()


# ### Returns distribution

# In[318]:


log_returns.plot(figsize = (10, 6))


# In[319]:


# Observations from above graph:
# 1) constant mean
# 2) returns follow normal distribution
# Hence we can use the Gerneral Brownian motion model


# ### Formula of MC 

# In[320]:


from PIL import Image


# In[321]:


MC_explanation = Image.open('/Users/user/Documents/Uni 19:20/LUIFS/MC_graph.png')


# In[322]:


MC_explanation
# delta S = change in share price
# mu = expected return
# delta t = time step
# omega = volatility
# Epsilon = random term


# ### Calculate inputs needed

# ##### Statistical values

# In[323]:


mean = log_returns.mean()
mean


# In[324]:


var = log_returns.var()
var


# In[325]:


stdev = log_returns.std()
stdev


# In[326]:


# drift element (slightly different from above formula but same reasoning)
drift = mean - (0.5 * var)
drift


# In[327]:


# !!for conceptual clearness!!

# equivalent to 95% confidence interval
norm.ppf(0.95)

# in other words this event is 1.65 std away from the mean


# In[328]:


# create random 10 rows by 2 columns matrix (numbers range between 0 and 1)
x = np.random.rand(10, 2)
x


# In[329]:


# same reasoning as above
# each data point is one observation/event


# In[330]:


norm.ppf(x)


# #### Alternatively...

# In[331]:


# combining both straight into one function (= is the same as the two steps above but just the reduced version)
Z = norm.ppf(np.random.rand(10,2))
Z
# Reminder: the array represents the distances from the mean expressed by std which in turn got generated randomly (since random walk)


# #### Create 10 different paths with each 1000 future share prices

# In[332]:


# n = last closing price available
n = 0 # since last day is today's day
days = n + 1000 # gives share price in 1000 days
iterations = 10 # gives 10 different paths


# #### Picture of MC formula needed

# In[333]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, iterations)))


# In[334]:


daily_returns
# is a numpy array


# In[335]:


# change to pandas to get an overview of set
df = pd.DataFrame(daily_returns) #,index=data[:,0])


# In[336]:


df.info()
df.head()


# In[337]:


# now get our last share price data point
Last_closing_share_price = Tesla_shr_prc[-1]
Last_closing_share_price


# In[338]:


# create empty matrix
price_list = np.zeros_like(daily_returns)


# In[339]:


price_list


# In[340]:


# need to set each starting share price to closing price of last available day
price_list[0] = Last_closing_share_price
price_list


# In[341]:


# Calculate share prices
for d in range(1,days):
    price_list[d] = price_list[d - 1] * daily_returns[d]


# In[342]:


price_list


# In[343]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# #### Share price predictions

# In[344]:


price_list[-1]


# In[345]:


np.mean(price_list[-1])


# In[346]:


np.median(price_list[-1])


# ### Now let's see with 100 paths

# In[347]:


# n = last closing price available
n = 0 # since last day is today's day
days = n + 1000 # gives share price in 1000 days
iterations = 100 # gives 100 different paths


# In[348]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, iterations)))


# In[349]:


daily_returns
# is a numpy array


# In[350]:


# now get our last share price data point
Last_closing_share_price = Tesla_shr_prc[-1]
Last_closing_share_price


# In[351]:


# create empty matrix
price_list = np.zeros_like(daily_returns)


# In[352]:


# need to set each starting share price to closing price of last available day
price_list[0] = Last_closing_share_price
price_list


# In[353]:


# Calculate share prices
for d in range(1,days):
    price_list[d] = price_list[d - 1] * daily_returns[d]


# In[354]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# #### Share price predictions

# In[355]:


price_list[-1]


# In[356]:


np.mean(price_list[-1])


# In[357]:


np.median(price_list[-1])


# ### FAANG + Tesla Analysis

# In[358]:


tickers = yf.Tickers('msft aapl goog nflx fb ndaq tsla')


# In[359]:


tickers


# In[360]:


FAANG_Data = tickers.history(period='1y',interval='5d')


# In[361]:


# get an overview
FAANG_Data.head()


# In[362]:


# just focus on closing prices since other columns are irrelevant for our analysis
FAANG_closing_shr_prc = FAANG_Data.iloc[:,0:7] # first two columns of data frame with all row


# In[363]:


FAANG_closing_shr_prc.info()


# In[364]:


# delete rows where no data is available
FAANG_closing_shr_prc = FAANG_closing_shr_prc.dropna()
# drop level 1 header
FAANG_closing_shr_prc.columns = FAANG_closing_shr_prc.columns.droplevel(0)


# In[365]:


FAANG_closing_shr_prc.head()


# In[366]:


FAANG_closing_shr_prc.plot()


# In[367]:


AAPL_closing_shr_prc = FAANG_closing_shr_prc['AAPL']
NFLX_closing_shr_prc = FAANG_closing_shr_prc['NFLX']
MSFT_closing_shr_prc = FAANG_closing_shr_prc['MSFT']
GOOG_closing_shr_prc = FAANG_closing_shr_prc['GOOG']
FB_closing_shr_prc = FAANG_closing_shr_prc['FB']
NDAQ_closing_shr_prc = FAANG_closing_shr_prc['NDAQ']
TSLA_closing_shr_prc = FAANG_closing_shr_prc['TSLA']


# In[368]:


# overview facebook dataset
FB_closing_shr_prc.head()


# In[369]:


plt.figure(figsize=(10,6))
plt.plot(MSFT_closing_shr_prc)


# In[370]:


#NOTE: as we can see above graph not very illustrative so need to rebase to 1


# In[371]:


# Rebasing to 100
MSFT_closing_shr_prc_re = MSFT_closing_shr_prc.div(MSFT_closing_shr_prc.iloc[0], axis=0).mul(100)
AAPL_closing_shr_prc_re = AAPL_closing_shr_prc.div(AAPL_closing_shr_prc.iloc[0], axis=0).mul(100)
NFLX_closing_shr_prc_re = NFLX_closing_shr_prc.div(NFLX_closing_shr_prc.iloc[0], axis=0).mul(100)
GOOG_closing_shr_prc_re = GOOG_closing_shr_prc.div(GOOG_closing_shr_prc.iloc[0], axis=0).mul(100)
FB_closing_shr_prc_re = FB_closing_shr_prc.div(FB_closing_shr_prc.iloc[0], axis=0).mul(100)
NDAQ_closing_shr_prc_re = NDAQ_closing_shr_prc.div(NDAQ_closing_shr_prc.iloc[0], axis=0).mul(100)
TSLA_closing_shr_prc_re = TSLA_closing_shr_prc.div(TSLA_closing_shr_prc.iloc[0], axis=0).mul(100)


# In[372]:


# for illustrative purposes
MSFT_closing_shr_prc_re.head()


# In[373]:


Rebased_FAANG = pd.concat([TSLA_closing_shr_prc_re, NDAQ_closing_shr_prc_re,MSFT_closing_shr_prc_re,AAPL_closing_shr_prc_re, NFLX_closing_shr_prc_re, GOOG_closing_shr_prc_re, FB_closing_shr_prc_re], axis=1)


# In[374]:


Rebased_FAANG.plot()


# ### Calculate statistics

# #### Returns

# In[375]:


TSLA_log_returns = np.log(1 + TSLA_closing_shr_prc.pct_change())
FB_log_returns = np.log(1 + FB_closing_shr_prc.pct_change())
GOOG_log_returns = np.log(1 + GOOG_closing_shr_prc.pct_change())
AAPL_log_returns = np.log(1 + AAPL_closing_shr_prc.pct_change())
NFLX_log_returns = np.log(1 + NFLX_closing_shr_prc.pct_change())
MSFT_log_returns = np.log(1 + MSFT_closing_shr_prc.pct_change())
NDAQ_log_returns = np.log(1 + NDAQ_closing_shr_prc.pct_change())


# In[376]:


TSLA_log_returns.head()


# In[377]:


# quick method but is simple % returns (i.e. not logarithmic)
FAANG_closing_shr_prc.pct_change().head()


# In[378]:


# now logarithmic returns
returns_data = np.log(FAANG_closing_shr_prc).diff()


# #### Correlations

# In[379]:


# correlations matrix
returns_data.corr()


# In[380]:


# let's define the axis for the scatterplot
y = returns_data['NDAQ']
a = returns_data['AAPL']


# In[381]:


# Plot
plt.scatter(a, y,alpha=0.5)
plt.title('Nasdaq against Apple')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# #### FAANG  portfolio

# In[382]:


# let's define the axis for the scatterplot
y = returns_data['NDAQ']
a = returns_data['AAPL']
g = returns_data['GOOG']
f = returns_data['FB']
m = returns_data['MSFT']
n = returns_data['NFLX']
# t = returns_data['TSLA']


# In[383]:


# FAANG equal % portfolio
portfolio = 0.2 * a + 0.2 * g + 0.2 * f + 0.2 * m + 0.2 * n


# In[384]:


portfolio.head()


# In[385]:


portfolio.mean()


# In[386]:


portfolio.std()


# In[387]:


y.mean()


# In[388]:


y.std()


# ### Other examples (if time - mostly Yahoo finance functions)

# In[389]:


msft = yf.Ticker('MSFT')


# In[390]:


msft.info


# In[391]:


hist = msft.history(period="max")
hist.tail()


# #### ESG

# In[392]:


msft.sustainability


# #### Recommendations

# In[393]:


tesla = yf.Ticker('TSLA')


# In[394]:


tesla.recommendations.tail()


# In[395]:


tesla.major_holders

