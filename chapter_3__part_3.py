import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
import re
from datetime import datetime
from dateutil import parser
from pandas.tseries.offsets import BDay
import numexpr
from pandas_datareader import data as pdr   # issue with Yahoo ecryption. use below code to fix
# ----- below code -----
import yfinance as yf
yf.pdr_override()
# ----- below code -----


#region Vectorized String Operations
# One strength of Python is its relative ease in handling and manipulating string data
# Pandas builds on this and provides a comprehensive set of vectorized string operations
# that become an essential piece of the type of munging required when one is working
# with (read: cleaning up) real-world data


#region Introducing Pandas String Operations
# tools like NumPy and Pandas generalize arithmetic operations 
# so that we can easily and quickly perform the same operation on many array elements
x = np.array([2, 3, 5, 7, 11, 13])
x * 2

# vectorization of operations simplifies the syntax of operating on arrays of data:
#   we no longer have to worry about the size or shape of the array, 
#   but just about what operation we want done

# For arrays of strings, NumPy does not provide such simple access, 
# and thus you’re stuck using a more verbose loop syntax
data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]


# This is perhaps sufficient to work with some data, 
# but it will break if there are any missing values
data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
[s.capitalize() for s in data]


# Pandas includes features to address both 
#   this need for vectorized string operations
#   and for correctly handling missing data 
# via the str attribute of Pandas Series and Index objects containing strings

names = pd.Series(data); names

# can now call a single method that will capitalize all the entries, 
# while skipping over any missing values

names.str.capitalize()


#endregion

#region Tables of Pandas String Methods
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                    'Eric Idle', 'Terry Jones', 'Michael Palin'])


#region Methods similar to Python string methods
# Nearly all Python’s built-in string methods are mirrored by a Pandas vectorized string method
# have various return values

# Some, like lower(), return a series of strings
monte.str.lower()

# some others return numbers
monte.str.len()

# Or Boolean values
monte.str.startswith('T')

# Still others return lists or other compound values for each element
monte.str.split()


#endregion

#region Methods using regular expressions
# there are several methods that accept regular expressions 
# to examine the content of each string element, 
# and follow some of the API conventions of Python’s built-in re module 

# With these, you can do a wide range of interesting operations

# e.g. can extract the first name from each 
# by asking for a contiguous group of characters at the beginning of each element:
monte.str.extract('([A-Za-z]+)')

# can do something more complicated, like
# finding all names that start and end with a consonant, 
# making use of the start-of-string (^) and end-of-string ($) regular expression characters
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')

# ability to concisely apply regular expressions across Series or DataFrame entries
# opens up many possibilities for analysis and cleaning of data


#endregion

#region Miscellaneous methods
# there are some miscellaneous methods that enable other convenient operations

#region Vectorized item access and slicing
# get() and slice() operations, in particular,
#   enable vectorized element access from each array
#       e.g. can get a slice of the first three characters of each array using str.slice(0, 3)
# this behavior is also available through Python’s normal indexing syntax
#   e.g. df.str.slice(0, 3) is equivalent to df.str[0:3]
monte.str[0:3]
# Indexing via df.str.get(i) and df.str[i] is similar


# These get() and slice() methods also let you access elements of arrays returned by split()

# e.g. to extract the last name of each entry, we can combine split() and get()
monte.str.split().str.get(-1)


#endregion

#region Indicator variables
# get_dummies() method
#   is useful when your data has a column containing some sort of coded indicator

# e.g. might have a dataset that contains information in the form of codes, such as 
#   A=“born in America,” B=“born in the United Kingdom,” C=“likes cheese,” D=“likes spam”
full_monte = pd.DataFrame({'name': monte,
                            'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
full_monte

# get_dummies() routine 
#   lets you quickly split out these indicator variables into a DataFrame
full_monte['info'].str.get_dummies('|')

# With these operations as building blocks, 
# you can construct an endless range of string processing procedures when cleaning your data


#endregion


#endregion


#endregion

#region Example: Recipe Database
# These vectorized string operations become most useful in the process of cleaning up messy, real-world data

# goal will be to parse the recipe data into ingredient lists, 
# so we can quickly find a recipe based on some ingredients we have on hand

# database is in JSON format, so we will try pd.read_json to read it
try:
    recipes = pd.read_json('Data/20170107-061401-recipeitems.json')
except ValueError as e:
    print("ValueError:", e)

# get a ValueError mentioning that there is “trailing data.” 
# Searching for this error on the Internet, 
#   it seems that it’s due to using a file in which each line is itself a valid JSON, 
#   but the full file is not. 
# Let’s check if this interpretation is true

with open('Data/20170107-061401-recipeitems.json') as f:
    line = f.readline()
pd.read_json(line).shape

# Yes, apparently each line is a valid JSON, so we’ll need to string them together

# One way we can do this is to actually construct a string representation containing all these JSON entries,
# and then load the whole thing with pd.read_json

# read the entire file into a Python array
with open('Data/20170107-061401-recipeitems.json', 'r', encoding="utf-8") as f:
    # Extract each line
    data = (line.strip() for line in f)
    # Reformat so each line is the element of a list
    data_json = "[{0}]".format(','.join(data))
# read the result as a JSON
recipes = pd.read_json(data_json)
recipes.shape


# look at one row to see what we have
recipes.iloc[0]

# There is a lot of information there, 
#   but much of it is in a very messy form, 
#   as is typical of data scraped from the Web
# In particular, the ingredient list is in string format;
#   we’re going to have to carefully extract the information we’re interested in

# Let’s start by taking a closer look at the ingredients
recipes.ingredients.str.len().describe()

# ingredient lists average 250 characters long, 
# with a minimum of 0 and a maximum of nearly 10,000 characters!

# let’s see which recipe has the longest ingredient list
recipes.name[np.argmax(recipes.ingredients.str.len())]

# can do other aggregate explorations

# e.g. let’s see how many of the recipes are for breakfast food
recipes.description.str.contains('[Bb]reakfast').sum()

# Or how many of the recipes list cinnamon as an ingredient
recipes.ingredients.str.contains('[Cc]innamon').sum()

# see whether any recipes misspell the ingredient as “cinamon”
recipes.ingredients.str.contains('[Cc]inamon').sum()

# This is the type of essential data exploration that is possible with Pandas string tools.
# It is data munging like this that Python really excels at


#region A simple recipe recommender
# start working on a simple recipe recommendation system:
# given a list of ingredients, find a recipe that uses all those ingredients

# task is complicated by the heterogeneity of the data: 
#   e.g. there is no easy operation to extract a clean list of ingredients from each row

spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
                'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
# can then build a Boolean DataFrame consisting of True and False values, 
#   indicating whether this ingredient appears in the list
spice_df = pd.DataFrame(dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE))
                                        for spice in spice_list))
spice_df.head()

# let’s say we’d like to find a recipe that uses parsley, paprika, and tarragon
# can compute this very quickly using the query() method of Data Frames
selection = spice_df.query('parsley & paprika & tarragon')
len(selection)

# use the index returned by this selection to discover the names of the recipes that have this combination
recipes.name[selection.index]


#endregion

#region Going further with recipes
# points to the truism that in data science, cleaning and munging of real-world data 
# often comprises the majority of the work, and 
# Pandas provides the tools that can help you do this efficiently


#endregion


#endregion


#endregion

#region Working with Time Series
# Pandas was developed in the context of financial modeling, so as you might expect, 
# it contains a fairly extensive set of tools for working with dates, times, and timeindexed data

# Date and time data comes in a few flavors
#   Time stamps reference particular moments in time
#   Time intervals and periods reference a length of time between a particular beginning and end point
#       Periods usually reference a special case of time intervals in which each interval is of uniform length and does not overlap
#   Time deltas or durations reference an exact length of time


#region Dates and Times in Python
# Python world has a number of available representations of dates, times, deltas, and timespans
# While the time series tools provided by Pandas tend to be the most useful for data science applications, 
#   it is helpful to see their relationship to other packages used in Python


#region Native Python dates and times: datetime and dateutil
# Python’s basic objects for working with dates and times reside in the built-in datetime module
# Along with the third-party dateutil module, 
#   you can use it to quickly perform a host of useful functionalities on dates and times
# related package to be aware of is pytz, 
#   which contains tools for working with the most migraine-inducing piece of time series data: time zones

# manually build a date using the datetime type
datetime(year=2015, month=7, day=4)

# using the dateutil module, you can parse dates from a variety of string formats
date = parser.parse("4th of July, 2015"); date

# Once you have a datetime object, you can do things like printing the day of the week
date.strftime('%A')


# power of datetime and dateutil lies in their flexibility and easy syntax: 
# you can use these objects and their built-in methods 
#   to easily perform nearly any operation you might be interested in
# Where they break down is when you wish to work with large arrays of dates and times
#   lists of Python datetime objects are suboptimal compared to typed arrays of encoded dates


#endregion

#region Typed arrays of times: NumPy’s datetime64
# datetime64 dtype encodes dates as 64-bit integers, 
#   and thus allows arrays of dates to be represented very compactly

# datetime64 requires a very specific input format
date = np.array('2015-07-04', dtype=np.datetime64); date


# Once we have this date formatted, however,
#   we can quickly do vectorized operations on it
date + np.arange(12)

# Because of the uniform type in NumPy datetime64 arrays, 
#   this type of operation can be accomplished much more quickly 
#   than if we were working directly with Python’s datetime objects, 
#   especially as arrays get large


# One detail of the datetime64 and timedelta64 objects is that they are built on a fundamental time unit
# Because the datetime64 object is limited to 64-bit precision, 
#   the range of encodable times is 2**64 times this fundamental uni
#  In other words, datetime64 imposes a trade-off between time resolution and maximum time span
# e.g.  if you want a time resolution of one nanosecond, 
#   you only have enough information to encode a range of 264 nanoseconds, or just under 600 years
# NumPy will infer the desired unit from the input

np.datetime64('2015-07-04')         # day-based datetime
np.datetime64('2015-07-04 12:00')   # minute-based datetime

# time zone is automatically set to the local time on the computer executing the code
# can force any desired fundamental unit using one of many format codes

# e.g. force a nanosecond-based time
np.datetime64('2015-07-04 12:59:59.50', 'ns')


# For the types of data we see in the real world, 
# a useful default is datetime64[ns], 
# as it can encode a useful range of modern dates with a suitably fine precision
#   Time span (relative) = ± 292 years          Time span (absolute) = [1678 AD, 2262 AD]

# while the datetime64 data type addresses some of the deficiencies of the built-in Python datetime type, 
# it lacks many of the convenient methods and functions provided by datetime and especially dateutil


#endregion

#region Dates and times in Pandas: Best of both worlds
# Pandas builds upon all the tools just discussed 
# to provide a Timestamp object, which combines
#   the ease of use of datetime and dateutil with 
#   the efficient storage and vectorized interface of numpy.datetime64

# From a group of these Timestamp objects,
# Pandas can construct a DatetimeIndex 
#   that can be used to index data in a Series or DataFrame

# e.g. parse a flexibly formatted string date, 
#   and use format codes to output the day of the week
date = pd.to_datetime("4th of July, 2015"); date
date.strftime('%A')

# can do NumPy-style vectorized operations directly on this same object
date + pd.to_timedelta(np.arange(12), 'D')


#endregion


#endregion

#region Pandas Time Series: Indexing by Time
# Where the Pandas time series tools really become useful is when you begin to index data by timestamps

# e.g. construct a Series object that has timeindexed data
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                            '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index); data

# Now that we have this data in a Series, 
# we can make use of any of the Series indexing patterns we discussed in previous sections, 
# passing values that can be coerced into dates
data['2014-07-04':'2015-07-04']

# There are additional special date-only indexing operations

# such as passing a year to obtain a slice of all data from that year
data['2015']


#endregion

#region Pandas Time Series Data Structures
# fundamental Pandas data structures for working with time series data
# • For time stamps, Pandas provides the Timestamp type. 
#       it is essentially a replacement for Python’s native datetime, 
#       but is based on the more efficient numpy.datetime64 data type. 
#       The associated index structure is DatetimeIndex.
# • For time periods, Pandas provides the Period type. 
#       This encodes a fixed-frequency interval based on numpy.datetime64. 
#       The associated index structure is PeriodIndex.
# • For time deltas or durations, Pandas provides the Timedelta type. 
#       Timedelta is a 
#           more efficient replacement for Python’s native datetime.timedelta type, and 
#           is based on numpy.timedelta64. 
#       The associated index structure is TimedeltaIndex.


# most fundamental of these date/time objects are the Timestamp and DatetimeIndex objects
# While these class objects can be invoked directly, 
#   it is more common to use the pd.to_datetime() function, 
#       which can parse a wide variety of formats
# Passing a single date to pd.to_datetime() yields a Timestamp; 
# passing a series of dates by default yields a DatetimeIndex

dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                            '2015-Jul-6', '07-07-2015', '20150708'])
dates


# Any DatetimeIndex can be converted to a PeriodIndex with the 
#   to_period() function with the 
#   addition of a frequency code

dates.to_period('D')    # use 'D' to indicate daily frequency

# TimedeltaIndex is created when one date is subtracted from another
dates - dates[0]


#region Regular sequences: pd.date_range()
# To make the creation of regular date sequences more convenient, 
# Pandas offers a few functions for this purpose: 
#   pd.date_range() for timestamps, 
#   pd.period_range() for periods, and 
#   pd.timedelta_range() for time deltas
# All of these require an understanding of Pandas frequency codes


# pd.date_range() accepts 
#   a start date, 
#   an end date, and 
#   an optional frequency code 
# to create a regular sequence of dates
# By default, the frequency is one day

pd.date_range('2015-07-03', '2015-07-10')

# Alternatively, the date range can be specified not with a start- and endpoint, 
#   but with a startpoint and a number of periods
pd.date_range('2015-07-03', periods=8)

# can modify the spacing by altering the freq argument, which defaults to D

pd.date_range('2015-07-03', periods=8, freq='H')    # e.g. construct a range of hourly timestamps


# To create regular sequences of period or time delta values, the very similar
# pd.period_range() and pd.timedelta_range() functions are useful

pd.period_range('2015-07', periods=8, freq='M')     # monthly periods

pd.timedelta_range(0, periods=10, freq='H')     # sequence of durations increasing by an hour


#endregion



#endregion

#region Frequencies and Ofsets
# Fundamental to these Pandas time series tools is the concept of a frequency or date offset
# can use such codes to specify any desired frequency spacing

# monthly, quarterly, and annual frequencies are all marked at the end of the specified period
# Adding an S suffix to any of these marks it instead at the beginning

# can change the month used to mark any quarterly or annual code by adding a three-letter month code as a suffix
#  can modify the split-point of the weekly frequency by adding a three-letter weekday code

# codes can be combined with numbers to specify other frequencies

# e.g. for a frequency of 2 hours 30 minutes, we can combine the hour (H) and minute (T) codes as follows
pd.timedelta_range(0, periods=9, freq='2H30T')

# All of these short codes refer to specific instances of Pandas time series offsets, 
# which can be found in the pd.tseries.offsets module

# e.g. create a business day offset directly as follows
from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())


#endregion

#region Resampling, Shifting, and Windowing
# Because Pandas was developed largely in a finance context, 
#   it includes some very specific tools for financial data

# e.g. accompanying pandas-datareader package
#   knows how to import financial data from a number of available sources

'''
from pandas_datareader import data
goog = data.DataReader('GOOG', start='2004', end='2016', data_source='google')

-- issues
---- google currently not supported
---- issue with Yahoo ecryption. use below code to fix
----- below code -----
import yfinance as yf
yf.pdr_override()
data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
----- below code -----

'''

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
goog = pdr.get_data_yahoo('GOOG', start='2004-01-01', end='2016-01-01')
goog.head()

# For simplicity, we’ll use just the closing price
goog = goog['Close']


# can visualize this using the plot() method, after the normal Matplotlib setup boilerplate
goog.plot()
plt.show()


#region Resampling and converting frequencies
# common need for time series data is resampling at a higher or lower frequency
# can do this using 
#   the resample() method, or the much simpler 
#   asfreq() method
# primary difference between the two is that 
#   resample() is fundamentally a data aggregation, while 
#   asfreq() is fundamentally a data selection


# let’s compare what the two return when we down-sample the data
# will resample the data at the end of business year
goog.plot(alpha=0.8, style='-')
goog.resample('BA').mean().plot(style=':')
goog.asfreq('BA').plot(style='--')
plt.legend(['input', 'resample', 'asfreq'], loc='upper left')
plt.show()

# resample reports the average of the previous year, while
# asfreq reports the value at the end of the year

# For up-sampling, resample() and asfreq() are largely equivalent, 
# though resample has many more options available
# default for both methods is 
# to leave the up-sampled points empty
#   that is, filled with NA values
# asfreq() accepts a method argument to specify how values are imputed


# resample the business day data at a daily frequency (i.e., including weekends)
fig, ax = plt.subplots(2, sharex=True)
data = goog.iloc[:10]

data.asfreq('D').plot(ax=ax[0], marker='o')
data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
ax[1].legend(["back-fill", "forward-fill"])

plt.show()


#endregion

#region Time-shifts
# Another common time series–specific operation is shifting of data in time
# Pandas has two closely related methods for computing this: 
#   shift() and 
#   tshift()
# difference between them is that 
#   shift() shifts the data, while 
#   tshift() shifts the index
# In both cases, the shift is specified in multiples of the frequency


# shift() and tshift() by 900 days
fig, ax = plt.subplots(3, sharey=True)

# apply a frequency to the data
goog = goog.asfreq('D', method='pad')

goog.plot(ax=ax[0])
goog.shift(900).plot(ax=ax[1])
goog.tshift(900).plot(ax=ax[2])

# legends and annotations
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')

ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=0.3, color='red')

ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')

ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[4].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red')

plt.show()

# shift(900) shifts the data by 900 days, 
#   pushing some of it off the end of the graph 
#   (and leaving NA values at the other end), while 
# tshift(900) shifts the index values by 900 days


# common context for this type of shift is computing differences over time

# e.g. use shifted values to compute the one-year return on investment for Google stock over the course of the dataset 
ROI = 100 * (goog.tshift(-365) / goog - 1)
ROI.plot()
plt.ylabel('% Return on Investments')
plt.show()


#endregion

#region Rolling windows
# Rolling statistics are a third type of time series–specific operation implemented by Pandas
# can be accomplished via the rolling() attribute of Series and Data Frame objects, 
#   which returns a view similar to what we saw with the groupby operation
# This rolling view makes available a number of aggregation operations by default


# e.g. one-year centered rolling mean and standard deviation of the Google stock prices
rolling = goog.rolling(365, center=True)
data = pd.DataFrame({'input': goog,
                        'one-year rolling_mean': rolling.mean(),
                        'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)
plt.show()


# aggregate() and apply() methods can be used for custom rolling computations


#endregion


#endregion

#region Where to Learn More


#endregion

#region Example: Visualizing Seattle Bicycle Counts
# use Pandas to read the CSV output into a DataFrame
# specify that we want the Date as an index, 
# and we want these dates to be automatically parsed

data = pd.read_csv('Data/Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)
data = data[data.index < datetime(2016, 8, 1)]  # select data only till 2016 to try match data in book
data.head(); data.shape

# For convenience, we’ll further process this dataset by shortening the column names
data.columns = ["Total", "East", "West"]

# look at the summary statistics for this data
data.dropna().describe()


#region Visualizing the data
# start by plotting the raw data
data.plot()
plt.ylabel('Hourly Bicycle Count')
plt.show()


# hourly samples are far too dense for us to make much sense of
# can gain more insight by resampling the data to a coarser grid
# resample by week
weekly = data.resample('W').sum()
weekly.plot(style=['-', '--', ':'])
plt.ylabel('Weekly Bicycle Count')
plt.show()


# Another way that comes in handy for aggregating the data is to use a rolling mean,
#   utilizing the pd.rolling_mean() function
# we’ll do a 30-day rolling mean of our data, 
#   making sure to center the window
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=['-', '--', ':'])
plt.ylabel('mean hourly count')
plt.show()


# jaggedness of the result is due to the hard cutoff of the window
# can get a smoother version of a rolling mean using a window function
#   for example, a Gaussian window

# specifies both
#   the width of the window (chose 50 days)
#   and the width of the Gaussian within the window (chose 10 days)
daily.rolling(50, center=True, win_type='gaussian').sum(std=10).plot(style=['-', '--', ':'])
plt.show()


#endregion

#region Digging into the data
# While the smoothed data views are useful to get an idea of the general trend in the data,
# they hide much of the interesting structure

# e.g. might want to look at the average traffic as a function of the time of day
# can do this using the GroupBy functionality
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks=hourly_ticks, style=['-', '--', ':'])
plt.show()


# might be curious about how things change based on the day of the week.
# Again, we can do this with a simple groupby
by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
by_weekday.plot(style=['-', '--', ':'])
plt.show()


# let’s do a compound groupby and look at the hourly trend on weekdays versus weekends
# start by grouping by both a flag marking the weekend, and the time of day
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays', xticks=hourly_ticks, style=['-', '--', ':'])
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekend', xticks=hourly_ticks, style=['-', '--', ':'])

plt.show()


#endregion


#endregion


#endregion

#region High-Performance Pandas: eval() and query()
# power of the PyData stack is built 
# upon the ability of NumPy and Pandas 
# to push basic operations into C via an intuitive syntax
# e.g. vectorized/broadcasted operations in NumPy, and
#       grouping-type operations in Pandas

# While these abstractions are efficient and effective for many common use cases, 
# they often rely on the creation of temporary intermediate objects, 
# which can cause undue overhead in computational time and memory use

# Pandas includes some tools 
# that allow you to directly access C-speed operations 
# without costly allocation of intermediate arrays

# These are the eval() and query() functions, 
# which rely on the Numexpr package


#region Motivating query() and eval(): Compound Expressions
# NumPy and Pandas support fast vectorized operations

# e.g. adding the elements of two arrays
rng = np.random.RandomState(42)
x = rng.rand(int(1E6))
y = rng.rand(int(1E6))
timeit.timeit('x + y', globals=globals(), number=100)

# this is much faster than doing the addition via a Python loop or comprehension
timeit.timeit('np.fromiter((xi + yi for xi, yi in zip(x, y)), dtype=x.dtype, count=len(x))', globals=globals(), number=100)

# this abstraction can become less efficient when you are computing compound expressions

# e.g.  
mask = (x > 0.5) & (y < 0.5)
# Because NumPy evaluates each subexpression, 
# this is roughly equivalent to the following
tmp1 = (x > 0.5)
tmp2 = (y < 0.5)
mask = tmp1 & tmp2

# In other words, every intermediate step is explicitly allocated in memory
# If the x and y arrays are very large, this can lead to significant memory and computational overhead

# Numexpr library gives you the ability to compute this type of compound expression element by element,
#   without the need to allocate full intermediate arrays
# library accepts a string giving the NumPy-style expression you’d like to compute
# benefit here is that Numexpr evaluates the expression in a way that does not use full-sized temporary arrays, 
#   and thus can be much more efficient than NumPy, especially for large arrays

import numexpr
mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
np.allclose(mask, mask_numexpr)


# Pandas eval() and query() tools are conceptually similar, 
#   and depend on the Numexpr package


#endregion

#region pandas.eval() for Eicient Operations
# eval() function in Pandas uses string expressions to efficiently compute operations using DataFrames

nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))

# To compute the sum of all four DataFrames using the typical Pandas approach, we can just write the sum
timeit.timeit('df1 + df2 + df3 + df4', globals=globals(), number=10)

# can compute the same result via pd.eval by constructing the expression as a string
timeit.timeit("pd.eval('df1 + df2 + df3 + df4')", globals=globals(), number=10)

np.allclose(df1 + df2 + df3 + df4, pd.eval('df1 + df2 + df3 + df4'))


#region Operations supported by pd.eval()
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3))) for i in range(5))


#region Arithmetic operators
# pd.eval() supports all arithmetic operators
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)


#endregion

#region Comparison operators
# pd.eval() supports all comparison operators, including chained expressions
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)


#endregion

#region Bitwise operators
# pd.eval() supports the & and | bitwise operators
result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)


# supports the use of the literal and and or in Boolean expressions
result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1, result3)


#endregion

#region Object attributes and indices
# pd.eval() supports access to 
#   object attributes via the obj.attr syntax, and 
#   indexes via the obj[index] syntax
result1 = df2.T[0] + df3.iloc[1]
result2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)


#endregion


#endregion

#endregion

#region DataFrame.eval() for Column-Wise Operations
# Just as Pandas has a top-level pd.eval() function, 
# DataFrames have an eval() method that works in similar ways

# benefit of the eval() method is that columns can be referred to by name


df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
df.head()

# Using pd.eval() as above, we can compute expressions with the three columns like this
result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
np.allclose(result1, result2)

# DataFrame.eval() method allows much more succinct evaluation of expressions with the columns
result3 = df.eval('(A + B) / (C - 1)')
np.allclose(result1, result3)


#region Assignment in DataFrame.eval()
# DataFrame.eval() also allows assignment to any column
df.head()

# use df.eval() to create a new column 'D' and assign to it a value computed from the other columns
df.eval('D = (A + B) / C', inplace=True); df.head()

# any existing column can be modified
df.eval('D = (A - B) / C', inplace=True); df.head()


#endregion

#region Local variables in DataFrame.eval()
# DataFrame.eval() method supports an additional syntax that lets it work with local Python variables
column_mean = df.mean(1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean')
np.allclose(result1, result2)


# @ character here marks a variable name rather than a column name, 
# and lets you efficiently evaluate expressions involving the two “namespaces”: 
#   the namespace of columns, and 
#   the namespace of Python objects

# @ character is only supported by the DataFrame.eval() method, not by the pandas.eval() function,
#   because the pandas.eval() function only has access to the one (Python) namespace


#endregion


#endregion

#region DataFrame.query() Method
# DataFrame has another method based on evaluated strings, called the query() method
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1, result2)

# It cannot be expressed using the DataFrame.eval() syntax, however!
# Instead, for this type of filtering operation, you can use the query() method
#   compared to the masking expression this is much easier to read and understand
result2 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result2)


# query() method also accepts the @ flag to mark local variables
Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')
np.allclose(result1, result2)


#endregion

#region Performance: When to Use These Functions
# there are two considerations: 
#   computation time and 
#   memory use

# Memory use is the most predictable aspect
# every compound expression involving NumPy arrays or Pandas DataFrames
#   will result in implicit creation of temporary arrays
# If the size of the temporary DataFrames 
#   is significant compared to your available system memory (typically several gigabytes), 
#   then it’s a good idea to use an eval() or query() expression
# can check the approximate size of your array in bytes using
df.values.nbytes


# On the performance side, 
#   eval() can be faster 
#       even when you are not maxing out your system memory
# issue is how your temporary DataFrames compare to the
#   size of the L1 or L2 CPU cache on your system
# if they are much bigger, 
#   then eval() can avoid some potentially slow movement of values between the different memory caches
# find that the difference in computation time 
#   between the traditional methods and the eval/query method
#   is usually not significant
#   if anything, the traditional method is faster for smaller arrays! 
# benefit of eval/query is 
#   mainly in the saved memory, and 
#   the sometimes cleaner syntax they offer


#endregion


#endregion

#region Further Resources

#endregion
