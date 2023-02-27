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


#region Combining Datasets: Concat and Append
# simple concatenation of Series and DataFrames with the pd.concat function

# define this function, 
#   which creates a DataFrame of a particular form that will be useful below
def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

# example DataFrame
make_df('ABC', range(3))


#region Recall: Concatenation of NumPy Arrays
# Concatenation of Series and DataFrame objects 
# is very similar to concatenation of NumPy arrays, 
#   which can be done via the np.concatenate function

# can combine the contents of two or more arrays into a single array
# first argument is a list or tuple of arrays to concatenate
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])

# takes an axis keyword 
#   that allows you to specify the axis along which the result will be concatenated
x = [[1, 2],
[3, 4]]
np.concatenate([x, x], axis=1)


#endregion

#region Simple Concatenation with pd.concat
# pd.concat() 
#   can be used for a simple concatenation of Series or DataFrame objects,
# just as np.concatenate() 
#   can be used for simple concatenations of arrays

ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])

# also works to concatenate higher-dimensional objects, such as DataFrames
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
df1; df2; pd.concat([df1, df2])

# By default, the concatenation takes place row-wise within the DataFrame (i.e., axis=0)

# pd.concat allows specification of an axis along which concatenation will take place
# could have equivalently specified axis=1; 
#   here we’ve used the more intuitive axis='columns'
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
df3; df4; pd.concat([df3, df4], axis='columns')


#region Duplicate indices
# One important difference between np.concatenate and pd.concat is that 
#   Pandas concatenation preserves indices, 
#       even if the result will have duplicate indices!

x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index   # make duplicate indices!
x; y; pd.concat([x, y])


# While this is valid within DataFrames, 
#   the outcome is often undesirable. 
# pd.concat() gives us a few ways to handle it


#region Catching the repeats as an error
# If you’d like to simply verify 
# that the indices in the result of pd.concat() do not overlap, 
# you can specify the verify_integrity flag
#   With this set to True, 
#   the concatenation will raise an exception if there are duplicate indices

try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)


#endregion

#region Ignoring the index
# Sometimes the index itself does not matter, 
#   and you would prefer it to simply be ignored
# can specify this option using the ignore_index flag
#   With this set to True, 
#       the concatenation will create a new integer index for the resulting Series

x; y; pd.concat([x, y], ignore_index=True)


#endregion

#region Adding MultiIndex keys
# Another alternative is to
#   use the keys option to specify a label for the data sources; 
#   the result will be a hierarchically indexed series containing the data
# result is a multiply indexed DataFrame

x; y; pd.concat([x, y], keys=['x', 'y'])


#endregion

#region Concatenation with joins
# data from different sources might have different sets of column names, 
# and pd.concat offers several options in this case

# Consider the concatenation of the following two DataFrames, 
#   which have some (but not all!) columns in common
# By default, the entries for which no data is available are filled with NA values
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
df5; df6; pd.concat([df5, df6])


# To change this, we can specify one of several options for the 
#   join and join_axes parameters of the concatenate function

# By default, the join is a union of the input columns (join='outer'), 
# but we can change this to an intersection of the columns using join='inner'
df5; df6; pd.concat([df5, df6], join='inner')

# Another option is to directly specify the index of the remaining colums 
df5; df6; pd.concat([df5, df6.reindex(columns=df5.columns)])


#endregion


#endregion


#endregion


#endregion

#region Combining Datasets: Merge and Join
# One essential feature offered by Pandas is its 
#   high-performance, in-memory join and merge operations
# main interface for this is the pd.merge function


#region Relational Algebra
# behavior implemented in pd.merge() is a subset of what is known as relational algebra, 
#   which is a formal set of rules for manipulating relational data, 
#   and forms the conceptual foundation of operations available in most databases

# strength of the relational algebra approach is that it proposes several primitive operations, 
#   which become the building blocks of more complicated operations on any dataset

# Pandas implements several of these fundamental building blocks in the 
#   pd.merge() function and the related 
#   join() method of Series and DataFrames


#endregion

#region Categories of Joins
# pd.merge() function implements a number of types of joins: the 
#   one-to-one,
#   many-to-one, and 
#   many-to-many joins

# All three types of joins are accessed via an identical call to the pd.merge() interface; 
# the type of join performed depends on the form of the input data


#region One-to-one joins
# is in many ways very similar to the column-wise concatenation 

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df1; df2

# To combine this information into a single DataFrame, we can use the pd.merge() function
df3 = pd.merge(df1, df2); df3

# pd.merge() function recognizes that each DataFrame has an “employee” column,
#   and automatically joins using this column as a key
# result of the merge is a new DataFrame that combines the information from the two inputs
# order of entries in each column is not necessarily maintained
# merge in general discards the index, 
#   except in the special case of merges by index


#endregion

#region Many-to-one joins
# For the many-to-one case, the resulting DataFrame will preserve those duplicate entries as appropriate

df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
df3; df4; pd.merge(df3, df4)


#endregion

#region Many-to-many joins
# Many-to-many joins are a bit confusing conceptually, but are nevertheless well defined

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                                'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                                'spreadsheets', 'organization']})
df1; df5; pd.merge(df1, df5)


#endregion


#endregion

#region Speciication of the Merge Key

#region The on keyword
# can explicitly specify the name of the key column using the on keyword, 
#   which takes a column name or a list of column names
# This option works only if
#   both the left and right DataFrames have the specified column name

df1; df2; pd.merge(df1, df2, on='employee')


#endregion

#region The left_on and right_on keywords
# At times you may wish to merge two datasets with different column names
# In this case, we can use the left_on and right_on keywords to specify the two column names

df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
df1; df3; pd.merge(df1, df3, left_on='employee', right_on='name')


# result has a redundant column that we can drop if desired
# e.g. by using the drop() method of DataFrames
pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis=1)


#endregion

#region The left_index and right_index keywords
# Sometimes, rather than merging on a column, you would instead like to merge on an index
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')

# can use the index as the key for merging 
# by specifying the left_index and/or right_index flags in pd.merge()
pd.merge(df1a, df2a, left_index=True, right_index=True)


# For convenience, 
# DataFrames implement the join() method, 
#   which performs a merge that defaults to joining on indices
df1a; df2a; df1a.join(df2a)


# If you’d like to mix indices and columns, 
# you can combine 
#   left_index with right_on or
#   left_on with right_index
# to get the desired behavior
df1a; df3; pd.merge(df1a, df3, left_index=True, right_on='name')


#endregion


#endregion

#region Specifying Set Arithmetic for Joins
# one important consideration in performing a join: 
#   the type of set arithmetic used in the join
# This comes up when a value appears in one key column but not the other

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                    columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                    columns=['name', 'drink'])
df6; df7; pd.merge(df6, df7)


# By default, an inner join
# can specify this explicitly using the how keyword,
#   which defaults to 'inner'
pd.merge(df6, df7, how='inner')

# Other options for the how keyword are 'outer', 'left', and 'right'
df6; df7; pd.merge(df6, df7, how='outer')
df6; df7; pd.merge(df6, df7, how='left')
df6; df7; pd.merge(df6, df7, how='right')


#endregion

#region Overlapping Column Names: The suixes Keyword
# may end up in a case where your two input DataFrames have conflicting column names
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
df8; df9; pd.merge(df8, df9, on='name')

# Because the output would have two conflicting column names, 
# the merge function automatically appends a suffix _x or _y to make the output columns unique
# If these defaults are inappropriate, 
#   it is possible to specify a custom suffix using the suffixes keyword
# work also if there are multiple overlapping columns
df8; df9; pd.merge(df8, df9, on='name', suffixes=["_L", "_R"])


#endregion

#region Example: US States Data
pop = pd.read_csv('data\state-population.csv')
areas = pd.read_csv('data\state-areas.csv')
abbrevs = pd.read_csv('data\state-abbrevs.csv')
pop.head(); areas.head(); abbrevs.head()


# say we want to compute a relatively straightforward result:
#   rank US states and territories by their 2010 population density

# start with a many-to-one merge that will give us the full state name within the population DataFrame
# use how='outer' to make sure no data is thrown away due to mismatched labels
merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1)    # drop duplicate info
merged.head()

# Let’s double-check whether there were any mismatches here, 
#   which we can do by looking for rows with nulls
merged.isnull().any()

# Some of the population info is null; let’s figure out which these are!
merged[merged['population'].isnull()].head()
# It appears that all the null population values are from Puerto Rico prior to the year 2000;
# this is likely due to this data not being available from the original source

# More importantly, we see also that some of the new state entries are also null,
# which means that there was no corresponding entry in the abbrevs key!
# Let’s figure out which regions lack this match
merged.loc[merged['state'].isnull(), 'state/region'].unique()

# can quickly infer the issue: 
#   our population data includes entries for Puerto Rico (PR) and the United States as a whole (USA), 
#   while these entries do not appear in the state abbreviation key

# can fix these quickly by filling in appropriate entries
merged.loc[merged['state/region']=='PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region']=='USA', 'state'] = 'United States'
merged.isnull().any()
# No more nulls in the state column: we’re all set!


# can merge the result with the area data using a similar procedure
final = pd.merge(merged, areas, on='state', how='left')
final.head()


# let’s check for nulls to see if there were any mismatches
final.isnull().any()

# There are nulls in the area column; 
# we can take a look to see which regions were ignored here
final.loc[final['area (sq. mi)'].isnull(), 'state/region'].unique()
final['state'][final['area (sq. mi)'].isnull()].unique()


# see that our areas DataFrame does not contain the area of the United States as a whole
# could insert the appropriate value (using the sum of all state areas, for instance), 
# but in this case we’ll just drop the null values 
#   because the population density of the entire United States is not relevant to our current discussion
final.dropna(inplace=True)
final.head()

# Now we have all the data we need

# To answer the question of interest, 
# let’s first select 
#   the portion of the data corresponding with the year 2010, and 
#   the total population
# use the query() function to do this quickly 
#   (this requires the numexpr package to be installed)
data2010 = final.query('year == 2010 & ages == "total"')
data2010.head()


# compute the population density and display it in order
# start by 
#   reindexing our data on the state, and
#   then compute the result
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']

density.sort_values(ascending=False, inplace=True)
density.head()
# result is a ranking of US states plus Washington, DC, and Puerto Rico 
# in order of their 2010 population density, 
# in residents per square mile


# can also check the end of the list
density.tail()


#endregion


#endregion

#region Aggregation and Grouping

#region Planets Data
# Planets dataset gives information on planets that astronomers
#   have discovered around other stars 
#       (known as extrasolar planets or exoplanets for short)
planets = sns.load_dataset('planets')
planets.shape

planets.head()


#endregion

#region Simple Aggregation in Pandas
# As with a onedimensional NumPy array, 
# for a Pandas Series 
#   the aggregates return a single value
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5)); ser

ser.sum(); ser.mean()


# For a DataFrame, by default 
#   the aggregates return results within each column
df = pd.DataFrame({'A': rng.rand(5),
                    'B': rng.rand(5)})
df

df.mean()

# By specifying the axis argument, you can instead aggregate within each row
df.mean(axis='columns')


# there is a convenience method describe() 
#   that computes several common aggregates for each column and returns the result
# can be a useful way to begin understanding the overall properties of a dataset

planets.dropna().describe()


#endregion

#region GroupBy: Split, Apply, Combine
# to aggregate conditionally on some label or index: 
# this is implemented in the socalled groupby operation
#   think of it in the terms first coined by Hadley Wickham of Rstats fame: 
#       split, 
#       apply, 
#       combine


#region Split, apply, combine
# what the GroupBy accomplishes:
#   • The split step involves breaking up and grouping a DataFrame depending on the value of the specified key.
#   • The apply step involves computing some function, usually an aggregate, transformation, or filtering, within the individual groups.
#   • The combine step merges the results of these operations into an output array

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                    'data': range(6)},
                    columns=['key', 'data'])
df

# can compute the most basic split-apply-combine operation with the groupby() method of DataFrames, 
#   passing the name of the desired key column
# what is returned is not a set of DataFrames, but a DataFrameGroupBy object
#   can think of it as a special view of the DataFrame, 
#   which is poised to dig into the groups 
#   but does no actual computation until the aggregation is applied
# This “lazy evaluation” approach means that 
#   common aggregates can be implemented very efficiently 
#   in a way that is almost transparent to the user

df.groupby('key')


# To produce a result, we can apply an aggregate to this DataFrameGroupBy object,
# which will perform the appropriate apply/combine steps to produce the desired result
df.groupby('key').sum()


#endregion

#region The GroupBy object
# GroupBy object is a very flexible abstraction
# can simply treat it as if it’s a collection of DataFrames, 
#   and it does the difficult things under the hood

# most important operations made available by a GroupBy are 
#   aggregate,
#   filter, 
#   transform, and 
#   apply


#region Column indexing
# GroupBy object supports column indexing in the same way as the DataFrame
# returns a modified GroupBy object

planets.groupby('method')
planets.groupby('method')['orbital_period']

# here selected a particular Series group from the original DataFrame group by reference to its column name

# no computation is done until we call some aggregate on the object
planets.groupby('method')['orbital_period'].mean()


#endregion

#region Iteration over groups
# GroupBy object supports direct iteration over the groups,
#   returning each group as a Series or DataFrame
# This can be useful for doing certain things manually, 
#   though it is often much faster to use the built-in apply functionality


for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))


#endregion

#region Dispatch methods
# Through some Python class magic, 
#   any method not explicitly implemented by the GroupBy object 
#   will be passed through and called on the groups,
#   whether they are DataFrame or Series objects

# e.g. can use the describe() method of DataFrames 
#   to perform a set of aggregations 
#   that describe each group in the data

planets.groupby('method')['year'].describe()
# Looking at this table helps us to better understand the data


# This is just one example of the utility of dispatch methods
# they are applied to each individual group, and 
#   the results are then combined within GroupBy and returned
# any valid DataFrame/Series method can be used on the corresponding GroupBy object, 
#   which allows for some very flexible and powerful operations!


#endregion


#endregion

#region Aggregate, filter, transform, apply
# GroupBy objects have 
#   aggregate(),
#   filter(), 
#   transform(), and 
#   apply() methods 
# that efficiently implement a variety of useful operations before combining the grouped data

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                    'data1': range(6),
                    'data2': rng.randint(0, 10, 6)},
                    columns = ['key', 'data1', 'data2'])
df


#region Aggregation
# aggregate() method allows for even more flexibility. 
# It can take a string, a function, or a list thereof, and compute all the aggregates at once

df.groupby('key').aggregate(['min', np.median, max])


# Another useful pattern is 
# to pass a dictionary mapping 
#   column names to 
#   operations to be applied on that column

df.groupby('key').aggregate({'data1': 'min',
                                'data2': 'max'})


#endregion

#region Filtering
# filtering operation allows you to drop data based on the group properties
# filter() function should return a Boolean value specifying whether the group passes the filtering

# e.g. might want to keep all groups in which the standard deviation is larger than some critical value
def filter_func(x):
    return x['data2'].std() > 4

df; df.groupby('key').std(); df.groupby('key').filter(filter_func)


#endregion

#region Transformation
# While aggregation must return a reduced version of the data, 
# transformation can return some transformed version of the full data to recombine
# For such a transformation, the output is the same shape as the input

# e.g. center the data by subtracting the group-wise mean
df.groupby('key').transform(lambda x: x - x.mean())


#endregion

#region The apply() method
# apply() method lets you apply an arbitrary function to the group results
# function should 
#   take a DataFrame, and 
#   return 
#       either a Pandas object (e.g., DataFrame, Series) or 
#       a scalar; 
# the combine operation will be tailored to the type of output returned

# # apply() within a GroupBy is quite flexible: 
#   the only criterion is that the function takes a DataFrame and returns a Pandas object or scalar; 
#   what you do in the middle is up to you!


# e.g. normalizes the first column by the sum of the second
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x

df; df.groupby('key').apply(norm_by_data2)


#endregion


#endregion

#region Specifying the split key

#region A list, array, series, or index providing the grouping keys
#  key can be any series or list with a length matching that of the DataFrame

L = [0, 1, 0, 1, 2, 0]
df; df.groupby(L).sum()


# Of course, this means there’s another, more verbose way of accomplishing the df.groupby('key') from before
df; df.groupby(df['key']).sum()


#endregion

#region A dictionary or series mapping index to group
# Another method is to provide a dictionary
#   that maps index values to the group keys

df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
df2; df2.groupby(mapping).sum()


#endregion

#region Any Python function
# can pass any Python function that will input the index value and output the group

df2; df2.groupby(str.lower).mean()


#endregion

#region A list of valid keys
# any of the preceding key choices can be combined to group on a multi-index

df2.groupby([str.lower, mapping]).mean()


#endregion


#endregion

#region Grouping example
# in a couple lines of Python code we can put all these together and
# count discovered planets by method and by decade

decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'

planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)


#endregion


#endregion


#endregion

#region Pivot Tables
# pivot table takes simple columnwise data as input, 
# and groups the entries into a two-dimensional table 
# that provides a multidimensional summarization of the data

# difference between pivot tables and GroupBy can sometimes cause confusion; 
# it helps to think of pivot tables as essentially a multidimensional version of GroupBy aggregation
# That is, you split-apply-combine, 
#   but both the split and the combine happen across 
#       not a one-dimensional index, 
#       but across a two-dimensional grid


#region Motivating Pivot Tables
# database of passengers on the Titanic
titanic = sns.load_dataset('titanic')
titanic.head()


#endregion

#region Pivot Tables by Hand
# e.g. let’s look at survival rate by gender
titanic.groupby('sex')['survived'].mean()


#  might like to go one step deeper and look at survival by both sex and, say, class
titanic.groupby(['sex', 'class'])['survived'].mean().unstack()
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()

# This gives us a better idea of how both gender and class affected survival, 
# but the code is starting to look a bit garbled
# While each step of this pipeline makes sense in light of the tools we’ve previously discussed, 
# the long string of code is not particularly easy to read or use

# This two-dimensional GroupBy is common enough that Pandas includes a convenience routine, pivot_table, 
# which succinctly handles this type of multidimensional aggregation


#endregion

#region Pivot Table Syntax
# Here is the equivalent to the preceding operation
# using the pivot_table method of DataFrames
titanic.pivot_table('survived', index='sex', columns='class')

# This is eminently more readable than the GroupBy approach, 
# and produces the same result


#region Multilevel pivot tables
# grouping in pivot tables can be specified with multiple levels, 
# and via a number of options

# e.g. might be interested in looking at age as a third dimension
#  bin the age using the pd.cut function
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')


# can apply this same strategy when working with the columns as well

# let’s add info on the fare paid using pd.qcut to automatically compute quantiles
fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

# result is a four-dimensional aggregation with hierarchical indices
# shown in a grid demonstrating the relationship between the values


#endregion

#region Additional pivot table options
# Two of the options, fill_value and dropna, 
#   have to do with missing data and are fairly straightforward

# aggfunc keyword 
#   controls what type of aggregation is applied, 
#   which is a mean by default

# aggregation specification can be 
#   a string representing one of several common choices or
#   a function that implements an aggregation
#   can be specified as a dictionary mapping a column to any of the above desired options
# omitted the values keyword; 
#   when you’re specifying a mapping for aggfunc, this is determined automatically

titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare': 'mean'})


# At times it’s useful to compute totals along each grouping
# can be done via the margins keyword
titanic.pivot_table('survived', index='sex', columns='class', margins=True)

# margin label can be specified with the margins_name keyword, 
#   which defaults to "All"


#endregion


#endregion

#region Example: Birthrate Data
# freely available data on births in the United States, provided by the Centers for Disease Control (CDC)
births = pd.read_csv('Data/births.csv')
births.head(); births.shape

# can start to understand this data a bit more by using a pivot table

# Let’s add a decade column, 
# and take a look at male and female births as a function of decade

births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')

# immediately see that male births outnumber female births in every decade
# To see this trend a bit more clearly, we can use the built-in plotting tools in Pandas 
#   to visualize the total number of births by year

sns.set()   # use Seaborn styles
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births by year')
plt.show()

# With a simple pivot table and plot() method, 
#   we can immediately see the annual trend in births by gender
# By eye, it appears that over the past 50 years
#   male births have outnumbered female births by around 5%


#region Further data exploration
# there are a few more interesting features we can pull out of this dataset 
# using the Pandas tools covered up to this point

# must start by cleaning the data a bit, 
#   removing outliers caused by mistyped dates (e.g., June 31st) or 
#   missing values (e.g., June 99th)

# One easy way to remove these all at once is to cut outliers; 
# we’ll do this via a robust sigma-clipping operation

quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
# This final line is a robust estimate of the sample mean, 
#   where the 0.74 comes from the interquartile range of a Gaussian distribution

# can use the query() method to filter out rows with births outside these values
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

# Next we set the day column to integers; 
#   previously it had been a string because some columns in the dataset contained the value 'null'
births['day'] = births['day'].astype(int)

# Finally, we can combine the day, month, and year to create a Date index
# This allows us to quickly compute the weekday corresponding to each row

# create a datetime index from the year, month, day
births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format='%Y%m%d')
#   births.year[0]; births.month[0]; births.day[0]; 10000 * births.year[0] + 100 * births.month[0] + births.day[0]
births['dayofweek'] = births.index.dayofweek


# Using this we can plot births by weekday for several decades
births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot()
# both work.
# plt.gca().set_xticks(sorted(births['dayofweek'].unique()))
plt.gca().set_xticks(births['dayofweek'].drop_duplicates().sort_values())
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')
plt.show()

# Apparently births are slightly less common on weekends than on weekdays! 
# Note that the 1990s and 2000s are missing because the CDC data contains only the month of birth starting in 1989


# Another interesting view is to plot the mean number of births by the day of the year
# first group the data by month and day separately

# births.pivot_table('births', index=['month', 'day'])  # gives same result
births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
births_by_date.head()

# To make this easily plottable, 
# let’s turn these months and days into a date by associating them with a dummy year variable 
# (making sure to choose a leap year so February 29th is correctly handled!
births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]
births_by_date.head()

# Focusing on the month and day only, 
#   we now have a time series reflecting the average number of births by date of the year
# From this, we can use the plot method to plot the data

# Plot the results
fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax=ax)
plt.show()

# In particular, the striking feature of this graph is the dip in birthrate on US holidays
#   (e.g., Independence Day, Labor Day, Thanksgiving, Christmas, New Year’s Day)
# although this likely reflects trends in scheduled/induced births 
# rather than some deep psychosomatic effect on natural births


#endregion


#endregion


#endregion
