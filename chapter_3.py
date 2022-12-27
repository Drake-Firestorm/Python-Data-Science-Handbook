# CHAPTER 3: Data Manipulation with Pandas
# ========================================
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


# Pandas is a newer package 
#   built on top of NumPy, and 
#   provides an efficient implementation of a DataFrame

# DataFrames 
#   are essentially multidimensional arrays 
#   with attached row and column labels, and 
#   often with heterogeneous types and/or missing data
#   implements a number of powerful data operations
#       familiar to users of both database frameworks and spreadsheet programs


#region Installing and Using Pandas
# Installing Pandas on your system requires NumPy to be installed

# import Pandas under the alias pd and check the version
# import pandas as pd
pandas.__version__


#endregion

#region Introducing Pandas Objects
# At the very basic level, Pandas objects can be thought of as enhanced versions of NumPy structured arrays 
#   in which the rows and columns are identified with labels rather than simple integer indices


# three fundamental Pandas data structures: 
#   Series, 
#   DataFrame, and 
#   Index


# start our code sessions with the standard NumPy and Pandas imports


#region The Pandas Series Object
# Pandas Series is a one-dimensional array of indexed data

# can be created from a list or array
data = pd.Series([0.25, 0.5, 0.75, 1.0]); data

# Series wraps both 
#   a sequence of values and 
#   a sequence of indices, 
# which we can access with the values and index attributes

#   values are simply a familiar NumPy array
data.values

# index is an array-like object of type pd.Index
data.index

# data can be accessed by the associated index via the familiar Python square-bracket notation
data[1]
data[1:3]

# Pandas Series is much more general and flexible than the one-dimensional NumPy array that it emulates


#region Series as generalized NumPy array
# essential difference is the presence of the index: 
#   while the NumPy array has an implicitly deined integer index used to access the values, 
#   the Pandas Series has an explicitly deined index associated with the values

# This explicit index definition gives the Series object additional capabilities
# e.g. index need not be an integer, but can consist of values of any desired type

#   e.g. strings as an index
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                    index=['a', 'b', 'c', 'd'])
data

# item access works as expected
data['b']

# can even use noncontiguous or nonsequential indices
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                    index=[2, 5, 3, 7])
data
data[5]


#endregion

#region Series as specialized dictionary
# In this way, you can think of a Pandas Series a bit like a specialization of a Python dictionary
#   dictionary is a structure that maps arbitrary keys to a set of arbitrary values
#   Series is a structure that maps typed keys to a set of typed values

# typing is important: 
#   type information of a Pandas Series makes it much more efficient than 
#   Python dictionaries for certain operations

# can make the Series-as-dictionary analogy even more clear by 
#   constructing a Series object directly from a Python dictionary
population_dict = {
    'California': 38332521,
    'Texas': 26448193,
    'New York': 19651127,
    'Florida': 19552860,
    'Illinois': 12882135
}
population = pd.Series(population_dict)

# typical dictionary-style item access can be performed
population['California']

# Unlike a dictionary, though, the Series also supports array-style operations such as slicing
population['California':'Illinois']


#endregion

#region Constructing Series objects
#  few ways of constructing a Pandas Series from scratch; 
#   all of them are some version of the following
# where 
#   index is an optional argument, and 
#   data can be one of many entities
pd.Series(data, index=index)

# e.g. 
# data can be a list or NumPy array, in which case index defaults to an integer sequence
pd.Series([2, 4, 6])
# data can be a scalar, which is repeated to fill the specified index
pd.Series(5, index=[100, 200, 300])
# data can be a dictionary, in which index defaults to the sorted dictionary keys
pd.Series({2:'a', 1:'b', 3:'c'})


# In each case, the index can be explicitly set if a different result is preferred
#   in this case, the Series is populated only with the explicitly identified keys
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])


#endregion


#endregion

#region The Pandas DataFrame Object
# DataFrame can be thought of either
#   as a generalization of a NumPy array, or
#   as a specialization of a Python dictionary

#region DataFrame as a generalized NumPy array
# DataFrame is an analog of a two-dimensional array
# with both flexible row indices and flexible column names

# Just as you might think of a two-dimensional array as an ordered sequence of aligned one-dimensional columns, 
# you can think of a DataFrame as a sequence of aligned Series objects
#    “aligned” mean that they share the same index

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
                'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)

# along with the population Series from before, 
#   we can use a dictionary to construct a single two-dimensional object containing this information
states = pd.DataFrame({'population': population, 'area': area})


# DataFrame has an index attribute that gives access to the index labels
states.index

# DataFrame has a columns attribute, which is an Index object holding the column labels
states.columns

# Thus the DataFrame can be thought of as a generalization of a two-dimensional NumPy array, 
#   where both the rows and columns have a generalized index for accessing the data


#endregion

#region DataFrame as specialized dictionary
# Where a dictionary maps a key to a value, 
# a DataFrame maps a column name to a Series of column data

# e.g. asking for the 'area' attribute returns the Series object containing the areas we saw earlier
states['area']

# potential point of confusion here: 
#   in a two-dimensional NumPy array, data[0] will return the first row. 
#   For a DataFrame, data['col0'] will return the first column
# Because of this, 
#   it is probably better to think about DataFrames as generalized dictionaries rather than generalized arrays, 
#   though both ways of looking at the situation can be useful


#endregion

#region Constructing DataFrame objects
# can be constructed in a variety of ways

#region From a single Series object
# DataFrame is a collection of Series objects, 
# and a singlecolumn DataFrame can be constructed from a single Series

pd.DataFrame(population, columns=['population'])


#endregion

#region From a list of dicts
# Any list of dictionaries can be made into a DataFrame

data = [{'a': i, 'b': 2 * i} for i in range(3)]
pd.DataFrame(data)

# Even if some keys in the dictionary are missing, Pandas will fill them in with NaN (i.e., “not a number”) values
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


#endregion

#region From a dictionary of Series objects
# DataFrame can be constructed from a dictionary of Series objects as well
pd.DataFrame({'population': population, 'area': area})


#endregion

#region From a two-dimensional NumPy array
# Given a two-dimensional array of data, we can create a DataFrame with any specified column and index names
# If omitted, an integer index will be used for each

pd.DataFrame(np.random.rand(3, 2), 
                columns=['foo', 'bar'], 
                index=['a', 'b', 'c'])


#endregion

#region From a NumPy structured array
# Pandas DataFrame operates much like a structured array, and can be created directly from one

A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')]); A
pd.DataFrame(A)


#endregion


#endregion


#endregion

#region The Pandas Index Object
# both the Series and DataFrame objects contain an explicit index that lets you reference and modify data

# This Index object is an interesting structure in itself, 
# and it can be thought of either as 
#   an immutable array or 
#   as an ordered set (technically a multiset, as Index objects may contain repeated values)
# Those views have some interesting consequences in the operations available on Index objects

ind = pd.Index([2, 3, 5, 7, 11]); ind


#region Index as immutable array
# Index object in many ways operates like an array

# e.g. can use standard Python indexing notation to retrieve values or slices
ind[1]; ind[::2]


# Index objects also have many of the attributes familiar from NumPy arrays
print(ind.size, ind.shape, ind.ndim, ind.dtype)


# One difference between Index objects and NumPy arrays is that 
#   indices are immutable—that is, they cannot be modified via the normal means
# This immutability makes it safer to share indices between multiple DataFrames and arrays, 
#   without the potential for side effects from inadvertent index modification
ind[1] = 0


#endregion

#region Index as ordered set
# Pandas objects are designed to facilitate operations such as joins across datasets,
#   which depend on many aspects of set arithmetic

# Index object follows many of the conventions used by Python’s built-in set data structure, 
#   so that unions, intersections, differences, and other combinations can be computed in a familiar way
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])

indA & indB # intersection
indA | indB # union
indA ^ indB # symmetric difference

# These operations may also be accessed via object methods—for example, 
indA.intersection(indB)


#endreigon


#endregion


#endregion


#endregion

#region Data Indexing and Selection

#region Data Selection in Series
# Series object acts 
#   in many ways like a onedimensional NumPy array, and 
#   in many ways like a standard Python dictionary
# If we keep these two overlapping analogies in mind, 
#   it will help us to understand the patterns of data indexing and selection in these arrays

#region Series as dictionary
# Like a dictionary, the Series object provides a mapping from a collection of keys to a collection of values

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                    index=['a', 'b', 'c', 'd'])
data; data['b']

# can also use dictionary-like Python expressions and methods to examine the keys/indices and values
'a' in data
data.keys()
list(data.items())

# Series objects can even be modified with a dictionary-like syntax
# Just as you can extend a dictionary by assigning to a new key, 
#   you can extend a Series by assigning to a new index value
data['e'] = 1.25


#endregion

#region Series as one-dimensional array
# Series builds on this dictionary-like interface and 
#   provides array-style item selection via the same basic mechanisms as NumPy arrays—that is, 
#       slices, 
#       masking, and
#       fancy indexing

# slicing by explicit index
data['a':'c']

# slicing by implicit integer index
data[0:2]

# masking
data[(data > 0.3) & (data < 0.8)]

# fancy indexing
data[['a', 'e']]

# Among these, slicing may be the source of the most confusion. 
#   when you are slicing with an explicit index (i.e., data['a':'c']), the final index is included in the slice,
#   while when you’re slicing with an implicit index (i.e., data[0:2]), the final index is excluded from the slice


#endregion

#region Indexers: loc, iloc, and ix
# slicing and indexing conventions can be a source of confusion

# if your Series has an explicit integer index, 
#   an indexing operation such as data[1] will use the explicit indices, while
#   a slicing operation like data[1:3] will use the implicit Python-style index

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5]); data

# explicit index when indexing
data[1]

# implicit index when slicing
data[1:3]

# Because of this potential confusion in the case of integer indexes, 
#   Pandas provides some special indexer attributes that explicitly expose certain indexing schemes
# These are not functional methods, 
#   but attributes that expose a particular slicing interface to the data in the Series

# loc attribute allows indexing and slicing that always references the explicit index
data.loc[1]
data.loc[1:3]

# iloc attribute allows indexing and slicing that always references the implicit Python-style index
data.iloc[1]
data.iloc[1:3]


# One guiding principle of Python code is that “explicit is better than implicit.”
# explicit nature of loc and iloc make them very useful in maintaining clean and readable code; 
# especially in the case of integer indexes, 
#   recommend using these both to make code easier to read and understand, 
#   and to prevent subtle bugs due to the mixed indexing/slicing convention


#endregion


#endregion

#region Data Selection in DataFrame
# DataFrame acts 
#   in many ways like a two-dimensional or structured array, and 
#   in other ways like a dictionary of Series structures sharing the same index
# These analogies can be helpful to keep in mind as we explore data selection within this structure

#region DataFrame as a dictionary
area = pd.Series({'California': 423967, 'Texas': 695662,
                    'New York': 141297, 'Florida': 170312,
                    'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                    'New York': 19651127, 'Florida': 19552860,
                    'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop}); data

# individual Series that make up the columns of the DataFrame can be accessed via dictionary-style indexing of the column name
data['area']

# Equivalently, we can use attribute-style access with column names that are strings
data.area

# attribute-style column access actually accesses the exact same object as the dictionary-style access:
data.area is data['area']

# Though this is a useful shorthand, keep in mind that it does not work for all cases!
#   if the column names are not strings, or 
#   if the column names conflict with methods of the DataFrame, 
# this attribute-style access is not possible

# e.g. DataFrame has a pop() method, so data.pop will point to this rather than the "pop" column
data.pop is data['pop']

# avoid the temptation to try column assignment via attribute
#   (i.e., 
#       use data['pop'] = z 
#       rather than data.pop = z)

# dictionary-style syntax can also be used to modify the object
data['density'] = data['pop'] / data['area']


#endregion

#region DataFrame as two-dimensional array
# can examine the raw underlying data array using the values attribute
data.values

# With this picture in mind, we can do many familiar array-like observations on the DataFrame itself

# e.g. can transpose the full DataFrame to swap rows and columns
data.T

# When it comes to indexing of DataFrame objects, however, it is clear that the
# dictionary-style indexing of columns precludes our ability to simply treat it as a NumPy array

# passing a single index to an array accesses a row
data.values[0]
# passing a single “index” to a DataFrame accesses a column
data['area']

# Thus for array-style indexing, we need another convention
# Here Pandas again uses the loc, iloc indexers mentioned earlier
# Using the iloc indexer, 
#   we can index the underlying array as if it is a simple NumPy array (using the implicit Python-style index), 
#   but the DataFrame index and column labels are maintained in the result
data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']

# Any of the familiar NumPy-style data access patterns can be used within these indexers
# e.g. in the loc indexer we can combine masking and fancy indexing 
data.loc[data.density > 100, ['pop', 'density']]

# Any of these indexing conventions may also be used to set or modify values; 
# this is done in the standard way that you might be accustomed to from working with NumPy
data.iloc[0, 2] = 90


#enderegion


#endregion

#region Additional indexing conventions
# indexing refers to columns, 
# slicing refers to rows
data['Florida':'Illinois']

# Such slices can also refer to rows by number rather than by index
data[1:3]

# direct masking operations are also interpreted row-wise rather than column-wise
data[data.density > 100]


# These two conventions are syntactically similar to those on a NumPy array, 
# and while these may not precisely fit the mold of the Pandas conventions, 
# they are nevertheless quite useful in practice


#endregion


#endregion


#endregion

#region Operating on Data in Pandas
# One of the essential pieces of NumPy is the ability to perform quick element-wise operations, both
#   with basic arithmetic (addition, subtraction, multiplication, etc.) and
#   with more sophisticated operations (trigonometric functions, exponential and logarithmic functions, etc.)
# Pandas inherits much of this functionality from NumPy, 
#   and the ufuncs are key to this

# Pandas includes a couple useful twists, however
#   for unary operations like negation and trigonometric functions, 
#       these ufuncs will preserve index and column labels in the output, and 
#   for binary operations such as addition and multiplication, 
#       Pandas will automatically align indices when passing the objects to the ufunc

# This means that
#   keeping the context of data and 
#   combining data from different sources
# —both potentially error-prone tasks with raw NumPy arrays—
# become essentially foolproof ones with Pandas


#region Ufuncs: Index Preservation
# Because Pandas is designed to work with NumPy, 
#   any NumPy ufunc will work on Pandas Series and DataFrame objects

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4)); ser
df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D']); df


# If we apply a NumPy ufunc on either of these objects, 
#   the result will be another Pandas object with the indices preserved
np.exp(ser)
np.sin(df * np.pi / 4)


#endregion

#region UFuncs: Index Alignment
# For binary operations on two Series or DataFrame objects, 
#   Pandas will align indices in the process of performing the operation
# This is very convenient when you are working with incomplete data

#region Index alignment in Series
# suppose we are combining two different data sources, 
#   and find only the top three US states by area and the top three US states by population
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                    'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

# when we divide these to compute the population density
population / area

# resulting array contains the union of indices of the two input arrays, 
#   which we could determine using standard Python set arithmetic on these indices
area.index | population.index

# Any item for which one or the other does not have an entry is marked with NaN, or “Not a Number,” 
#   which is how Pandas marks missing data 
# This index matching is implemented this way for any of Python’s built-in arithmetic expressions; 
#   any missing values are filled in with NaN by default
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B

# If using NaN values is not the desired behavior, 
#   we can modify the fill value using appropriate object methods in place of the operators

# e.g. calling A.add(B)
#   is equivalent to calling A + B, 
#   but allows optional explicit specification of the fill value for any elements in A or B that might be missing
A.add(B, fill_value=0)


#endregion

#region Index alignment in DataFrame
# similar type of alignment takes place for both columns and indices 
#   when you are performing operations on DataFrames

A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB')); A
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list('BAC')); B
A + B


# Notice that 
#   indices are aligned correctly irrespective of their order in the two objects, and 
#   indices in the result are sorted

# As was the case with Series, 
#   we can use the associated object’s arithmetic method and 
#   pass any desired fill_value to be used in place of missing entries

# fill with the mean of all values in A 
#   (which we compute by first stacking the rows of A)
fill = A.stack().mean()
A.add(B, fill_value=fill)


#endregion


#endregion

#region Ufuncs: Operations Between DataFrame and Series
# When you are performing operations between a DataFrame and a Series, 
#   the index and column alignment is similarly maintained
# Operations between a DataFrame and a Series 
#   are similar to operations 
#   between a two-dimensional and one-dimensional NumPy array

# Consider one common operation, 
#   where we find the difference of a two-dimensional array and one of its rows
# According to NumPy’s broadcasting rules
#   subtraction between a two-dimensional array and one of its rows is applied row-wise
A = rng.randint(10, size=(3, 4)); A
A - A[0]


# In Pandas, the convention similarly operates row-wise by default
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]


# If you would instead like to operate column-wise, 
#   you can use the object methods mentioned earlier, 
#   while specifying the axis keyword
df.subtract(df['R'], axis=0)


# these DataFrame/Series operations, like the operations discussed before,
#   will automatically align indices between the two elements
halfrow = df.iloc[0, ::2]; halfrow
df - halfrow


# This preservation and alignment of indices and columns means that 
#   operations on data in Pandas will always maintain the data context, 
#   which prevents the types of silly errors that might come up 
#   when you are working with heterogeneous and/or misaligned data in raw NumPy arrays


#endregion


#endregion

#region Handling Missing Data

#region Trade-Ofs in Missing Data Conventions
# number of schemes have been developed to indicate the presence of missing data in a table or DataFrame
# Generally, they revolve around one of two strategies: 
#   using a mask that globally indicates missing values, or 
#   choosing a sentinel value that indicates a missing entry

# In the masking approach, 
#   the mask might be an entirely separate Boolean array, or 
#   it may involve appropriation of one bit in the data representation to locally indicate the null status of a value

# In the sentinel approach, 
#   the sentinel value could be some data-specific convention,
#       such as indicating a missing integer value with –9999 or some rare bit pattern, or 
#   it could be a more global convention, 
#       such as indicating a missing floating-point value with NaN (Not a Number), 
#           a special value which is part of the IEEE floating-point specification

# None of these approaches is without trade-offs: 
#   use of a separate mask array 
#       requires allocation of an additional Boolean array, 
#       which adds overhead in both storage and computation. 
#   A sentinel value 
#       reduces the range of valid values that can be represented, and 
#       may require extra (often non-optimized) logic in CPU and GPU arithmetic. 
#   Common special values like NaN are not available for all data types

# As in most cases where no universally optimal choice exists, 
#   different languages and systems use different conventions


#endregion

#region Missing Data in Pandas
# way in which Pandas handles missing values 
#   is constrained by its reliance on the NumPy package, 
#       which does not have a built-in notion of NA values for nonfloating-point data types

# Pandas chose to use sentinels for missing data, 
# and further chose to use two already-existing Python null values: 
#   the special floatingpoint NaN value, and 
#   the Python None object
# This choice has some side effects
# but in practice ends up being a good compromise in most cases of interest 


#region None: Pythonic missing data
# first sentinel value used by Pandas is None, 
#   a Python singleton object 
#   that is often used for missing data in Python code

# Because None is a Python object, 
#   it cannot be used in any arbitrary NumPy/Pandas array, 
#   but only in arrays with data type 'object' (i.e., arrays of Python objects)
# dtype=object means that 
#   the best common type representation NumPy could infer for the contents of the array is that they are Python objects

vals1 = np.array([1, None, 3, 4]); vals1

# While this kind of object array is useful for some purposes, 
#   any operations on the data will be done at the Python level, 
#   with much more overhead than the typically fast operations seen for arrays with native types

for dtype in ['object', 'int']:
    print("dtype =", dtype)
    timeit.timeit("np.arange(1E6, dtype=dtype).sum()", globals=globals(), number=100)
    print()


# use of Python objects in an array also means that 
# if you perform aggregations
#   like sum() or min() 
# across an array with a None value, 
# you will generally get an error
#   This reflects the fact that addition between an integer and None is undefined

vals1.sum()


#endregion

#region NaN: Missing numerical data
# other missing data representation, NaN (acronym for Not a Number), is different;
#   it is a special floating-point value 
#   recognized by all systems that use the standard IEEE floating-point representation

vals2 = np.array([1, np.nan, 3, 4]); vals2.dtype
# NumPy chose a native floating-point type for this array: 
# this means that 
#   unlike the object array from before, 
#   this array supports fast operations pushed into compiled code 

# should be aware that NaN is a bit like a data virus—
#   it infects any other object it touches
# Regardless of the operation, the result of arithmetic with NaN will be another NaN

1 + np.nan
0 * np.nan

# this means that aggregates over the values are well defined 
#   (i.e., they don’t result in an error) 
# but not always useful
vals2.sum(), vals2.min(), vals2.max()


# NumPy does provide some special aggregations that will ignore these missing values
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)


# Keep in mind that 
# NaN is specifically a floating-point value; 
#   there is no equivalent NaN value for integers, strings, or other types


#endregion

#region NaN and None in Pandas
# NaN and None both have their place, and 
# Pandas is built to handle the two of them nearly interchangeably, 
#   converting between them where appropriate

pd.Series([1, np.nan, 2, None])

# For types that don’t have an available sentinel value, 
#   Pandas automatically type-casts when NA values are present

# e.g. if we set a value in an integer array to np.nan, 
# it will automatically be upcast to a floating-point type to accommodate the NA

x = pd.Series(range(2), dtype=int); x
x[0] = None; x  # auto upcast int to float


# in Pandas, string data is always stored with an object dtype


#endregion


#endregion

#region Operating on Null Values
# Pandas treats None and NaN as essentially interchangeable for indicating missing or null values

# To facilitate this convention, 
# there are several useful methods for 
#   detecting, 
#   removing, and 
#   replacing 
# null values in Pandas data structures.

# They are
# isnull()
#   Generate a Boolean mask indicating missing values
# notnull()
#   Opposite of isnull()
# dropna()
#   Return a filtered version of the data
# fillna()
#   Return a copy of the data with missing values filled or imputed


#region Detecting null values
# Pandas data structures have two useful methods for detecting null data: 
#   isnull() and
#   notnull(). 
# Either one will return a Boolean mask over the data

data = pd.Series([1, np.nan, 'Hello', None])
data.isnull()
data[data.notnull()]


#endregion

#region Dropping null values
# there are the convenience methods, 
#   dropna() (which removes NA values) and 
#   fillna() (which fills in NA values)


# For a Series, the result is straightforward
data.dropna()


# For a DataFrame, there are more options
df = pd.DataFrame([[1, np.nan, 2],
                    [2, 3, 5],
                    [np.nan, 4, 6]])
df

# cannot drop single values from a DataFrame; 
# we can only drop 
#   full rows or 
#   full columns
# Depending on the application, you might want one or the other, 
#   so dropna() gives a number of options for a DataFrame

# By default, dropna() will drop all rows in which any null value is present
df.dropna()

# Alternatively, you can drop NA values along a different axis; 
#   axis=1 drops all columns containing a null value
df.dropna(axis='columns')

# But this drops some good data as well; 
# you might rather be interested in dropping rows or columns 
#   with all NA values, or 
#   a majority of NA values
# can be specified through the 
#   how or 
#   thresh parameters, 
# which allow fine control of the number of nulls to allow through

# default is how='any', 
#   such that any row or column (depending on the axis keyword) 
#   containing a null value will be dropped

# can also specify how='all', 
#   which will only drop rows/columns that are all null values
df[3] = np.nan; df
df.dropna(axis='columns', how='all')

# For finer-grained control, 
# the thresh parameter 
#   lets you specify a minimum number of non-null values for the row/column to be kept
df.dropna(axis='rows', thresh=3)


#endregion

#region Filling null values
# Sometimes rather than dropping NA values, you’d rather replace them with a valid value
# This value 
#   might be a single number like zero, or 
#   it might be some sort of imputation or interpolation from the good values

# could do this in-place using the isnull() method as a mask, 
# but because it is such a common operation 
# Pandas provides the fillna() method, 
#   which returns a copy of the array with the null values replaced

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde')); data

# can fill NA entries with a single value
data.fillna(0)

# can specify a forward-fill to propagate the previous value forward
data.fillna(method='ffill')
# can specify a back-fill to propagate the next values backward
data.fillna(method='bfill')


# For DataFrames, the options are similar, but we can also specify an axis along which the fills take place
df
df.fillna(method='ffill', axis=1)


# if a previous value is not available during a forward fill, the NA value remains


#endregion


#endregion


#endregion

#region Hierarchical Indexing
# Often it is useful to store higher-dimensional data—that is, 
#   data indexed by more than one or two keys

# Pandas does provide Panel and Panel4D objects
#   that natively handle three-dimensional and four-dimensional data

# far more common pattern in practice is to make use of 
#   hierarchical indexing (also known as multi-indexing) 
#       to incorporate multiple index levels within a single index
# In this way, higher-dimensional data can be compactly represented within the familiar 
#   one-dimensional Series and 
#   two-dimensional DataFrame objects


#region A Multiply Indexed Series

#region The bad way
# Suppose you would like to track data about states from two different years.
# might be tempted to simply use Python tuples as keys

index = [('California', 2000), ('California', 2010),
            ('New York', 2000), ('New York', 2010),
            ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
                18976457, 19378102,
                20851820, 25145561]
pop = pd.Series(populations, index=index); pop


# With this indexing scheme, 
# you can straightforwardly index or slice the series based on this multiple index
pop[('California', 2000):('Texas', 2000)]


# But the convenience ends there

# e.g. to select all values from 2010, 
#   you’ll need to do some messy (and potentially slow) munging to make it happen
pop[[i for i in pop.index if i[1] == 2010]]

# This produces the desired result, 
# but is not as clean (or as efficient for large datasets)
#   as the slicing syntax we’ve grown to love in Pandas


#endregion

#region The better way: Pandas MultiIndex
# tuple-based indexing is essentially a rudimentary multi-index, and 
# the Pandas MultiIndex type gives us the type of operations we wish to have

# can create a multi-index from the tuples as follows
#   MultiIndex contains 
#       multiple levels of indexing as well as 
#       multiple codes for each data point which encode these levels
index = pd.MultiIndex.from_tuples(index); index.levels; index.codes

# If we reindex our series with this MultiIndex, 
#   we see the hierarchical representation of the data
# in this multi-index representation, 
#   any blank entry indicates
#    the same value as the line above it

pop = pop.reindex(index); pop
# first two columns of the Series representation show the multiple index values, 
# while the third column shows the data


# to access all data for which the second index is 2010, 
# we can simply use the Pandas slicing notation
pop[:, 2010]


# This syntax is 
#   much more convenient 
#   (and the operation is much more efficient!) 
# than the homespun tuple-based multi-indexing solution that we started with


#endregion

#region MultiIndex as extra dimension
# could easily have stored the same data using a simple DataFrame with index and column labels

# In fact, Pandas is built with this equivalence in mind

# unstack() method 
#   will quickly convert a multiplyindexed Series 
#   into a conventionally indexed DataFrame
pop_df = pop.unstack(); pop_df

# stack() method 
#   provides the opposite operation
pop_df.stack()


# Each extra level in a multi-index represents an extra dimension of data; 
# taking advantage of this property gives us much more flexibility in the types of data we can represent


# might want to add another column of demographic data for each state at each year
#   (say, population under 18); 
# with a MultiIndex this is as easy as adding another column to the DataFrame
pop_df = pd.DataFrame({'total': pop,
                        'under18': [9267089, 9284094,
                                    4687374, 4318033,
                                    5906301, 6879014]})
pop_df


# In addition, 
#   all the ufuncs and other functionality
#   work with hierarchical indices as well
# This allows us to easily and quickly manipulate and explore even high-dimensional data

# compute the fraction of people under 18 by year
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()


#endregion


#endregion

#region Methods of MultiIndex Creation
# most straightforward way to construct a multiply indexed Series or DataFrame is to
#   simply pass a list of two or more index arrays to the constructor
# work of creating the MultiIndex is done in the background

df = pd.DataFrame(np.random.rand(4, 2),
                    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                    columns=['data1', 'data2'])
df


# if you pass a dictionary with appropriate tuples as keys, 
# Pandas will automatically recognize this and use a MultiIndex by default
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)


#region Explicit MultiIndex constructors
# For more flexibility in how the index is constructed, 
# you can instead use the class method constructors available in the pd.MultiIndex

# e.g. construct the MultiIndex from a simple list of arrays, 
#   giving the index values within each level
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])

# construct it from a list of tuples, giving the multiple index values of each point
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])

# construct it from a Cartesian product of single indices
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])

# can construct the MultiIndex directly using its internal encoding by
#   passing levels (a list of lists containing available index values for each level) and
#   codes (a list of lists that reference these labels)
pd.MultiIndex(levels=[['a', 'b'], [1, 2]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])


# can pass any of these objects 
#   as the index argument when creating a Series or DataFrame, or 
#   to the reindex method of an existing Series or DataFrame


#endregion

#region MultiIndex level names
# Sometimes it is convenient to name the levels of the MultiIndex

# can accomplish this by 
#   passing the names argument to any of the above MultiIndex constructors, or
#   by setting the names attribute of the index after the fact
pop.index.names = ['state', 'year']; pop


# With more involved datasets, 
# this can be a useful way to keep track of the meaning of various index values


#endregion

#region MultiIndex for columns
# In a DataFrame, 
#   the rows and columns are completely symmetric, 
#   and just as the rows can have multiple levels of indices, 
#   the columns can have multiple levels as well

# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                        names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                        names=['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns); health_data


# we see where the multi-indexing for both rows and columns can come in very handy
# This is fundamentally four-dimensional data, where the dimensions are
#   the subject, 
#   the measurement type, 
#   the year, and 
#   the visit number

# e.g. index the top-level column by the person’s name and 
#   get a full Data Frame containing just that person’s information
health_data['Guido']


# For complicated records 
#   containing multiple labeled measurements 
#   across multiple times 
#   for many subjects (people, countries, cities, etc.), 
# use of hierarchical rows and columns can be extremely convenient!


#endregion


#endregion

#region Indexing and Slicing a MultiIndex
# Indexing and slicing on a MultiIndex is designed to be intuitive, 
#   and it helps if you think about the indices as added dimensions


#region Multiply indexed Series
pop

# can access single elements by indexing with multiple terms
pop['California', 2000]

# MultiIndex also supports partial indexing, 
#   or indexing just one of the levels in the index
# result is another Series, 
#   with the lower-level indices maintained
pop['California']

# Partial slicing is available as well, 
#   as long as the MultiIndex is sorted
pop.loc['California':'New York']

# With sorted indices, 
#   we can perform partial indexing on lower levels 
#   by passing an empty slice in the first index
pop[:, 2000]

# Other types of indexing and selection work as well
# e.g. selection based on Boolean masks
pop[pop > 22E6]

# Selection based on fancy indexing also works
pop[['California', 'Texas']]


#endregion

#region Multiply indexed DataFrames
# multiply indexed DataFrame behaves in a similar manner
health_data

# Remember that columns are primary in a DataFrame,     
#   and the syntax used for multiply indexed Series applies to the column

# e.g.  recover Guido’s heart rate data with a simple operation
health_data['Guido', 'HR']

# with the single-index case, we can use the loc, iloc indexers
health_data.iloc[:2, :2]


# These indexers provide an array-like view of the underlying two-dimensional data,
#   but each individual index in loc or iloc can be passed a tuple of multiple indices
health_data.loc[:, ('Bob', 'HR')]

# Working with slices within these index tuples is not especially convenient; 
#   trying to create a slice within a tuple will lead to a syntax error
health_data.loc[(:, 1), (:, 'HR')]

# could get around this 
#   by building the desired slice explicitly using Python’s builtin slice() function, 
# but a better way in this context is 
#   to use an IndexSlice object,
#       which Pandas provides for precisely this situation
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]


#endregion


#endregion

#region Rearranging Multi-Indices
# There are a number of operations that will preserve all the information in the dataset, 
# but rearrange it for the purposes of various computations
#   e.g. stack() and unstack() methods


#region Sorted and unsorted indices
# Many of the MultiIndex slicing operations will fail if the index is not sorted

# indices are not lexographically sorted
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data

# If we try to take a partial slice of this index, it will result in an error
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)


# Pandas provides a number of convenience routines to perform this type (i.e., lexographical) order of sorting
# e.g. sort_index() and sortlevel() methods of the DataFrame

# simplest, sort_index()
data = data.sort_index(); data

# With the index sorted in this way, partial slicing will work as expected
data['a':'b']


#endregion

#region Stacking and unstacking indices
# it is possible to convert a dataset from a stacked multi-index
# to a simple two-dimensional representation, optionally specifying the level to use

pop.unstack(level=0)
pop.unstack(level=1)

# opposite of unstack() is stack(), 
#   which here can be used to recover the original series
pop.unstack().stack()


#endregion

#region Index setting and resetting
# Another way to rearrange hierarchical data is to turn the index labels into columns
# can be accomplished with the reset_index method
# For clarity, we can optionally specify the name of the data for the column representation
pop_flat = pop.reset_index(name='population'); pop_flat


# Often when you are working with data in the real world, 
# the raw input data looks like this and 
# it’s useful to build a MultiIndex from the column values

# can be done with the set_index method of the DataFrame, 
#   which returns a multiply indexed Data Frame
pop_flat.set_index(['state', 'year'])


#endregion


#endregion

#region Data Aggregations on Multi-Indices
# For hierarchically indexed data, 
#   aggregation methods can be passed a level parameter 
#   that controls which subset of the data the aggregate is computed on

health_data

# to average out the measurements in the two visits each year. 
# can do this by naming the index level we’d like to explore, 
#   in this case the year
data_mean = health_data.mean(level='year'); data_mean

# making use of the axis keyword, 
#   we can take the mean among levels on the columns as well
data_mean.mean(axis=1, level='type')

# This syntax is actually a shortcut to the GroupBy functionality


#endregion


#endregion

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
