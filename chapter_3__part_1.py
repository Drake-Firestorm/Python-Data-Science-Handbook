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
