# CHAPTER 2: Introduction to NumPy
# ================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import timeit



#region Understanding Data Types in Python

#region Creating Arrays from Python Lists
# integer array:
np.array([1, 4, 2, 5, 3])

# If types do not match, NumPy will upcast if possible
np.array([3.14, 4, 2, 3]) 

# to explicitly set the data type of the resulting array
np.array([1, 2, 3, 4], dtype='float32')

# nested lists result in multidimensional arrays
#  inner lists are treated as rows of the resulting two-dimensional array
np.array([range(i, i + 3) for i in [2, 4, 6]])

#endregion

#region Creating Arrays from Scratch
# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)

# Create a 3x5 floating-point array filled with 1s
np.ones((3, 5), dtype=float)

# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)

# Create an array filled with a linear sequence
#   Starting at 0, ending at 20, stepping by 2
#       (this is similar to the built-in range() function)
np.arange(0, 20, 2)

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))

# Create a 3x3 identity matrix
np.eye(3)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that
# memory location
np.empty(3)


#endregion


#endregion

#region The Basics of NumPy Arrays

#region NumPy Array Attributes
np.random.seed(0) # seed for reproducibility
x1 = np.random.randint(10, size=6) # One-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # Three-dimensional array

print("x3 ndim: ", x3.ndim)     # number of dimensions
print("x3 shape:", x3.shape)    # size of each dimension
print("x3 size: ", x3.size)     # total size of the array

print("dtype:", x3.dtype)       # data type of the array
print("itemsize:", x3.itemsize, "bytes")    # size (in bytes) of each array element
print("nbytes:", x3.nbytes, "bytes")        # total size (in bytes) of the array
# In general, we expect that nbytes = itemsize * size


#endregion

#region Array Indexing: Accessing Single Elements
# access the ith value (counting from zero)
#   by specifying the desired index in square brackets,
#       just as with Python lists
x1[0]
x1[4]

# To index from the end of the array, 
#   use negative indices
x1[-1]
x1[-4]

# In a multidimensional array, 
#   access items using a comma-separated tuple of indices
x2[0, 0]
x2[2, 0]
x2[2, -1]

# can also modify values using any of the above index notation
x2[0, 0] = 12

# NumPy arrays have a fixed type.
# This means, for example, that 
#   if you attempt to insert a floating-point value to an integer array, 
#   the value will be silently truncated
x1[0] = 3.14159 # this will be truncated!


#endregion

#region Array Slicing: Accessing Subarrays
# to access a slice of an array x, use:
#   default to the values
#       start=0,
#       stop=size of dimension,
#       step=1
x[start:stop:step]

#region One-dimensional subarrays
x = np.arange(10)

x[:5] # first five elements
x[5:] # elements after index 5
x[4:7] # middle subarray
x[::2] # every other element
x[1::2] # every other element, starting at index 1


# when the step value is negative.
#   the defaults for start and stop are swapped.
#       This becomes a convenient way to reverse an array
x[::-1] # all elements, reversed
x[5::-2] # reversed every other from index 5


#endregion

#region Multidimensional subarrays
# Multidimensional slices work in the same way,
#   with multiple slices separated by commas
x2[:2, :3] # two rows, three columns
x2[:3, ::2] # all rows, every other column

# subarray dimensions can even be reversed together
x2[::-1, ::-1]


#endregion

#region Accessing array rows and columns
# can do this by combining indexing and slicing,
#   using an empty slice marked by a single colon (:)
print(x2[:, 0]) # first column of x2
print(x2[0, :]) # first row of x2
# In the case of row access, the empty slice can be omitted for a more compact syntax
print(x2[0]) # equivalent to x2[0, :]


#endregion

#region Subarrays as no-copy views
# array slices return views rather than copies of the array data.
# This is one area in which NumPy array slicing differs from Python list slicing:
#   in lists, slices will be copies
x2_sub = x2[:2, :2]; print(x2_sub)
x2_sub[0, 0] = 99; print(x2_sub); print(x2)

# This default behavior is actually quite useful: 
#   it means that when we work with large datasets, 
#   we can access and process pieces of these datasets 
#   without the need to copy the underlying data buffer


#endregion

#region Creating copies of arrays
# easily done with the copy() method
x2_sub_copy = x2[:2, :2].copy(); print(x2_sub_copy)
# original array is not touched
x2_sub_copy[0, 0] = 42; print(x2_sub_copy); print(x2)


#endregion


#endregion

#region Reshaping of Arrays
# most flexible way of doing this is with the reshape() method
#   size of the initial array must match the size of the reshaped array
grid = np.arange(1, 10).reshape((3, 3))


# conversion of a one-dimensional array into a two-dimensional row or column matrix
#   can do this with the
#       reshape method, or
#       more easily by making use of the newaxis keyword within a slice operation
x = np.array([1, 2, 3])
x.reshape((1, 3))   # row vector via reshape
x[np.newaxis, :]    # row vector via newaxis
x.reshape((3, 1))   # column vector via reshape
x[:, np.newaxis]    # column vector via newaxis


#endregion

#region Array Concatenation and Splitting

#region Concatenation of arrays
# primarily accomplished through the routines
#   np.concatenate, 
#   np.vstack, and 
#   np.hstack

# np.concatenate takes a tuple or list of arrays as its first argument
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

# can also concatenate more than two arrays at once
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

# can also be used for two-dimensional arrays
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

# concatenate along the first axis
np.concatenate([grid, grid])

# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)


# For working with arrays of mixed dimensions, it can be clearer to use the functions
#   np.vstack (vertical stack) and 
#   np.hstack (horizontal stack) 
x = [1, 2, 3]
grid = np.array([[9, 8, 7],
                [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])

# horizontally stack the arrays
y = [[99],
     [99]]
np.hstack([y, grid])

# np.dstack will stack arrays along the third axis


#endregion

#region Splitting of arrays
# implemented by the functions
#   np.split, 
#   np.hsplit, and 
#   np.vsplit
#       N split points lead to N + 1 subarrays.
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5]); print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
upper, lower = np.vsplit(grid, [2]); print(grid, "\n", upper, "\n", lower)
left, right = np.hsplit(grid, [2]); print(grid, "\n", left, "\n", right)

# np.dsplit will split arrays along the third axis


#endregion


#endregion


#endregion

#region Computation on NumPy Arrays: Universal Functions
# Computation on NumPy arrays can be very fast, or it can be very slow. 
# The key to making it fast is to use vectorized operations, 
#   generally implemented through NumPy’s universal functions (ufuncs)
#       can be used to make repeated calculations on array elements much more efficient

#region The Slowness of Loops
# relative sluggishness of Python generally manifests itself in situations where
#   many small operations are being repeated—for instance, 
#   looping over arrays to operate on each element
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

import timeit
big_array = np.random.randint(1, 100, size=1000000)
timeit.timeit("compute_reciprocals(big_array)", globals=globals(), number=1)
# takes several seconds to compute these million operations and to store the result
# bottleneck here is not the operations themselves,
#   but the type-checking and function dispatches that CPython must do at each cycle of the loop

# If we were working in compiled code instead, 
#   this type specification would be known before the code executes and 
#   the result could be computed much more efficiently


#endregion

#region Introducing UFuncs
# For many types of operations, NumPy provides a convenient interface into just this 
#   kind of statically typed, compiled routine.
# This is known as a vectorized operation.
# You can accomplish this by simply performing an operation on the array,
#   which will then be applied to each element.
# This vectorized approach is designed 
#   to push the loop into the compiled layer that underlies NumPy, 
#   leading to much faster execution

print(compute_reciprocals(values))
print(1.0 / values)

timeit.timeit("print(1.0 / big_array)", globals=globals(), number=100)

# Vectorized operations in NumPy are implemented via ufuncs,
#   whose main purpose is to quickly execute repeated operations on values in NumPy arrays

# can also operate between two arrays
#   for arrays of the same size, 
#       binary operations are performed on an element-by-element basis:
np.arange(5) / np.arange(1, 6)

# they can act on multidimensional arrays as well
x = np.arange(9).reshape((3, 3))
2 ** x

# Computations using vectorization through ufuncs are nearly always more efficient
#   than their counterpart implemented through Python loops, 
#   especially as the arrays grow in size


#endregion

#region Exploring NumPy’s UFuncs
# Ufuncs exist in two flavors:
#   unary ufuncs, which operate on a single input, and 
#   binary ufuncs, which operate on two inputs

#region Array arithmetic
# standard addition, subtraction, multiplication, and division can all be used
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) # floor division

# also a unary ufunc for negation, a ** operator for exponentiation, and a % operator for modulus
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)

# these can be strung together however you wish, and 
#   the standard order of operations is respected
-(0.5*x + 1) ** 2

# All of these arithmetic operations are simply convenient wrappers around specific functions built into NumPy


#endregion

#region Absolute value
# NumPy understands Python’s built-in absolute value function
x = np.array([-2, -1, 0, 1, 2])
abs(x)

# corresponding NumPy ufunc is np.absolute, which is also available under the alias np.abs
np.absolute(x)
np.abs(x)

# ufunc can also handle complex data, in which the absolute value returns the magnitude
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)


#endregion

#region Trigonometric functions
theta = np.linspace(0, np.pi, 3)    # defining an array of angles

# compute some trigonometric functions
print("theta         = ", theta)
print("sin(theta)    = ", np.sin(theta))
print("cos(theta)    = ", np.cos(theta))
print("tan(theta)    = ", np.tan(theta))

# Inverse trigonometric functions
x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x)    = ", np.arcsin(x))
print("arccos(x)    = ", np.arccos(x))
print("arctan(x)    = ", np.arctan(x))



#endregion

#region Exponents and logarithms
# exponentials
x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))


# logarithms, are also available
x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))


# for maintaining precision with very small input
#   When x is very small, these functions give more precise values than if the raw np.log or np.exp were used
x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))


#endregion

#region Specialized ufuncs
# Another excellent source for more specialized and obscure ufuncs is the submodule
#   scipy.special
from scipy import special

# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))

# Error function (integral of Gaussian)
#   its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


#endregion


#endregion

#region Advanced Ufunc Features

#region Specifying output
# For large calculations, it is sometimes useful to be able to specify the array where the result of the calculation will be stored
# Rather than creating a temporary array,
#   you can use this to write computation results directly to the memory location where you’d like them to be.
# For all ufuncs, you can do this using the out argument of the function
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)


# This can even be used with array views
y = np.zeros(10)
np.power(2, x, out=y[::2]); print(y)
# If we had instead written y[::2] = 2 ** x, 
#   this would have resulted in the creation of a temporary array to hold the results of 2 ** x, 
#   followed by a second operation copying those values into the y array

# This doesn’t make much of a difference for such a small computation, 
#   but for very large arrays the memory savings from careful use of the out argument can be significant


#endregion

#region Aggregates
# For binary ufuncs, there are some interesting aggregates that can be computed directly from the object

# reduce method of any ufunc. 
#   repeatedly applies a given operation to the elements of an array 
#   until only a single result remains
x = np.arange(1, 6)
np.add.reduce(x)
np.multiply.reduce(x)

# If we’d like to store all the intermediate results of the computation, we can instead use 
#   accumulate
np.add.accumulate(x)
np.multiply.accumulate(x)

# for these particular cases, there are dedicated NumPy functions to compute the results
#   np.sum, np.prod, np.cumsum, np.cumprod


#endregion

#region Outer products
# any ufunc can compute the output of all pairs of two different inputs using the 
#   outer method
x = np.arange(1, 6)
np.multiply.outer(x, x)


#endregion


#endregion


#endregion

#region Aggregations: Min, Max, and Everything in Between

#region Summing the Values in an Array

# Python itself can do this using the built-in sum function
L = np.random.random(100)
sum(L)

np.sum(L)   # NumPy's version

# because it executes the operation in compiled code, 
#   NumPy’s version of the operation is computed much more quickly
big_array = np.random.rand(1000000)

timeit.timeit("sum(big_array)", globals=globals(), number=1)
timeit.timeit("np.sum(big_array)", globals=globals(), number=1)

# the sum function and the np.sum function are not identical, 
#   which can sometimes lead to confusion! 
# In particular, their optional arguments have different meanings, 
#   and np.sum is aware of multiple array dimensions


#endregion

#region Minimum and Maximum
# Python has built-in min and max functions, 
#   used to find the minimum value and maximum value of any given array
min(big_array), max(big_array)

# NumPy’s corresponding functions have similar syntax, 
#   and again operate much more quickly
np.min(big_array), np.max(big_array)

timeit.timeit("min(big_array)", globals=globals(), number=1)
timeit.timeit("np.min(big_array)", globals=globals(), number=1)


# For min, max, sum, and several other NumPy aggregates, 
#   a shorter syntax is to use
#       methods of the array object itself
print(big_array.min(), big_array.max(), big_array.sum())

# Whenever possible, make sure that you are using the NumPy version of these aggregates when operating on NumPy arrays!


#endregion

#region Multidimensional aggregates

# One common type of aggregation operation is an aggregate along a row or column
M = np.random.random((3, 4))

# By default, each NumPy aggregation function will return the aggregate over the entire array
M.sum()

# Aggregation functions take an additional argument 
#   specifying the axis along which the aggregate is computed
M.min(axis=0)
M.max(axis=1)

# axis keyword specifies the dimension of the array that will be collapsed,
#   rather than the dimension that will be returned


#endregion

#region Other aggregation functions
# most aggregates have a NaN-safe counterpart 
#   that computes the result while ignoring missing values, 
#       which are marked by the special IEEE floating-point NaN value 

import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

print("Mean height: ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height: ", heights.min())
print("Maximum height: ", heights.max())

print("25th percentile: ", np.percentile(heights, 25))
print("Median: ", np.median(heights))
print("75th percentile: ", np.percentile(heights, 75))

import matplotlib.pyplot as plt
import seaborn; seaborn.set()   # set plot style

plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')
plt.show();


#endregion


#endregion

#region Computation on Arrays: Broadcasting

# Another means of vectorizing operations is 
#   to use NumPy’s broadcasting functionality
# Broadcasting is simply a
#   set of rules 
#   for applying binary ufuncs (addition, subtraction, multiplication, etc.) 
#   on arrays of different sizes

#region Introducing Broadcasting
# for arrays of the same size, 
#   binary operations are performed on an element-by-element basis:
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

# Broadcasting allows binary operations to be performed on arrays of different sizes
a + 5
#   can think of this as an operation that stretches or duplicates the value 5 
#       into the array [5, 5, 5], and adds the results
# advantage of NumPy’s broadcasting is that
#   this duplication of values does not actually take place,
#       this extra memory is not actually allocated in the course of the operation
#   but it is a useful mental model as we think about broadcasting

# can similarly extend this to arrays of higher dimension
M = np.ones((3, 3))
M + a


# more complicated cases can involve broadcasting of both arrays
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a + b


#endregion

#region Rules of Broadcasting

# Broadcasting in NumPy follows a strict set of rules to determine the interaction between the two arrays:
# Rule 1:
#   If the two arrays differ in their number of dimensions, the shape of the
#   one with fewer dimensions is padded with ones on its leading (left) side.
# Rule 2:
#   If the shape of the two arrays does not match in any dimension, the array
#   with shape equal to 1 in that dimension is stretched to match the other shape.
# Rule 3:
#   If in any dimension the sizes disagree and neither is equal to 1,
#   an error is raised.

#region Broadcasting example 1
M = np.ones((2, 3))
a = np.arange(3)

M.shape
a.shape
# see by rule 1 that
# array a has fewer dimensions, 
# so we pad it on the left with ones
#   a.shape -> (1, 3)

# By rule 2, 
# we now see that the first dimension disagrees, 
# so we stretch this dimension to match
#   a.shape -> (2, 3)

# The shapes match, 
# and we see that the final shape will be (2, 3)
M + a


#endregion

#region Broadcasting example 2
a = np.arange(3).reshape((3, 1))
b = np.arange(3)

a.shape
b.shape

# Rule 1 says we must pad the shape of b with ones:
#   b.shape -> (1, 3)

# rule 2 tells us that 
# we upgrade each of these ones to match the corresponding size of the other array
#   a.shape -> (3, 3)
#   b.shape -> (3, 3)

# Because the result matches, these shapes are compatible
a + b


#endregion

#region Broadcasting example 3

# two arrays are not compatible 
M = np.ones((3, 2))
a = np.arange(3)

M.shape
a.shape

# rule 1 tells us that we must pad the shape of a with ones
#   a.shape -> (1, 3)

# By rule 2, the first dimension of a is stretched to match that of M
#   a.shape -> (3, 3)

# Now we hit rule 3
# the final shapes do not match, 
# so these two arrays are incompatible
M + a


# you could imagine making a and M compatible by, say, 
#   padding a’s shape with ones on the right rather than the left. 
# But this is not how the broadcasting rules work!
# That sort of flexibility might be useful in some cases, 
#   but it would lead to potential areas of ambiguity

# If right-side padding is what you’d like,
# you can do this explicitly by reshaping the array

a[:, np.newaxis].shape

M + a[:, np.newaxis]

np.logaddexp(M, a[:, np.newaxis])


#endregion


#endregion

#region Broadcasting in Practice

#region Centering an array

# Imagine you have an array of 10 observations, 
#   each of which consists of 3 values
X = np.random.random((10, 3))

# can compute the mean of each feature using the mean aggregate across the first dimension
Xmean = X.mean(0)

# now we can center the X array by subtracting the mean (this is a broadcasting operation)
X_centered = X - Xmean
# To double-check that we’ve done this correctly, 
# we can check that the centered array has near zero mean
X_centered.mean(0)


#endregion

#region Plotting a two-dimensional function

# displaying images based on twodimensional functions

# If we want to define a function z = f(x, y), 
# broadcasting can be used to compute the function across the grid

# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y + x) * np.cos(x)

import matplotlib.pyplot as plt

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
            cmap='viridis')
plt.colorbar()
plt.show()


#endregion


#endregion


#endregion

#region Comparisons, Masks, and Boolean Logic
# Boolean masks
#   used to examine and manipulate values within NumPy arrays

# Masking 
#   comes up when you want to extract, modify, count, or otherwise manipulate values
#   in an array based on some criterion

#region Example: Counting Rainy Days
import pandas as pd
# use Pandas to extract rainfall inches as a NumPy array
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254     # 1/10mm -> inches
inches.shape

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.hist(inches, 40)
plt.show()


#endregion

#region Comparison Operators as ufuncs

# NumPy also implements comparison operators 
#   such as < (less than) and > (greater than) 
# as element-wise ufuncs

# result of these comparison operators is always an array with a Boolean data type
# All six of the standard comparison operations are available
x = np.array([1, 2, 3, 4, 5])

x < 3 # less than
x > 3 # greater than
x <= 3 # less than or equal
x >= 3 # greater than or equal
x != 3 # not equal
x == 3 # equal


# also possible to do an element-by-element comparison of two arrays, 
# and to include compound expressions
(2 * x) == (x ** 2)

# these will work on arrays of any size and shape
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
x < 6


#endregion

#region Working with Boolean Arrays

#region Counting entries
# np.count_nonzero
#   To count the number of True entries in a Boolean array

# how many values less than 6?
np.count_nonzero(x < 6)

# Another way to get at this information is 
#   to use np.sum; 
#       in this case, 
#           False is interpreted as 0, and 
#           True is interpreted as 1
np.sum(x < 6)

# benefit of sum() is that 
#   like with other NumPy aggregation functions, 
#   this summation can be done along rows or columns as well

# how many values less than 6 in each row?
np.sum(x < 6, axis=1)


# If we’re interested in quickly checking whether any or all the values are true, 
# we can use
#   np.any() or
#   np.all()

# are there any values greater than 8?
np.any(x > 8)
# are there any values less than zero?
np.any(x < 0)
# are all values less than 10?
np.all(x < 10)
# are all values equal to 6?
np.all(x == 6)

# np.all() and np.any() can be used along particular axes as well

# are all values in each row less than 8?
np.all(x < 8, axis=1)


#endregion

#region Boolean operators

# Python’s bitwise logic operators, &, |, ^, and ~
# Like with the standard arithmetic operators, 
#   NumPy overloads these as ufuncs 
#   that work element-wise on (usually Boolean) arrays

np.sum((inches > 0.5) & (inches < 1))

# parentheses here are important
#   because of operator precedence rules,
#   with parentheses removed this expression would be evaluated as follows, 
#   which results in an error
inches > (0.5 & inches) < 1

# Using the equivalence of A AND B and NOT (A OR B)
# can compute the same result in a different manner
np.sum(~( (inches <= 0.5) | (inches >= 1) ))


print("Number of days without rain:", np.sum(inches == 0))
print("Number days with rain: ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.1 inches :", np.sum( (inches > 0) & (inches < 0.2 )))


#endregion


#endregion

#region Boolean Arrays as Masks

# more powerful pattern is to use Boolean arrays as masks, 
# to select particular subsets of the data themselves

# masking operation
#   to select these values from the array, 
#   we can simply index on this Boolean array;
x[x < 5]

# What is returned is a one-dimensional array filled with all the values that meet this condition; 
#   in other words, all the values in positions at which the mask array is True
# are then free to operate on these values as we wish

# construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)

print("Median precip on rainy days in 2014 (inches): ", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches): ", np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ", np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):", np.median(inches[~summer & rainy]))


#endregion

#region Using the Keywords and/or Versus the Operators &/|
# and and or gauge the truth or falsehood of entire object, while 
# & and | refer to bits within each object

# When you use and or or, 
#   it’s equivalent to asking Python to treat the object as a single Boolean entity
# In Python, all nonzero integers will evaluate as True
bool(42), bool(0)
bool(42 and 0)
bool(42 or 0)

# When you use & and | on integers, 
#   the expression operates on the bits of the element,
#   applying the and or the or to the individual bits making up the number
bin(42)
bin(59)
bin(42 & 59)
bin(42 | 59)
# corresponding bits of the binary representation are compared in order to yield the result


# When you have an array of Boolean values in NumPy, 
# this can be thought of as a string of bits 
#   where 1 = True and 0 = False, 
# and the result of & and | operates in a similar manner as before
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B

# Using or on these arrays will try to evaluate the truth or falsehood of the entire array object, 
# which is not a well-defined value
A or B

# when doing a Boolean expression on a given array, 
# you should use | or & rather than or or and
x = np.arange(10)
(x > 4) & (x < 8)

# Trying to evaluate the truth or falsehood of the entire array will give the same ValueError we saw previously
(x > 4) and (x < 8)


#endregion


#endregion

#region Fancy Indexing
# Fancy indexing is like the simple indexing we’ve already seen, 
#   but we pass arrays of indices in place of single scalars
# This allows us to very quickly access and modify complicated subsets of an array’s values

#region Exploring Fancy Indexing
rand = np.random.RandomState(42)

x = rand.randint(100, size=10); print(x)

# Suppose we want to access three different elements. 
# We could do it like this:
[x[3], x[7], x[4]]
# Alternatively, we can pass a single list or array of indices to obtain the same result
ind = [3, 7, 4]
x[ind]


# With fancy indexing, 
#   the shape of the result reflects the shape of the index arrays
#   rather than the shape of the array being indexed
ind = np.array([[3, 7],
                [4, 5]])
x[ind]


# Fancy indexing also works in multiple dimensions
X = np.arange(12).reshape((3, 4)); print(X)

# Like with standard indexing, the 
#   first index refers to the row, and 
#   second to the column
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

# pairing of indices in fancy indexing follows all the broadcasting rules 
# if we combine a column vector and a row vector within the indices, we get a two-dimensional result
X[row[:, np.newaxis], col]
row[:, np.newaxis] * col

# with fancy indexing 
#   the return value reflects the broadcasted shape of the indices, 
#   rather than the shape of the array being indexed


#endregion

#region Combined Indexing
# For even more powerful operations, 
#   fancy indexing can be combined with the other indexing schemes we’ve seen
print(X)

# can combine fancy and simple indices
X[2, [2, 0, 1]]

# can also combine fancy indexing with slicing
X[1:, [2, 0, 1]]

# can combine fancy indexing with masking
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]
X[:, mask]  # same as above


#endregion

#region Example: Selecting Random Points
# One common use of fancy indexing is the 
#   selection of subsets of rows from a matrix

mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

import matplotlib.pyplot as plt
import seaborn; seaborn.set() # for plot styling
plt.scatter(X[:, 0], X[:, 1]); plt.show()

indices = np.random.choice(X.shape[0], 20, replace=True); print(indices)

selection = X[indices]  # fancy indexing here
selection.shape

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='none', s=200)
plt.show()

# This sort of strategy is often used 
#   to quickly partition datasets, 
#       as is often needed in train/test splitting for validation of statistical models
#   in sampling approaches to answering statistical questions


#endregion

#region Modifying Values with Fancy Indexing
# fancy indexing can also be used to modify parts of an array

# have an array of indices and we’d like to set the corresponding items in an array to some value
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99; print(x)

# can use any assignment-type operator for this
x[i] -= 10; print(x)

# repeated indices with these operations can cause some potentially unexpected results
x = np.zeros(10)
x[[0, 0]] = [4, 6]; print(x)
# Where did the 4 go? 
# The result of this operation 
#   is to first assign x[0] = 4, 
#   followed by x[0] = 6. 
# The result, of course, is that x[0] contains the value 6

i = [2, 3, 3, 4, 4, 4]
x[i] += 1
# might expect that x[3] would contain the value 2, and x[4] would contain the value 3, 
#   as this is how many times each index is repeated. 
# Why is this not the case?
# Conceptually, this is because x[i] += 1 is meant as a shorthand of x[i] = x[i] + 1.
#   x[i] + 1 is evaluated, 
#       and then the result is assigned to the indices in x. 
#   With this in mind, 
#       it is not the augmentation that happens multiple times, 
#       but the assignment,
#       which leads to the rather nonintuitive results


# if you want the other behavior where the operation is repeated
#   can use the at() method of ufuncs
x = np.zeros(10)
np.add.at(x, i, 1); print(x)

# at() method 
#   does an in-place application of the given operator 
#   at the specified indices (here, i) 
#   with the specified value (here, 1)


# Another method that is similar in spirit is the 
#   reduceat() method of ufuncs
np.add.reduceat(x, i)

#endregion

#region Example: Binning Data
# can use these ideas to efficiently bin data to create a histogram by hand
# example
# imagine we have 1,000 values and would like to quickly find where they fall within an array of bins. 
#   could compute it using ufunc.at like this

np.random.seed(42)
x = np.random.randn(100)

# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)

# add 1 to each of these bins
np.add.at(counts, i, 1)

#  counts now reflect the number of points within each bin—in other words, a histogram

# plot the results
plt.plot(bins, counts, drawstyle = 'steps'); plt.show()


# Matplotlib provides the plt.hist() routine, which does the same in a single line
plt.hist(x, bins, histtype='step'); plt.show()


# To compute the binning, Matplotlib uses the np.histogram function, 
#   which does a very similar computation to what we did before

print("NumPy routines")
timeit.timeit("np.histogram(x, bins)", globals=globals(), number=10000)
print("Custom routines")
timeit.timeit("np.add.at(counts, np.searchsorted(bins, x), 1)", globals=globals(), number=10000)

# Our own one-line algorithm is several times faster than the optimized algorithm in NumPy! 
# How can this be? 
#   If you dig into the np.histogram source code, 
#   you’ll see that it’s quite a bit more involved than the simple search-and-count that we’ve done; 
#       this is because NumPy’s algorithm is more flexible, 
#       and particularly is designed for better performance when the number of data points becomes large:

x = np.random.randn(10**6)

print("NumPy routines")
timeit.timeit("np.histogram(x, bins)", globals=globals(), number=10)
print("Custom routines")
timeit.timeit("np.add.at(counts, np.searchsorted(bins, x), 1)", globals=globals(), number=10)


# What this comparison shows is that algorithmic efficiency is almost never a simple question
# An algorithm efficient for large datasets will not always be the best choice for small datasets, and vice versa


#endregion


#endregion

#region Sorting Arrays
# selection sort
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.array([2, 1, 4, 3, 5])
selection_sort(x)

# selection sort is useful for its simplicity, 
#   but is much too slow to be useful for larger arrays
#   selection sort averages O[N**2]:
#       if you double the number of items in the list, 
#       the execution time will go up by about a factor of four

# Even selection sort, though, is much better than bogosort
def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x

x = np.array([2, 1, 4, 3, 5])
bogosort(x)

# This silly sorting method relies on pure chance:
#   it repeatedly applies a random shuffling of the array until the result happens to be sorted
# With an average scaling of O[N × N!] (that’s N times N factorial), 
#   this should—quite obviously—never be used for any real computation

# Python contains built-in sorting algorithms that are much more efficient than either of the simplistic algorithms just shown


#region Fast Sorting in NumPy: np.sort and np.argsort
# Although Python has built-in sort and sorted functions to work with lists
# NumPy’s np.sort function turns out to be much more efficient and useful
# By default np.sort uses an O[N log N], quick‐sort algorithm, 
#   though mergesort and heapsort are also available
#       For most applications, the default quicksort is more than sufficient

# To return a sorted version of the array without modifying the input, can use np.sort
x = np.array([2, 1, 4, 3, 5])
np.sort(x)

# to sort the array in-place, can instead use the sort method of arrays
x.sort()

# related function is argsort, 
#   which returns the indices of the sorted elements
# These indices can then be used (via fancy indexing) to construct the sorted array if desired
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x); print(i, x)
x[i]



#region Sorting along rows or columns
# useful feature of NumPy’s sorting algorithms 
#   is the ability to sort along specific rows or columns of a multidimensional array using the axis argument
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6)); print(X)

# sort each column of X
np.sort(X, axis=0)

# sort each row of X
np.sort(X, axis=1)

# this treats each row or column as an independent array, 
#   and any relationships between the row or column values will be lost!


#endregion


#endregion

#region Partial Sorts: Partitioning
# to find the K smallest values in the array
# np.partition takes an array and a number K; 
#   the result is a new array 
#   with the smallest K values to the left of the partition, and 
#   the remaining values to the right, 
#   in arbitrary order (Within the two partitions, the elements have arbitrary order)

x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)


# can partition along an arbitrary axis of a multidimensional array
np.partition(X, 2, axis=1)


# np.argpartition 
#   computes indices of the partition


#endregion

#region Example: k-Nearest Neighbors
# use this argsort function along multiple axes to find the nearest neighbors of each point in a set

# create a random set of 10 points on a two-dimensional plane
X = rand.rand(10, 2)
plt.scatter(X[:, 0], X[:, 1], s=100); plt.show()

# compute the distance between each pair of points
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)

# This operation has a lot packed into it, 
#   and it might be a bit confusing if you’re unfamiliar with NumPy’s broadcasting rules
# When you come across code like this, 
#   it can be useful to break it down into its component steps

# for each pair of points, compute differences in their coordinates
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape

# square the coordinate differences
sq_differences = differences ** 2
sq_differences.shape

# sum the coordinate differences to get the squared distance
dist_sq = sq_differences.sum(-1)
dist_sq.shape

# Just to double-check what we are doing, we should see that 
#   the diagonal of this matrix (i.e., the set of distances between each point and itself) is all zero
dist_sq.diagonal()

# With the pairwise square-distances converted, we can now use np.argsort to sort along each row
# leftmost columns will then give the indices of the nearest neighbors
nearest = np.argsort(dist_sq, axis=1); print(nearest)

# Notice that the first column gives the numbers 0 through 9 in order: 
#   this is due to the fact that each point’s closest neighbor is itself, as we would expect

# By using a full sort here, we’ve actually done more work than we need to in this case
# If we’re simply interested in the nearest k neighbors, all we need is to 
#   partition each row so that the smallest k + 1 squared distances come first, 
#   with larger distances filling the remaining positions of the array

# can do this with the np.argpartition function
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

# In order to visualize this network of neighbors, 
#   quickly plot the points 
#   along with lines representing the connections from each point to its two nearest neighbors

plt.scatter(X[:, 0], X[:, 1], s=100)
# draw lines from each point to its two nearest neighbors
K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K + 1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')
plt.show()

# Each point in the plot has lines drawn to its two nearest neighbors
# At first glance, it might seem strange that some of the points have more than two lines coming out of them: 
#   this is due to the fact that 
#       if point A is one of the two nearest neighbors of point B, 
#       this does not necessarily imply that point B is one of the two nearest neighbors of point A

# Although the broadcasting and row-wise sorting of this approach might seem less straightforward than writing a loop, 
#   it turns out to be a very efficient way of operating on this data in Python
# manually looping through the data and sorting each set of neighbors individually 
#   would almost certainly lead to a slower algorithm 
#   than the vectorized version we used
#  beauty of this approach is that it’s written in a way that’s agnostic to the size of the input data


# when doing very large nearest-neighbor searches, 
#   there are treebased and/or approximate algorithms that can scale as O[N log N] or better 
#   rather than the O{N**2} of the brute-force algorithm
#       One example of this is the KD-Tree implemented in Scikit-Learn


#endregion


#endregion

#region Structured Data: NumPy’s Structured Arrays
# use of NumPy’s structured arrays and record arrays, 
#   which provide efficient storage for compound, heterogeneous data
# While the patterns shown here are useful for simple operations,
#   scenarios like this often lend themselves to the use of Pandas DataFrames


# Imagine that we have several categories of data on a number of people (say, name, age, and weight), 
# and we’d like to store these values for use in a Python program. 

# It would be possible to store these in three separate arrays:
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# But this is a bit clumsy. 
#   There’s nothing here that tells us that the three arrays are related; 
# it would be more natural if we could use a single structure to store all of this data

# NumPy can handle this through structured arrays, 
#   which are arrays with compound data types

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                            'formats':('U10', 'i4', 'f8')})
print(data.dtype)
# 'U10' translates to “Unicode string of maximum length 10,” 
# 'i4' translates to “4-byte (i.e., 32 bit) integer,” and 
# 'f8' translates to “8-byte (i.e., 64 bit) float.”

# Now that we’ve created an empty container array, we can fill the array with our lists of values
# data is now arranged together in one convenient block of memory
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# handy thing with structured arrays is that you can now refer to values either by index or by name

# Get all names
data['name']

# Get first row of data
data[0]

# Get the name from the last row
data[-1]['name']


# Using Boolean masking, this even allows you to do some more sophisticated operations 
#   such as filtering on age

# Get names where age is under 30
data[data['age'] < 30]['name']


# if you’d like to do any operations that are any more complicated than these,
#   you should probably consider the Pandas package
# Pandas provides a DataFrame object, 
#   which is a structure built on NumPy arrays 
#   that offers a variety of useful data manipulation functionality 
#       similar to what we’ve shown here, 
#       as well as much, much more


#region Creating Structured Arrays
# Structured array data types can be specified in a number of ways
#   dictionary method
np.dtype({'names':('name', 'age', 'weight'),
            'formats':('U10', 'i4', 'f8')})

# For clarity, numerical types can be specified with Python types or NumPy dtypes instead
np.dtype({'names':('name', 'age', 'weight'),
            'formats':((np.str_, 10), int, np.float32)})

# compound type can also be specified as a list of tuples
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])

# If the names of the types do not matter to you, 
#   can specify the types alone in a comma-separated string
np.dtype('S10, i4, f8')

# shortened string format codes may seem confusing, but they are built on simple principles
#   first (optional) character is < or >, 
#       which means “little endian” or “big endian,” respectively, and 
#       specifies the ordering convention for significant bits
#   next character specifies the type of data: 
#       characters, bytes, ints, floating points, and so on
#   last character or characters represents the size of the object in bytes


#endregion

#region More Advanced Compound Types
# e.g. create a type where each element contains an array or matrix of values
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0]); print(X['mat'][0])

# Why would you use this rather than a simple multidimensional array, or perhaps a Python dictionary?
#   reason is that this NumPy dtype directly maps onto a C structure definition, 
#   so the buffer containing the array content can be accessed directly within an appropriately written C program


#endregion

#region RecordArrays: Structured Arrays with a Twist
# NumPy also provides the np.recarray class, 
#   which is almost identical to the structured arrays just described, 
#   but with one additional feature: 
#       fields can be accessed as attributes rather than as dictionary keys

# e.g.
data['age']
# If we view our data as a record array instead, we can access this with slightly fewer keystrokes
data_rec = data.view(np.recarray)
data_rec.age

# downside is that for record arrays, 
#   there is some extra overhead involved in accessing the fields, 
#   even when using the same syntax
# Whether the more convenient notation is worth the additional overhead will depend on your own application
timeit.timeit('data["age"]', globals=globals(), number=10**5)
timeit.timeit('data_rec["age"]', globals=globals(), number=10**5)
timeit.timeit('data_rec.age', globals=globals(), number=10**5)


#endregion

#region On to Pandas
# Structured arrays
#   are good to know about for certain situations, 
#       especially in case you’re using NumPy arrays to map onto binary data formats in C, Fortran, or another language
# For day-to-day use of structured data, the Pandas package is a much better choice


#endregion


#endregion
