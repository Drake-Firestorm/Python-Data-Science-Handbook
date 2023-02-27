# CHAPTER 5: In Depth: Linear Regression
# ======================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

from pandas.tseries.holiday import USFederalHolidayCalendar

# --------------------------------------------------------------


# linear regression models are a good starting point for regression tasks
# Such models are popular because 
#   they can be fit very quickly, 
#   and are very interpretable
# probably familiar with the simplest form of a linear regression model (i.e., fitting a straight line to data), 
#   but such models can be extended to model more complicated data behavior


#region Simple Linear Regression
# most familiar linear regression, a straight-line fit to data
# straight-line fit is a model of the form 
#   y = ax + b where 
#   a is commonly known as the slope, and 
#   b is commonly known as the intercept


# following data, which is scattered about a line with a slope of 2 and an intercept of –5
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)
plt.show()


# can use Scikit-Learn’s LinearRegression estimator to fit this data and construct the best-fit line
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()


# slope and intercept of the data are contained in the model’s fit parameters, 
#   which in Scikit-Learn are always marked by a trailing underscore

# relevant parameters are coef_ and intercept_
print("Model slope: ", model.coef_); print("Model intercept: ", model.intercept_)

# see that the results are very close to the inputs, as we might hope


# LinearRegression estimator is much more capable than this, however—
# in addition to simple straight-line fits, 
# it can also handle multidimensional linear models of the form:
#   y = a0 + a1x1 + a2x2 + ⋯
#       where there are multiple x values

# Geometrically, this is akin to fitting 
#   a plane to points in three dimensions, or 
#   a hyper-plane to points in higher dimensions


# multidimensional nature of such regressions makes them more difficult to visualize, 
# but we can see one of these fits in action by building some example data, 
#   using NumPy’s matrix multiplication operator

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X,  [1.5, -2., 1.])

model.fit(X, y)
print(model.intercept_); print(model.coef_)

# Here the y data is constructed from three random x values, 
# and the linear regression recovers the coefficients used to construct the data

# In this way, we can use the single LinearRegression estimator to fit lines, planes, or hyperplanes to our data
# It still appears that this approach would be limited to strictly linear relationships between variables, 
#   but it turns out we can relax this as well


#endregion

#region Basis Function Regression
# One trick you can use to adapt linear regression to nonlinear relationships between variables 
#   is to transform the data according to basis functions

# idea is to take our multidimensional linear model:
#   y = a0 + a1x1 + a2x2 + a3x3 + ⋯
# and build the x1, x2, x3, and so on from our single-dimensional input x
# That is, we let
#   xn = fn(x) , where fn is some function that transforms our data
# e.g.
#   if fn(x) = x**n, our model becomes a polynomial regression
# this is still a linear model—
#   the linearity refers to the fact that 
#       the coefficients an never multiply or divide each other
# What we have effectively done is 
#   taken our one-dimensional x values and 
#   projected them into a higher dimension, 
#   so that a linear fit can fit more complicated relationships between x and y


#region Polynomial basis functions
# polynomial projection is useful enough that 
# it is built into Scikit-Learn, 
#   using the PolynomialFeatures transformer

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])

# see here that the transformer has converted our one-dimensional array 
#   into a # three-dimensional array by taking the exponent of each value
# This new, higherdimensional data representation can then be plugged into a linear regression
# cleanest way to accomplish this is to use a pipeline

from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

# With this transform in place, 
# we can use the linear model to fit much more complicated relationships between x and y

# e.g. sine wave with noise
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()


#endregion

#region Gaussian basis functions
# other basis functions are possible
# one useful pattern is 
#   to fit a model 
#   that is not a sum of polynomial bases, 
#   but a sum of Gaussian bases

# These Gaussian basis functions are not built into Scikit-Learn, 
#   but we can write a custom transformer that will create them
# Scikit-Learn transformers are implemented as Python classes; 
#   reading Scikit-Learn’s source is a good way to see how they can be created

from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def __gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    
    def transform(self, X):
        return self.__gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)
    
gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)
plt.show()


# there is nothing magic about polynomial basis functions: 
# if you have some sort of intuition into the generating process of your data 
# that makes you think one basis or another might be appropriate, 
# you can use them as well


#endregion


#endregion

#region Regularization
# introduction of basis functions into our linear regression 
#   makes the model much more flexible, 
#   but it also can very quickly lead to overfitting

# e.g.
# if we choose too many Gaussian basis functions, we end up with results that don’t look so good

model = make_pipeline(GaussianFeatures(30), LinearRegression())
model.fit(x[:, np.newaxis], y)

plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.show()

# With the data projected to the 30-dimensional basis, 
#   the model has far too much flexibility 
#   and goes to extreme values between locations where it is constrained by data
# can see the reason for this if we plot the coefficients of the Gaussian bases with respect to their locations 

def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))

    if title:
        ax[0].set_title(title)
    
    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))

    plt.show()

model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)


# lower panel  shows the amplitude of the basis function at each location
# This is typical overfitting behavior when basis functions overlap: 
#   the coefficients of adjacent basis functions blow up and cancel each other out
# know that such behavior is problematic, 
#   and it would be nice if we could limit such spikes explicitly in the model 
#   by penalizing large values of the model parameters
# Such a penalty is known as regularization, and comes in several forms


#region Ridge regression (L2 regularization)
# most common form of regularization is known as 
#   ridge regression / L2 regularization / Tikhonov regularization
# This proceeds by penalizing the sum of squares (2-norms) of the model coefficients; 
#   in this case, the penalty on the model fit would be
#   P = α∑[n = 1, N] θn ** 2
#   where α is a free parameter that controls the strength of the penalty

# This type of penalized model is built into Scikit-Learn with the Ridge estimator 
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')

# α parameter is essentially a knob controlling the complexity of the resulting model
# In the limit α -> 0, we recover the standard linear regression result; 
# in the limit α -> ∞, all model responses will be suppressed
# One advantage of ridge regression in particular is that 
#   it can be computed very efficiently—
#   at hardly more computational cost than the original linear regression model


#endregion

#region Lasso regularization (L1)
# Another very common type of regularization is known as lasso, 
#   and involves penalizing the sum of absolute values (1-norms) of regression coefficients
#   P = α∑[n = 1, N] | θn |

# Though this is conceptually very similar to ridge regression, the results can differ surprisingly

# e.g.
# due to geometric reasons lasso regression tends to favor sparse models where possible; 
#   that is, it preferentially sets model coefficients to exactly zero

from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')


# With the lasso regression penalty, 
#   the majority of the coefficients are exactly zero,
#   with the functional behavior being modeled by a small subset of the available basis functions

# As with ridge regularization, 
#   the α parameter tunes the strength of the penalty, and should be determined via
#   for e.g., 
#       cross-validation


#endregion


#endregion

#region Example: Predicting Bicycle Traic
# e.g. of
# how the tools of Scikit-Learn can be used in a statistical modeling framework, 
#   in which the parameters of the model are assumed to have interpretable meaning
# this is not a standard approach within machine learning, 
#   but such interpretation is possible for some models

# loading the two datasets, indexing by date
counts = pd.read_csv('Data\Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('Data\BicycleWeather.csv', index_col='DATE', parse_dates=True)
counts.head(); weather.head()

# compute the total daily bicycle traffic, and put this in its own DataFrame
daily = counts.resample('d').sum()
daily = daily[['Fremont Bridge Total']]; daily.head()

# saw previously that the patterns of use generally vary from day to day; 
# let’s account for this in our data by adding binary columns that indicate the day of the week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)
daily.head()

# Similarly, we might expect riders to behave differently on holidays; 
#   let’s add an indicator of this as well
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar(); cal.holidays()[:5]
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True); daily.head()

# We also might suspect that the hours of daylight would affect how many people ride;
#   let’s use the standard astronomical calculation to add this information
def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - datetime.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return  24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot(); plt.show()

# can also add the average temperature and total precipitation to the data
#   also add a flag that indicates whether a day is dry (has zero precipitation)

# temperatures are in 1/10 deg C; convert to C
weather.head()
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

# add a counter that increases from day 1, 
#   and measures how many years have passed. 
# This will let us measure any observed annual increase or decrease in daily crossings
daily['annual'] = (daily.index - daily.index[0]).days / 365.

# Now our data is in order, and we can take a look at it:
daily.head()
#   weather data is not up to date in github. Filter daily to match dates
daily = daily[daily.index <= weather.index[-1]]


# choose the columns to use, and fit a linear regression model to our data
# set fit_intercept = False, because the daily flags essentially operate as their own day-specific intercepts
column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]; X.head()
y = daily['Fremont Bridge Total']; y.head()

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X); daily.head()

# compare the total and predicted bicycle traffic visually
daily[['Fremont Bridge Total', 'predicted']].plot(alpha=0.5); plt.show()

# It is evident that we have missed some key features, especially during the summer time
# Either our features are not complete 
#   (i.e., people decide whether to ride to work based on more than just these) 
# or there are some nonlinear relationships that we have failed to take into account
#   (e.g., perhaps people ride less at both high and low temperatures)
# Nevertheless, our rough approximation is enough to give us some insights,
#   and we can take a look at the coefficients of the linear model 
#       to estimate how much each feature contributes to the daily bicycle count
params = pd.Series(model.coef_, index=X.columns); params

# These numbers are difficult to interpret without some measure of their uncertainty
#  compute these uncertainties quickly using bootstrap resamplings of the data
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_ 
                for i in range(1000)], 0)

# With these errors estimated, let’s again look at the results
print(pd.DataFrame({'effect': params.round(0),
                    'error': err.round(0)}))


# first see that there is a relatively stable trend in the weekly baseline: 
#   there are many more riders on weekdays than on weekends and holidays
# for each additional hour of daylight, 129 ± 9 more people choose to ride
# temperature increase of one degree Celsius encourages 65 ± 4 people to grab their bicycle
# dry day means an average of 546 ± 33 more riders
# each inch of precipitation means 665 ± 62 more people leave their bike at home
# Once all these effects are accounted for, we see a modest increase of 27 ± 18 new daily riders each year


# Our model is almost certainly missing some relevant information
# non‐linear effects 
#   (such as effects of precipitation and cold temperature) 
# and nonlinear trends within each variable 
#   (such as disinclination to ride at very cold and very hot temperatures) 
# cannot be accounted for in this model
# Additionally, we have thrown away some of the finer-grained information 
#   (such as the difference between a rainy morning and a rainy afternoon), 
# and we have ignored correlations between days
#   (such as the possible effect of a rainy Tuesday on Wednesday’s numbers, 
#       or the effect of an unexpected sunny day after a streak of rainy days)


#endregion
