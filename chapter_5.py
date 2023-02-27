# CHAPTER 5: Machine Learning
# ===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from numpy import nan
import datetime
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mode


from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_circles
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_sample_image


# -------------------------------------------------------------

#region What Is Machine Learning?
# Fundamentally, machine learning involves building mathematical models to help understand data. 
# “Learning” enters the fray when we give these models tunable parameters that can be adapted to observed data; 
#   in this way the program can be considered to be “learning” from the data. 
# Once these models have been fit to previously seen data, 
#   they can be used to predict and understand aspects of newly observed data

# Understanding the problem setting in machine learning is essential to using these tools effectively


#region Categories of Machine Learning
# At the most fundamental level, machine learning can be categorized into two main types: 
#   supervised learning and 
#   unsupervised learning

# Supervised learning 
#   involves somehow modeling the relationship between measured features of data and some label associated with the data; 
#   once this model is determined, it can be used to apply labels to new, unknown data
# further subdivided into 
#   classiication tasks and 
#       labels are discrete categories
#   regression tasks
#       labels are continuous quantities

# Unsupervised learning 
#   involves modeling the features of a dataset without reference to any label, 
#   and is often described as “letting the dataset speak for itself.”
# models include tasks such as 
#   clustering and 
#       identify distinct groups of data
#   dimensionality reduction
#       search for more succinct representations of the data

# semi-supervised learning methods, 
#   which fall somewhere between supervised learning and unsupervised learning. 
#   often useful when only incomplete labels are available


#endregion

#region Qualitative Examples of Machine Learning Applications

#region Classiication: Predicting discrete labels
# simple classiication task, 
#   in which you are given a set of labeled points and 
#   want to use these to classify some unlabeled points


# common plot formatting for below
def format_plot(ax, title):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xlabel('feature 1', color='gray')
    ax.set_ylabel('feature 2', color='gray')
    ax.set_title(title, color='gray')


# following code generates the figures from the Classification section
from sklearn.datasets._samples_generator import make_blobs
from sklearn.svm import SVC
# create 50 separable points
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
# fit the support vector classifier model
clf = SVC(kernel='linear')
clf.fit(X, y)
# create some new points to predict
X2, _ = make_blobs(n_samples=80, centers=2, random_state=0, cluster_std=0.80)
X2 = X2[50:]
# predict the labels
y2 = clf.predict(X2)



# plot the data
fig, ax = plt.subplots(figsize=(8, 8))
point_style = dict(cmap='Paired', s=50)
ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)
# format plot
format_plot(ax, 'Input Data')
ax.axis([-1, 4, -2, 7])
plt.show()


# have two-dimensional data;
#   that is, we have two features for each point, 
#   represented by the (x,y) positions of the points on the plane
# In addition, we have one of two class labels for each point, 
#   here represented by the colors of the points
# From these features and labels, 
#   we would like to create a model that will let us decide 
#   whether a new point should be labeled “blue” or “red.”

# number of possible models for such a classification task
#  will make the assumption that 
#   the two groups can be separated 
#   by drawing a straight line through the plane between them, 
#   such that points on each side of the line fall in the same group
# model 
#   is a quantitative version of the statement “a straight line separates the classes,” while the 
# model parameters 
#   are the particular numbers describing the location and orientation of that line for our data
# optimal values for these model parameters are learned from the data 
#   (this is the “learning” in machine learning), 
#   which is often called training the model

# Get contours describing the model
xx = np.linspace(-1, 4, 10)
yy = np.linspace(-2, 7, 10)
xy1, xy2 = np.meshgrid(xx, yy)
Z = np.array([clf.decision_function([t])
              for t in zip(xy1.flat, xy2.flat)]).reshape(xy1.shape)
# plot points and model
fig, ax = plt.subplots(figsize=(8, 6))
line_style = dict(levels = [-1.0, 0.0, 1.0],
                  linestyles = ['dashed', 'solid', 'dashed'],
                  colors = 'gray', linewidths=1)
ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)
ax.contour(xy1, xy2, Z, **line_style)
# format plot
format_plot(ax, 'Model Learned from Input Data')
ax.axis([-1, 4, -2, 7])
plt.show()


# Now that this model has been trained, it can be generalized to new, unlabeled data
# In other words, 
#   we can take a new set of data, 
#   draw this model line through it, and
#   assign labels to the new points based on this model. 
# This stage is usually called prediction

# plot the results
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
ax[0].scatter(X2[:, 0], X2[:, 1], c='gray', **point_style)
ax[0].axis([-1, 4, -2, 7])
ax[1].scatter(X2[:, 0], X2[:, 1], c=y2, **point_style)
ax[1].contour(xy1, xy2, Z, **line_style)
ax[1].axis([-1, 4, -2, 7])
format_plot(ax[0], 'Unknown Data')
format_plot(ax[1], 'Predicted Labels')
plt.show()


# This is the basic idea of a classification task in machine learning, 
# where “classification” indicates that the data has discrete class labels


#endregion

#region Regression: Predicting continuous labels
# The following code generates the figures from the regression section.
from sklearn.linear_model import LinearRegression
# Create some data for the regression
rng = np.random.RandomState(1)
X = rng.randn(200, 2)
y = np.dot(X, [-2, 1]) + 0.1 * rng.randn(X.shape[0])
# fit the regression model
model = LinearRegression()
model.fit(X, y)
# create some new points to predict
X2 = rng.randn(100, 2)
# predict the labels
y2 = model.predict(X2)


# plot data points
fig, ax = plt.subplots()
points = ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                    cmap='viridis')
# format plot
format_plot(ax, 'Input Data')
ax.axis([-4, 4, -3, 3])
plt.show()


# have two-dimensional data; 
# that is, 
#   there are two features describing each data point. 
#   The color of each point represents the continuous label for that point

# number of possible regression models we might use for this type of data
# simple linear regression model 
#   assumes that if we treat the label as a third spatial dimension, 
#   we can fit a plane to the data
# is a higher-level generalization of the well-known problem 
#   of fitting a line to data with two coordinates


# can visualize this setup as
from mpl_toolkits.mplot3d.art3d import Line3DCollection
points = np.hstack([X, y[:, None]]).reshape(-1, 1, 3)
segments = np.hstack([points, points])
segments[:, 0, 2] = -8
# plot points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c=y, s=35,
           cmap='viridis')
ax.add_collection3d(Line3DCollection(segments, colors='gray', alpha=0.2))
ax.scatter(X[:, 0], X[:, 1], -8 + np.zeros(X.shape[0]), c=y, s=10,
           cmap='viridis')
# format plot
ax.patch.set_facecolor('white')
ax.view_init(elev=20, azim=-70)
ax.set_zlim3d(-8, 8)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.zaxis.set_major_formatter(plt.NullFormatter())
ax.set(xlabel='feature 1', ylabel='feature 2', zlabel='label')
# Hide axes (is there a better way?)
ax.w_xaxis.line.set_visible(False)
ax.w_yaxis.line.set_visible(False)
ax.w_zaxis.line.set_visible(False)
for tick in ax.w_xaxis.get_ticklines():
    tick.set_visible(False)
for tick in ax.w_yaxis.get_ticklines():
    tick.set_visible(False)
for tick in ax.w_zaxis.get_ticklines():
    tick.set_visible(False)
plt.show()


# feature 1–feature 2 plane here is the same as in the two-dimensional plot from before; 
# in this case, however, we have represented the labels by both color and three-dimensional axis position
# From this view, 
#   it seems reasonable that fitting a plane through this three-dimensional data 
#   would allow us to predict the expected label for any set of input parameters


# when we fit such a plane we get the result
from matplotlib.collections import LineCollection
# plot data points
fig, ax = plt.subplots()
pts = ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                 cmap='viridis', zorder=2)
# compute and plot model color mesh
xx, yy = np.meshgrid(np.linspace(-4, 4),
                     np.linspace(-3, 3))
Xfit = np.vstack([xx.ravel(), yy.ravel()]).T
yfit = model.predict(Xfit)
zz = yfit.reshape(xx.shape)
ax.pcolorfast([-4, 4], [-3, 3], zz, alpha=0.5,
              cmap='viridis', norm=pts.norm, zorder=1)
# format plot
format_plot(ax, 'Input Data with Linear Fit')
ax.axis([-4, 4, -3, 3])
plt.show()


# Visually, we find the results
# plot the model fit
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
ax[0].scatter(X2[:, 0], X2[:, 1], c='gray', s=50)
ax[0].axis([-4, 4, -3, 3])
ax[1].scatter(X2[:, 0], X2[:, 1], c=y2, s=50,
              cmap='viridis', norm=pts.norm)
ax[1].axis([-4, 4, -3, 3])
# format plots
format_plot(ax[0], 'Unknown Data')
format_plot(ax[1], 'Predicted Labels')
plt.show()


# power of these methods is that they can be straightforwardly applied and evaluated 
# in the case of data with many, many features


#endregion

#region Clustering: Inferring labels on unlabeled data
# “clustering,” 
#   in which data is automatically assigned to some number of discrete groups


# The following code generates the figures from the clustering section.
from sklearn.cluster import KMeans
# create 50 separable points
X, y = make_blobs(n_samples=100, centers=4,
                  random_state=42, cluster_std=1.5)
# Fit the K Means model
model = KMeans(4, random_state=0)
y = model.fit_predict(X)


# plot the input data
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], s=50, color='gray')
# format the plot
format_plot(ax, 'Input Data')
plt.show()


# By eye, it is clear that each of these points is part of a distinct group
# Given this input, a clustering model will
#   use the intrinsic structure of the data 
#   to determine which points are related


# k-means fits a model consisting of k cluster centers; 
#   the optimal centers are assumed to be those 
#   that minimize the distance of each point from its assigned center

# plot the data with cluster labels
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], s=50, c=y, cmap='viridis')
# format the plot
format_plot(ax, 'Learned Cluster Labels')
plt.show()


#endregion

#region Dimensionality reduction: Inferring structure of unlabeled data
# Dimensionality reduction 
#   is a bit more abstract than the examples we looked at before, 
# but generally 
#   it seeks to pull out some low-dimensional representation of data
#   that in some way preserves relevant qualities of the full dataset
# Different dimensionality reduction routines 
#   measure these relevant qualities in different ways


# following code generates the figures from the dimensionality reduction section.
from sklearn.datasets import make_swiss_roll
# make data
X, y = make_swiss_roll(200, noise=0.5, random_state=42)
X = X[:, [0, 2]]
# visualize data
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], color='gray', s=30)
# format the plot
format_plot(ax, 'Input Data')
plt.show()


# Visually, it is clear that there is some structure in this data: 
#   it is drawn from a one-dimensional line 
#   that is arranged in a spiral within this two-dimensional space
# could say that this data is “intrinsically” only one dimensional, 
#   though this one-dimensional data is embedded in higher-dimensional space
# suitable dimensionality reduction model in this case 
#   would be sensitive to this nonlinear embedded structure, 
#   and be able to pull out this lower-dimensionality representation


# visualization of the results of the Isomap algorithm, 
#   a manifold learning algorithm that does exactly this
from sklearn.manifold import Isomap
model = Isomap(n_neighbors=8, n_components=1)
y_fit = model.fit_transform(X).ravel()
# visualize data
fig, ax = plt.subplots()
pts = ax.scatter(X[:, 0], X[:, 1], c=y_fit, cmap='viridis', s=30)
cb = fig.colorbar(pts, ax=ax)
# format the plot
format_plot(ax, 'Learned Latent Parameter')
cb.set_ticks([])
cb.set_label('Latent Variable', color='gray')
plt.show()


# colors 
#   (which represent the extracted one-dimensional latent variable)
# change uniformly along the spiral, 
#   which indicates that the algorithm did in fact detect the structure we saw by eye


#endregion


#endregion

#region Summary


#endregion


#endregion

#region Introducing Scikit-Learn
# several Python libraries that provide solid implementations of a range of machine learning algorithms
# One of the best known is Scikit-Learn
#   a package that provides efficient versions of a large number of common algorithms
#   is characterized
#       by a clean, uniform, and streamlined API, as well as 
#       by very useful and complete online documentation
#   benefit of this uniformity is that 
#       once you understand the basic use and syntax of Scikit-Learn for one type of model, 
#       switching to a new model or algorithm is very straightforward


#region Data Representation in Scikit-Learn
# best way to think about data within Scikit-Learn is in terms of tables of data


#region Data as table
# basic table is a two-dimensional grid of data, in which 
#   the rows represent individual elements of the dataset, and 
#   the columns represent quantities related to each of these elements
# In general, we will refer to 
#   the columns of the matrix as features, and 
#   the number of columns as n_features


# e.g.
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()


#endregion

#region Features matrix
# This table layout makes clear that 
# the information can be thought of as a two-dimensional numerical array or matrix, 
#   which we will call the features matrix

# By convention, this features matrix is often stored in a variable named X
# The features matrix is assumed to be two-dimensional, 
#   with shape [n_samples, n_features], and 
#   is most often contained in 
#       a NumPy array or 
#       a Pandas DataFrame, 
#       though some ScikitLearn models also accept SciPy sparse matrices

# samples (i.e., rows) always refer to the individual objects described by the dataset
# features (i.e., columns) always refer to the distinct observations that describe each sample in a quantitative manner
#   Features 
#       are generally real-valued, 
#       but may be Boolean or discrete-valued in some cases


#endregion

#region Target array
# In addition to the feature matrix X, 
# we also generally work with a label or target array,
#   which by convention we will usually call y

# target array 
#   is usually one dimensional, 
#       with length n_samples, and 
#   is generally contained in a NumPy array or Pandas Series

# target array may have 
#   continuous numerical values, 
#   or discrete classes/labels

# some Scikit-Learn estimators do handle multiple target values 
#   in the form of a two-dimensional [n_samples, n_targets] target array

# Often one point of confusion is how the target array differs from the other features columns
#   distinguishing feature of the target array is that 
#       it is usually the quantity we want to predict from the data: 
#       in statistical terms, it is the dependent variable

# e.g.
# in the preceding data we may wish to construct a model that can predict the 
# species of flower based on the other measurements; 
# in this case, the species column would be considered the feature

# use Seaborn to conveniently visualize the data
sns.set()
sns.pairplot(iris, hue='species', size=1.5)
plt.show()

# For use in Scikit-Learn, 
# we will extract the features matrix and target array from the DataFrame, 
# which we can do using some of the Pandas DataFrame operations
X_iris = iris.drop('species', axis=1); X_iris.shape
y_iris = iris['species']; y_iris.shape


#endregion


#endregion

#region Scikit-Learn’s Estimator API
# Scikit-Learn API is designed with the following guiding principles in mind
#   Consistency
#       All objects share a common interface drawn from a limited set of methods, 
#       with consistent documentation.
#   Inspection
#       All specified parameter values are exposed as public attributes.
#   Limited object hierarchy
#       Only algorithms are represented by Python classes; 
#       datasets are represented in standard formats
#           (NumPy arrays, Pandas DataFrames, SciPy sparse matrices) and
#       parameter names use standard Python strings.
#   Composition
#       Many machine learning tasks can be expressed as sequences of more fundamental algorithms, and 
#       Scikit-Learn makes use of this wherever possible.
#   Sensible defaults
#       When models require user-specified parameters, 
#       the library defines an appropriate default value

# In practice, these principles make Scikit-Learn very easy to use, once the basic principles are understood
# Every machine learning algorithm in Scikit-Learn is implemented via the Estimator API, 
# which provides a consistent interface for a wide range of machine learning applications


#region Basics of the API
# Most commonly, the steps in using the Scikit-Learn estimator API are as follows
#   1. Choose a class of model by importing the appropriate estimator class from ScikitLearn.
#   2. Choose model hyperparameters by instantiating this class with desired values.
#   3. Arrange data into a features matrix and target vector
#   4. Fit the model to your data by calling the fit() method of the model instance.
#   5. Apply the model to new data:
#       • For supervised learning,
#           often we predict labels for unknown data 
#           using the predict() method.
#       • For unsupervised learning, 
#           we often transform or infer properties of the data 
#           using the transform() or predict() method


#endregion

#region Supervised learning example: Simple linear regression
# simple data for our regression example 
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
plt.show()

# walk through the process
# 1. Choose a class of model
# In Scikit-Learn, every class of model is represented by a Python class
# to compute a simple linear regression model, we can import the linear regression class
from sklearn.linear_model import LinearRegression

# other, more general linear regression models exist as well


# 2. Choose model hyperparameters
# important point is that a class of model is not the same as an instance of a model
# Once we have decided on our model class, there are still some options open to us
# Depending on the model class we are working with, we might need to answer one or more questions like the following:
#   • Would we like to fit for the offset (i.e., intercept)?
#   • Would we like the model to be normalized?
#   • Would we like to preprocess our features to add model flexibility?
#   • What degree of regularization would we like to use in our model?
#   • How many model components would we like to use?

# These are examples of the important choices that must be made once the model class is selected
# These choices are often represented as hyperparameters, 
#   or parameters that must be set before the model is fit to data

# In Scikit-Learn, we choose hyperparameters 
#   by passing values at model instantiation

# can instantiate the LinearRegression class 
# and specify that we would like to fit the intercept 
# using the fit_intercept hyperparameter
model = LinearRegression(fit_intercept=True); model

# when the model is instantiated, the only action is the storing of these hyperparameter values
# have not yet applied the model to any data
# Scikit-Learn API makes very clear the distinction between 
#   choice of model and 
#   application of model to data


# 3. Arrange data into a features matrix and target vector
# target variable y is already in the correct form (a length-n_samples array)
# need to massage the data x to make it a matrix of size [n_samples, n_features]

# this amounts to a simple reshaping of the one-dimensional array
X = x[:, np.newaxis]; X.shape


# 4. Fit the model to your data
# to apply our model to data
#  can be done with the fit() method of the model
model.fit(X, y)

# fit() command 
#   causes a number of model-dependent internal computations to take place, 
#   and the results of these computations are stored in model-specific attributes 
#       that the user can explore
# In Scikit-Learn, by convention 
#   all model parameters that were learned during the fit() process 
#   have trailing underscores

model.coef_; model.intercept_
# Comparing to the data definition, 
# we see that they are very close to the
#   input slope of 2 and 
#   intercept of –1


# One question that frequently comes up regards the uncertainty in such internal model parameters
# In general, Scikit-Learn does not provide tools to draw conclusions from internal model parameters themselves
# interpreting model parameters 
#   is much more a statistical modeling question 
#   than a machine learning question
# Machine learning rather focuses on what the model predicts
# to dive into the meaning of fit parameters within the model, other tools are available


# 5. Predict labels for unknown data
# Once the model is trained, the main task of supervised machine learning 
#   is to evaluate it based on what it says about new data that was not part of the training set
# In Scikit-Learn, we can do this using the predict() method

# For the sake of this example, 
# our “new data” will be a grid of x values, 
# and we will ask what y values the model predicts
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# visualize the results by plotting first the raw data, and then this model fit
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()


#endregion

#region Supervised learning example: Iris classiication
# question will be this: 
#   given a model trained on a portion of the Iris data,
#   how well can we predict the remaining labels?

# use an extremely simple generative model known as Gaussian naive Bayes, 
#   which proceeds by assuming each class is drawn from an axis-aligned Gaussian distribution
# Because it is so fast and has no hyperparameters to choose, 
#   Gaussian naive Bayes is often a good model to use as a baseline classification, 
#   before you explore whether improvements can be found through more sophisticated models

# like to evaluate the model on data it has not seen before, 
#   and so we will split the data into a training set and a testing set
# could be done by hand, 
#   but it is more convenient to use the train_test_split utility function
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)


# With the data arranged, we can follow our recipe to predict the labels
from sklearn.naive_bayes import GaussianNB      # 1. choose model class 
model = GaussianNB()                            # 2. instantiate model  
model.fit(Xtrain, ytrain)                       # 3. fit model to data 
y_model = model.predict(Xtest)                  # 4. predict on new data


# use the accuracy_score utility to see the fraction of predicted labels that match their true value
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

# With an accuracy topping 97%, 
# we see that even this very naive classification algorithm is effective for this particular dataset!


#endregion

#region Unsupervised learning example: Iris dimensionality
# look at reducing the dimensionality of the Iris data so as to more easily visualize it
#  Iris data is four dimensional: 
#   there are four features recorded for each sample

# task of dimensionality reduction is to ask 
#   whether there is a suitable lower-dimensional representation 
#   that retains the essential features of the data

# Often dimensionality reduction is used as an aid to visualizing data; 
# after all, it is much easier to plot data in two dimensions than in four dimensions or higher!

# will use principal component analysis
#   which is a fast linear dimensionality reduction technique
# will ask the model to return two components—
#   that is, a two-dimensional representation of the data

# Following the sequence of steps outlined earlier
from sklearn.decomposition import PCA   # 1. Choose the model class
model = PCA(n_components=2)             # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                       # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)          # 4. Transform the data to two dimensions

# plot the results
# quick way to do this is to 
#   insert the results into the original Iris DataFrame, 
#   and use Seaborn’s lmplot to show the results
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot(x='PCA1', y='PCA2', hue='species', data=iris, fit_reg=False)
plt.show()

# see that in the two-dimensional representation, 
#   the species are fairly well separated, 
#   even though the PCA algorithm had no knowledge of the species labels!
# This indicates to us that 
#   a relatively straightforward classification will probably be effective on the dataset


#endregion

#region Unsupervised learning: Iris clustering
# look at applying clustering to the Iris data
# clustering algorithm 
#   attempts to find distinct groups of data 
#   without reference to any labels

# use a powerful clustering method called a Gaussian mixture model (GMM)
#   GMM attempts to model the data as a collection of Gaussian blobs


# fit the Gaussian mixture model as follows
from sklearn.mixture import GaussianMixture                         # 1. Choose the model class
model = GaussianMixture(n_components=3, covariance_type='full')     # 2. Instantiate the model w/ hyperparameters
model.fit(X_iris)                                                   # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)                                       # 4. Determine cluster labels


# add the cluster label to the Iris DataFrame and use Seaborn to plot the results
iris['cluster'] = y_gmm
sns.lmplot(x='PCA1', y='PCA2', data=iris, hue='species', col='cluster', fit_reg=False)
plt.show()

# By splitting the data by cluster number, 
# we see exactly how well the GMM algorithm has recovered the underlying label: 
#   the setosa species is separated perfectly within cluster 1, 
#   while there remains a small amount of mixing between versicolor and virginica
#  This means that even without an expert to tell us the species labels of the individual flowers, 
#   the measurements of these flowers are distinct enough 
#   that we could automatically identify the presence of these different groups of species 
#   with a simple clustering algorithm! 
# This sort of algorithm might further give experts in the field clues
#   as to the relationship between the samples they are observing


#endregion


#endregion

#region Application: Exploring Handwritten Digits
# optical character recognition problem: the identification of handwritten digits
# In the wild, this problem involves both locating and identifying characters in an image

# we’ll take a shortcut and use Scikit-Learn’s set of preformatted digits


#region Loading and visualizing the digits data
from sklearn.datasets import load_digits
digits = load_digits(); digits.images.shape

# visualize the first hundred of these 
fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
plt.show()


# In order to work with this data within Scikit-Learn, 
#   we need a two-dimensional, [n_samples, n_features] representation
# can accomplish this by treating each pixel in the image as a feature—
#   that is, by flattening out the pixel arrays 
#   so that we have a length-64 array of pixel values representing each digit
# Additionally, we need the target array, 
#   which gives the previously determined label for each digit
# These two quantities are built into the digits dataset under the data and target attributes, respectively

X = digits.data; X.shape
y = digits.target; y.shape


#endregion

#region Unsupervised learning: Dimensionality reduction
# like to visualize our points within the 64-dimensional parameter space, 
#   but it’s difficult to effectively visualize points in such a high-dimensional space
# Instead we’ll reduce the dimensions to 2, using an unsupervised method

# make use of a manifold learning algorithm called Isomap
#   and transform the data to two dimensions

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

# plot this data to see if we can learn anything from its structure
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

# This plot gives us some good intuition into how well various numbers are separated in the larger 64-dimensional space
# e.g.
# zeros (in black) and ones (in purple) have very little overlap in parameter space
#   Intuitively, this makes sense: 
#       a zero is empty in the middle of the image, while 
#       a one will generally have ink in the middle
# On the other hand, there seems to be a more or less continuous spectrum between ones and fours: 
#   we can understand this by realizing that 
#       some people draw ones with “hats” on them, 
#       which cause them to look similar to fours
# Overall, however, the different groups appear to be fairly well separated in the parameter space: 
# this tells us that even a very straightforward supervised classification algorithm should perform suitably on this data


#endregion

#region Classiication on digits
# will split the data into a training and test set, and fit a Gaussian naive Bayes model
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# gauge its accuracy by comparing the true values of the test set to the predictions
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

# find about 80% accuracy for classification of the digits!
# However, this single number doesn’t tell us where we’ve gone wrong—
#   one nice way to do this is to 
#       use the confusion matrix, 
#       which we can compute with Scikit-Learn and plot with Seaborn
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

# This shows us where the mislabeled points tend to be: 
# e.g., a large number of twos here are misclassified as either ones or eights

# Another way to gain intuition into the characteristics of the model 
#   is to plot the inputs again, with their predicted labels
# We’ll use green for correct labels, and red for incorrect labels 
test_images = Xtest.reshape(-1, 8, 8)
fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]), transform=ax.transAxes, color='green' if (ytest[i] == y_model[i]) else 'red')
plt.show()


# Examining this subset of the data, we can gain insight regarding where the algorithm might not be performing optimally
# To go beyond our 80% classification rate, we might move to a more sophisticated algorithm


#endregion


#endregion

#region Summary


#endregion

#endregion

#region Hyperparameters and Model Validation
# choice of model and choice of hyperparameters—
#   are perhaps the most important part of using these tools and techniques effectively

# In order to make an informed choice, we need a way to validate that 
#   our model and our hyperparameters are a good fit to the data
# there are some pitfalls that you must avoid to do this effectively


#region Thinking About Model Validation
# In principle, model validation is very simple: 
#   after choosing a model and its hyperparameters, 
#   we can estimate how effective it is 
#       by applying it to some of the training data and 
#       comparing the prediction to the known value


#region Model validation the wrong way
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# we’ll use a k-neighbors classifier with n_neighbors=1
# This is a very simple and intuitive model that says 
#   “the label of an unknown point is the same as the label of its closest training point”
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

# train the model, and use it to predict labels for data we already know
model.fit(X, y)
y_model = model.predict(X)

# compute the fraction of correctly labeled points
from sklearn.metrics import accuracy_score
accuracy_score(y, y_model)

# We see an accuracy score of 1.0, which indicates that 100% of points were correctly labeled by our model! 
# But is this truly measuring the expected accuracy? 
#   Have we really come upon a model that we expect to be correct 100% of the time?
# answer is no
# In fact, this approach contains a fundamental flaw: 
#   it trains and evaluates the model on the same data
# Furthermore, the nearest neighbor model is an instance-based estimator 
#   that simply stores the training data, 
#   and predicts labels by comparing new data to these stored points; 
# except in contrived cases, it will get 100% accuracy every time!


#endregion

#region Model validation the right way: Holdout sets
# can get a better sense of a model’s performance using what’s known as a holdout set; 
#   that is, we hold back some subset of the data from the training of the model, 
#   and then use this holdout set to check the model performance
# can do this splitting using the train_test_split utility in Scikit-Learn

from sklearn.model_selection import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

# fit the model on one set of data
model.fit(X1, y1)

# evaluate the model on the second set of data
y_model = model.predict(X2)
accuracy_score(y2, y_model)

# see here a more reasonable result: 
#   the nearest-neighbor classifier is about 90% accurate on this holdout set
# holdout set is similar to unknown data, because the model has not “seen” it before


#endregion

#region Model validation via cross-validation
# One disadvantage of using a holdout set for model validation is that 
#   we have lost a portion of our data to the model training
# In the previous case, half the dataset does not contribute to the training of the model!
# This is not optimal, and can cause problems—
#   especially if the initial set of training data is small

# One way to address this is to use cross-validation—
#   that is, to do a sequence of fits
#   where each subset of the data is used 
#       both as a training set and as a validation set
# Here we do two validation trials, 
#   alternately using each half of the data as a holdout set

# Using the split data from before, we could implement it like this
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model); accuracy_score(y2, y2_model)

# What comes out are two accuracy scores, 
# which we could combine 
#   (by, say, taking the mean) 
# to get a better measure of the global model performance

# This particular form of cross-validation is a two-fold cross-validation—
#   one in which we have split the data into two sets 
#   and used each in turn as a validation set

# could expand on this idea to use even more trials, and more folds in the data
# e.g. five-fold cross-validation
# would be rather tedious to do by hand,
# and so we can use Scikit-Learn’s cross_val_score convenience routine to do it succinctly

from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)

# Repeating the validation across different subsets of the data 
# gives us an even better idea of the performance of the algorithm

# Scikit-Learn implements a number of cross-validation schemes that are useful in particular situations; 
# these are implemented via iterators in the cross_validation module

# e.g.
# might wish to go to the extreme case in which 
# our number of folds is equal to the number of data points; 
#   that is, we train on all points but one in each trial
# This type of cross-validation is known as leave-one-out cross-validation

from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut()); scores

# Because we have 150 samples, the leave-one-out cross-validation yields scores for 150 trials, 
# and the score indicates either successful (1.0) or unsuccessful (0.0) prediction

# Taking the mean of these gives an estimate of the error rate
scores.mean()


# Other cross-validation schemes can be used similarly


#endregion


#endregion

#region Selecting the Best Model
# Of core importance is the following question: 
# if our estimator is underperforming, how should we move forward? 
# There are several possible answers:
#    • Use a more complicated/more flexible model
#    • Use a less complicated/less flexible model
#    • Gather more training samples
#    • Gather more data to add features to each sample

# answer to this question is often counterintuitive
#   sometimes using a more complicated model will give worse results, and 
#   adding more training samples may not improve your results! 

# ability to determine what steps will improve your model 
# is what separates the successful machine learning practitioners from the unsuccessful


#region The bias–variance trade-of
# Fundamentally, the question of “the best model” is about 
#   finding a sweet spot in the trade-off between bias and variance

# model has high bias
#   Such a model is said to underit the data; 
#   that is,    
#       it does not have enough model flexibility 
#       to suitably account for all the features in the data.

# model has high variance
#   Such a model is said to overit the data; 
#   that is, 
#       it has so much model flexibility that the
#       model ends up accounting for random errors as well as the underlying data distribution.


# R**2 score, or coefficient of determination
#   measures how well a model performs relative to a simple mean of the target values
#   R**2 = 1 indicates a perfect match, 
#   R**2 = 0 indicates the model does no better than simply taking the mean of the data, and 
#   negative values mean even worse models

# an observation that holds more generally:
#    • For high-bias models, 
#       the performance of the model on the validation set 
#       is similar to the performance on the training set.
#    • For high-variance models, 
#       the performance of the model on the validation set is
#       far worse than the performance on the training set

# validation curve, 
# we see the following essential features:
#    • The training score is everywhere higher than the validation score. 
#       This is generally the case: 
#       the model will be a better fit to data it has seen 
#       than to data it has not seen.
#    • For very low model complexity (a high-bias model), 
#       the training data is underfit,
#       which means that the model is a poor predictor 
#       both for the training data and for any previously unseen data.
#    • For very high model complexity (a high-variance model), 
#       the training data is overfit, 
#       which means that the model predicts the training data very well, 
#       but fails for any previously unseen data.
#    • For some intermediate value, 
#       the validation curve has a maximum. 
#       This level of complexity indicates a suitable trade-off between bias and variance


# means of tuning the model complexity varies from model to model


#endregion

#region Validation curves in Scikit-Learn
# using cross-validation to compute the validation curve for a class of models

# use a polynomial regression model: 
#   this is a generalized linear model in which 
#   the degree of the polynomial is a tunable parameter
# e.g.
#   degree-1 polynomial fits a straight line to the data; 
#       for model parameters a and b:
#       y = ax + b
#   A degree-3 polynomial fits a cubic curve to the data; 
#       for model parameters a, b, c, d:
#       y = ax**3 + bx**2 + cx + d

# can generalize this to any number of polynomial features

# In Scikit-Learn, we can implement this 
#   with a simple linear regression combined with the polynomial pre‐processor
# will use a pipeline to string these operations together

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# create some data to which we will fit our model
def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)

# visualize our data, along with polynomial fits of several degrees
X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
plt.show()


# knob controlling model complexity in this case 
#   is the degree of the polynomial, 
#   which can be any non-negative integer
# useful question to answer is this: 
#   what degree of polynomial provides a suitable trade-off 
#   between bias (underfitting) and variance (overfitting)?

# can make progress in this 
#   by visualizing the validation curve for this particular data and model; 
# we can do this straightforwardly using the 
#   validation_curve convenience routine provided by Scikit-Learn

from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y, param_name = 'polynomialfeatures__degree', param_range = degree, cv=7)
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()


# This shows precisely the qualitative behavior we expect: 
#   the training score is everywhere higher than the validation score; 
#   the training score is monotonically improving with increased model complexity; and 
#   the validation score reaches a maximum before dropping off as the model becomes overfit


# From the validation curve, we can read off that the optimal trade-off between bias and variance 
# is found for a third-order polynomial; 
# we can compute and display this fit over the original data as follows
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.show()


# finding this optimal model did not actually require us to compute the training score, 
#   but examining the relationship between the training score and validation score 
#   can give us useful insight into the performance of the model


#endregion


#endregion

#region Learning Curves
# One important aspect of model complexity is that 
#   the optimal model will generally depend on the size of your training data

# e.g. generate a new dataset with a factor of five more points 
X2, y2 = make_data(200)
plt.scatter(X2.ravel(), y2)
plt.show()


# will duplicate the preceding code to plot the validation curve for this larger dataset; 
# for reference let’s over-plot the previous results as well 

degree = np.arange(0, 21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2, param_name = 'polynomialfeatures__degree', param_range = degree, cv=7)
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=0.3, linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=0.3, linestyle='dashed')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()

# It is clear from the validation curve that the larger dataset can support a much more complicated model: 
#   the peak here is probably around a degree of 6, 
#   but even a degree-20 model is not seriously overfitting the data—the validation and training scores remain very close

# Thus we see that the behavior of the validation curve has not one, but two, important inputs: 
#   the model complexity and 
#   the number of training points
# often useful to explore the behavior of the model as a function of the number of training points,
#   which we can do by using increasingly larger subsets of the data to fit our model

# plot of the training/validation score with respect to the size of the training set 
#   is known as a learning curve

#  general behavior we would expect from a learning curve is this:
#    • A model of a given complexity will overfit a small dataset: 
#       this means the training score will be relatively high, 
#       while the validation score will be relatively low.
#    • A model of a given complexity will underfit a large dataset: 
#       this means that the training score will decrease, 
#       but the validation score will increase.
#    • A model will never, except by chance, give a better score to the validation set than the training set: 
#       this means the curves should keep getting closer together but never cross

# notable feature of the learning curve is 
#   the convergence to a particular score 
#   as the number of training samples grows
# once you have enough points that a particular model has converged, 
#   adding more training data will not help you!
# only way to increase model performance in this case is to use another (often more complex) model


#region Learning curves in Scikit-Learn
# Scikit-Learn offers a convenient utility for computing such learning curves from your models

# e.g.
# will compute a learning curve for our original dataset 
# with a second-order polynomial model and a ninth-order polynomial

from sklearn.model_selection import learning_curve

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree), X, y, cv=7, train_sizes=np.linspace(0.3, 1, 25))

    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyle='dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree={0}'.format(degree), size=14)
    ax[i].legend(loc='best')
plt.show()


# This is a valuable diagnostic, 
#   because it gives us a visual depiction of how our model responds to increasing training data
# when your learning curve has already converged 
#   (i.e., when the training and validation curves are already close to each other), 
# adding more training data will not signiicantly improve the it!
# This situation is seen in the left panel, with the learning curve for the degree-2 model

# only way to increase the converged score is to use a different (usually more complicated) model
# see this in the right panel: 
#   by moving to a much more complicated model, 
#   we increase the score of convergence (indicated by the dashed line), 
#   but at the expense of higher model variance (indicated by the difference between the training and validation scores)
#  If we were to add even more data points, 
#   the learning curve for the more complicated model would eventually converge


# Plotting a learning curve for your particular choice of model and dataset 
# can help you to make this type of decision about how to move forward in improving your analysis


#endregion


#endregion

#region Validation in Practice: Grid Search
# In practice, models generally have more than one knob to turn, and thus 
#   plots of validation and learning curves 
#   change from lines to multidimensional surfaces
# In these cases, such visualizations are difficult and 
#   we would rather simply find the particular model that maximizes the validation score

# Scikit-Learn provides automated tools to do this in the grid_search module

# e.g. 
# to find the optimal polynomial model
# explore a three-dimensional grid of model features—namely, 
#   the polynomial degree,
#   the flag telling us whether to fit the intercept, and 
#   the flag telling us whether to normalize the problem
# set this up using Scikit-Learn’s GridSearchCV metaestimator

from sklearn.model_selection import GridSearchCV
param_grid = {
    'polynomialfeatures__degree': np.arange(21),
    'linearregression__fit_intercept': [True, False]
    # 'linearregression__normalize': [True, False],     # normalize has been removed
}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)

# like a normal estimator, this has not yet been applied to any data
# Calling the fit() method will fit the model at each grid point, 
#   keeping track of the scores along the way
grid.fit(X, y)

# ask for the best parameters as follows
grid.best_params_; grid.best_score_

# use the best model and show the fit to our data using code from before
model = grid.best_estimator_

plt. scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.show()


#endregion

#region Summary


#endregion


#endregion

#region Feature Engineering
# one of the more important steps in using machine learning in practice is feature engineering—
#   that is, 
#   taking whatever information you have about your problem and
#   turning it into numbers that you can use to build your feature matrix
# few common examples of feature engineering tasks:
#   features for representing categorical data, 
#   features for representing text, and 
#   features for representing images
# derived features 
#   for increasing model complexity and imputation of missing data. 
#   Often this process is known as vectorization, 
#       as it involves converting arbitrary data into well-behaved vectors


#region Categorical Features
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

# might be tempted to encode this data with a straightforward numerical mapping
{'Queen Anne': 1, 'Fremont': 2, 'Wallingford': 3}

# this is not generally a useful approach in Scikit-Learn: 
# the package’s models make the fundamental assumption that numerical features reflect algebraic quantities
#   Thus such a mapping would imply, for example, 
#       that Queen Anne < Fremont < Wallingford, 
#       or even that Wallingford - Queen Anne = Fremont, 
#   which does not make much sense

# In this case, one proven technique is to use one-hot encoding, 
#   which effectively creates extra columns 
#   indicating the presence or absence of a category with a value of 1 or 0, respectively
# When your data comes as a list of dictionaries, 
#   Scikit-Learn’s DictVectorizer will do this for you

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)

# neighborhood column has been expanded into three separate columns,
#   representing the three neighborhood labels, and 
#   that each row has a 1 in the column associated with its neighborhood
# With these categorical features thus encoded,
#   can proceed as normal with fitting a Scikit-Learn model

# To see the meaning of each column, you can inspect the feature names
vec.get_feature_names_out()

# There is one clear disadvantage of this approach: 
#   if your category has many possible values, 
#   this can greatly increase the size of your dataset
# However, because the encoded data contains mostly zeros, 
#   a sparse output can be a very efficient solution

vec = DictVectorizer(sparse=True, dtype=int)
X = vec.fit_transform(data)

# sklearn.preprocessing.OneHotEncoder and
# sklearn.feature_extraction.FeatureHasher 
# are two additional tools that ScikitLearn includes to support this type of encoding


#endregion

#region Text Features
# common need in feature engineering is to convert text to a set of representative numerical values
# One of the simplest methods of encoding data is by word counts: 
#   you take each snippet of text, 
#   count the occurrences of each word within it, and 
#   put the results in a table

# e.g.
sample = ['problem of evil',
            'evil queen',
            'horizon problem']

# For a vectorization of this data based on word count, 
# we could construct a column representing 
#   the word “problem,” the word “evil,” the word “horizon,” and so on
# While doing this by hand would be possible, 
#   we can avoid the tedium by using ScikitLearn’s CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample); X

# result is a sparse matrix recording the number of times each word appears; 
# it is easier to inspect if we convert this to a DataFrame with labeled columns
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())


# some issues with this approach, however: 
#   the raw word counts lead to features 
#   that put too much weight on words that appear very frequently, and 
#   this can be suboptimal in some classification algorithms

# One approach to fix this is known as term frequency–inverse document frequency (TF–IDF), 
#   which weights the word counts 
#   by a measure of how often they appear in the documents

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())


#endregion

#region Image Features
# common need is to suitably encode images for machine learning analysis
# simplest approach is using the pixel values themselves
# But depending on the application, such approaches may not be optimal

# can find excellent implementations of many of the standard approaches in the Scikit-Image project


#endregion

#region Derived Features
# Another useful type of feature is 
# one that is mathematically derived from some input features

# convert a linear regression into a polynomial regression 
#   not by changing the model, 
#   but by transforming the input! 
# This is sometimes known as basis function regression

# e.g.
# data clearly cannot be well described by a straight line
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)
plt.show()

# Still, we can fit a line to the data using LinearRegression and get the optimal result
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)
plt.show()

# clear that we need a more sophisticated model to describe the relationship between x and y

# can do this by transforming the data, 
#   adding extra columns of features to drive more flexibility in the model
# e.g. add polynomial features to the data
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X); X2

# derived feature matrix has 
#   one column representing x, and 
#   a second column representing x**2, and 
#   a third column representing x**3

# Computing a linear regression on this expanded input gives a much closer fit to our data
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)
plt.show()


# This idea of improving a model 
#   not by changing the model, 
#   but by transforming the inputs, 
# is fundamental to many of the more powerful machine learning methods

# this is one motivational path to the powerful set of techniques known as kernel methods


#endregion

#region Imputation of Missing Data
# common need in feature engineering is handling missing data

# e.g. dataset
from numpy import nan
X = np.array([[ nan, 0, 3 ],
                [ 3, 7, 9 ],
                [ 3, 5, 2 ],
                [ 4, nan, 6 ],
                [ 8, 8, 1 ]])
y = np.array([14, 16, -1, 8, -5])


# When applying a typical machine learning model to such data, 
# we will need to first replace such missing data with some appropriate fill value
#   This is known as imputation of missing values, 
# and strategies range from 
#   simple 
#       (e.g., replacing missing values with the mean of the column) to 
#   sophisticated 
#   (e.g., using matrix completion or a robust model to handle such data)
# sophisticated approaches tend to be very application-specific

# For a baseline imputation approach, 
#   using the mean, median, or most frequent value, 
# Scikit-Learn provides the Imputer class
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X); X2

# in the resulting data, the two missing values have been replaced with the mean of the remaining values in the column

# This imputed data can then be fed directly into, for example, a LinearRegression estimator
model = LinearRegression().fit(X2, y)
model.predict(X2)


#endregion

#region Feature Pipelines
# it can quickly become tedious to do the transformations by hand, 
# especially if you wish to string together multiple steps

# e.g.
# might want a processing pipeline that looks something like this
#   1. Impute missing values using the mean
#   2. Transform features to quadratic
#   3. Fit a linear regression


# To streamline this type of processing pipeline, Scikit-Learn provides a pipeline object
from sklearn.pipeline import make_pipeline

model = make_pipeline(SimpleImputer(strategy='mean'),
                        PolynomialFeatures(degree=2),
                        LinearRegression())

# This pipeline looks and acts like a standard Scikit-Learn object, 
# and will apply all the specified steps to any input data
#   All the steps of the model are applied automatically

model.fit(X, y)     # X with missing values, from above
y; model.predict(X)


#endregion


#endregion
