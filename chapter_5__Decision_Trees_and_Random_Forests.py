# CHAPTER 5: In-Depth: Decision Trees and Random Forests
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits

# -------------------------------------------------------

# random forests
#   is a non‐parametric algorithm
# are an example of an ensemble method, 
#   a method that relies on aggregating the results of an ensemble of simpler estimators
# somewhat surprising result with such ensemble methods is that 
#   the sum can be greater than the parts; 
#       that is, a majority vote among a number of estimators 
#       can end up being better than 
#       any of the individual estimators doing the voting!

#region Motivating Random Forests: Decision Trees
# Random forests are an example of an ensemble learner built on decision trees

# Decision trees are extremely intuitive ways to classify or label objects: 
#   you simply ask a series of questions designed to zero in on the classification
# binary splitting makes this extremely efficient: 
#   in a well-constructed tree, 
#   each question will cut the number of options by approximately half, 
#   very quickly narrowing the options even among a large number of classes
#  trick, of course, comes in deciding which questions to ask at each step
# In machine learning implementations of decision trees, 
#   the questions generally take the form of axis-aligned splits in the data;
#       that is, each node in the tree splits the data into two groups 
#       using a cutoff value within one of the features


#region Creating a decision tree
# following two-dimensional data, which has one of four class labels
# from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow'); plt.show()


# A simple decision tree built on this data 
#   will iteratively split the data 
#       along one or the other axis 
#       according to some quantitative criterion, 
#   and at each level 
#       assign the label of the new region 
#       according to a majority vote of points within it

# after each split, 
#   Except for nodes that contain all of one features, 
#   at each level every region is again split along one of the two features

# This process of fitting a decision tree to our data can be done in Scikit-Learn with the
#   DecisionTreeClassifier estimator

# from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)

# quick utility function to help us visualize the output of the classifier
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200)
                            , np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3
                            , levels=np.arange(n_classes + 1) - 0.5
                            , cmap=cmap, clim=(y.min(), y.max())
                            , zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


# examine what the decision tree classification looks like
visualize_classifier(DecisionTreeClassifier(), X, y); plt.show()

    
# as the depth increases, we tend to get very strangely shaped classification regions
#  this is less a result of the true, intrinsic data distribution, 
#   and more a result of the particular sampling or noise properties of the data
# That is, this decision tree, even at only five levels deep, is clearly overfitting our data


#endregion

#region Decision trees and overitting
# Such overfitting turns out to be a general property of decision trees; 
# it is very easy to go too deep in the tree, 
#   and thus to fit details of the particular data 
#   rather than the overall properties of the distributions they are drawn from

# Another way to see this overfitting is to look at models trained on different subsets of the data
# key observation is that 
#   the inconsistencies tend to happen 
#   where the classification is less certain, 
# and thus by using information from both of these trees, 
#   we might come up with a better result!


# Just as using information from two trees improves our results, 
#   we might expect that using information from many trees would improve our results even further.


#endregion


#endregion

#region Ensembles of Estimators: Random Forests
# This notion—
#   that multiple overfitting estimators can be combined to reduce the effect of this overfitting
# —is what underlies an ensemble method called bagging

# Bagging 
#   makes use of an ensemble (a grab bag, perhaps) of parallel estimators, 
#   each of which overfits the data, 
#   and averages the results to find a better classification

# An ensemble of randomized decision trees is known as a random forest

# can do this type of bagging classification manually using Scikit-Learn’s 
#   Bagging Classifier meta-estimator

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)

bag.fit(X, y)
visualize_classifier(bag, X, y); plt.show()

# have randomized the data by fitting each estimator with a random subset of 80% of the training points
# In practice, decision trees are more effectively randomized 
#   when some stochasticity is injected in how the splits are chosen;
#   this way, all the data contributes to the fit each time, 
#       but the results of the fit still have the desired randomness
# e.g. when determining which feature to split on,
#   the randomized tree might select from among the top several features

# In Scikit-Learn, such an optimized ensemble of randomized decision trees 
#   is implemented in the RandomForestClassifier estimator, 
#   which takes care of all the randomization automatically
# All you need to do is select a number of estimators, 
#   and it will very quickly (in parallel, if desired) fit the ensemble of trees

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y); plt.show()


# by averaging over 100 randomly perturbed models, 
#   we end up with an overall model 
#   that is much closer to our intuition about how the parameter space should be split.


#endregion

#region Random Forest Regression
# Random forests can also be made to work in the case of regression 
#   (that is, continuous rather than categorical variables)
# estimator to use for this is the 
#   RandomForestRegressor

# following data, drawn from the combination of a fast and slow oscillation
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)

def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))

    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, 0.3, fmt='o'); plt.show()


# Using the random forest regressor, we can find the best-fit curve
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)

plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit, yfit, '-r')
plt.plot(xfit, ytrue, '-k', alpha=0.5)
plt.show()


# true model is shown by the smooth curve, 
#   while the random forest model is shown by the jagged curve
# nonparametric random forest model is flexible enough to fit the multiperiod data, 
#   without us needing to specify a multiperiod model!


#endregion

#region Example: Random Forest for Classifying Digits
# look at the handwritten digits data
# to see how the random forest classifier can be used in this context

from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

# visualize the first few data points

# set up the figure
fig = plt.figure(figsize=(6, 6))    # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
plt.show()


# can quickly classify the digits using a random forest as follows
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

# look at the classification report for this classifier
from sklearn import metrics
print(metrics.classification_report(ypred, ytest))

# plot the confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# find that a simple, untuned random forest results in a very accurate classification of the digits data


#endregion

#region Summary of Random Forests
# Random forests are a powerful method with several advantages:
# • Both training and prediction are very fast, 
#       because of the simplicity of the underlying decision trees. 
#   In addition, both tasks can be straightforwardly parallelized,
#       because the individual trees are entirely independent entities.
# • The multiple trees allow for a probabilistic classification: 
#       a majority vote among estimators gives an estimate of the probability 
#       (accessed in Scikit-Learn with the predict_proba() method).
# • The nonparametric model is extremely flexible, 
#       and can thus perform well on tasks that are underfit by other estimators


# primary disadvantage of random forests is that 
#   the results are not easily interpretable; 
#       that is, if you would like to draw conclusions about the meaning of the classification model, 
#       random forests may not be the best choice


#endregion
