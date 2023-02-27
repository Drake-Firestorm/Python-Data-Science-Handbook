# CHAPTER 5: In Depth: Gaussian Mixture Models
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse

from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits

# -------------------------------------------------------------

# k-means clustering model is simple and relatively easy to understand, 
#   but its simplicity leads to practical challenges in its application
# In particular, 
#       the nonprobabilistic nature of k-means and 
#       its use of simple distance-from-cluster-center 
#   to assign cluster membership 
#   leads to poor performance for many real-world situations

# Gaussian mixture models, 
#   can be viewed as an extension of the ideas behind k-means, 
#   but can also be a powerful tool for estimation beyond simple clustering


#region Motivating GMM: Weaknesses of k-Means
# given simple, wellseparated data, k-means finds suitable clustering results

# e.g.
#  if we have simple blobs of data, 
#   the k-means algorithm can quickly label those clusters 
#   in a way that closely matches what we might do by eye

# Generate some data
# from sklearn.datasets._samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]   # flip axes for better plotting

# Plot the data with k-means labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis'); plt.show()


# From an intuitive standpoint, 
#   we might expect that the clustering assignment for
#   some points is more certain than others
# e.g.
#   there appears to be a very slight overlap between the two middle clusters, 
#   such that we might not have complete confidence in the cluster assignment of points between them

# Unfortunately, the k-means model 
#   has no intrinsic measure of probability or uncertainty of cluster assignments
#   (although it may be possible to use a bootstrap approach to estimate this uncertainty)
# For this, we must think about generalizing the model

# One way to think about the k-means model is that 
#   it places a circle (or, in higher dimensions, a hyper-sphere) 
#   at the center of each cluster, 
#   with a radius defined by the most distant point in the cluster
# This radius acts as a hard cutoff for cluster assignment within the training set: 
#   any point outside this circle 
#   is not considered a member of the cluster

# visualize this cluster model with the following function
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the k-means model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X); plt.show()


# important observation for k-means is that 
#   these cluster models must be circular: 
#   kmeans has no built-in way of accounting for oblong or elliptical clusters

# e.g.
# if we take the same data and transform it, the cluster assignments end up becoming muddled
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched); plt.show()


# By eye, we recognize that these transformed clusters are noncircular, 
#   and thus circular clusters would be a poor fit
# Nevertheless, k-means is not flexible enough to account for this, 
#   and tries to force-fit the data into four circular clusters
# This results in a mixing of cluster assignments 
#   where the resulting circles overlap: 
#   especially the bottom right of this plot

# might imagine addressing this particular situation
#   by preprocessing the data with PCA
# but in practice 
#   there is no guarantee that such a global operation 
#   will circularize the individual data

# two disadvantages of k-means—
#   its lack of flexibility in cluster shape and 
#   lack of probabilistic cluster assignment—
# mean that for many datasets 
#   (especially low-dimensional datasets) 
#   it may not perform as well as you might hope

# might imagine addressing these weaknesses by generalizing the k-means model
# e.g.
# could measure uncertainty in cluster assignment 
#   by comparing the distances of each point 
#   to all cluster centers, 
#   rather than focusing on just the closest
# might also imagine 
#   allowing the cluster boundaries to be ellipses 
#   rather than circles, 
#   so as to account for noncircular clusters
# turns out these are two essential components 
#   of a different type of clustering model, 
#   Gaussian mixture models


#endregion

#region Generalizing E–M: Gaussian Mixture Models
# Gaussian mixture model (GMM) 
#   attempts to find a mixture of multidimensional Gaussian probability distributions 
#   that best model any input dataset
# In the simplest case, 
#   GMMs can be used for finding clusters in the same manner as k-means

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis'); plt.show()


# But because GMM contains a probabilistic model under the hood, 
#   it is also possible to find probabilistic cluster assignments—
#   in Scikit-Learn we do this using the predict_proba method
#       This returns a matrix of size [n_samples, n_clusters] 
#       that measures the probability that 
#       any point belongs to the given cluster

probs = gmm.predict_proba(X)
print(probs[:5].round(3))

# can visualize this uncertainty by
# e.g.
#   making the size of each point proportional to the certainty of its prediction

size = 50 * probs.max(1) ** 2   # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis'); plt.show()
# precisely the points at the boundaries between clusters that reflect this uncertainty of cluster assignment


# Under the hood, a Gaussian mixture model is very similar to k-means: 
#   it uses an expectation–maximization approach that qualitatively does the following:
#   1. Choose starting guesses for the location and shape
#   2. Repeat until converged:
#       a. E-step: 
#           for each point, 
#           find weights encoding the probability of membership in each cluster
#       b. M-step: 
#           for each cluster, 
#           update its location, normalization, and shape based on all data points, making use of the weights

# result of this is that 
#   each cluster is associated not with a hard-edged sphere, 
#   but with a smooth Gaussian model
# Just as in the k-means expectation–maximization approach, 
#   this algorithm can sometimes miss the globally optimal solution, 
#   and thus in practice multiple random initializations are used


# function that will help us visualize the locations and shapes of the GMM clusters 
#   by drawing ellipses based on the gmm output
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    elif covariance.shape == ():
        angle = 0
        width = height = 2 * np.sqrt(covariance)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))



def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

# With this in place, we can take a look at what the four-component GMM gives us for our initial data
gmm = GaussianMixture(n_components=4, random_state=42)
plot_gmm(gmm, X); plt.show()

# we can use the GMM approach to fit our stretched dataset; 
#   allowing for a full covariance, the model will fit even very oblong, stretched-out clusters
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched); plt.show()

# This makes clear that GMMs address the two main practical issues with k-means encountered before.


#region Choosing the covariance type
# If you look at the details of the preceding fits, 
#   you will see that the covariance_type option was set differently within each
# This hyperparameter 
#   controls the degrees of freedom in the shape of each cluster; 
#   it is essential to set this carefully for any given problem
# covariance_type="diag", which means that
#   the size of the cluster 
#       along each dimension 
#       can be set independently, 
#   with the resulting ellipse 
#       constrained to align with the axes
#   default
# covariance_type="spherical", 
#   which constrains the shape of the cluster 
#       such that all dimensions are equal
#   resulting clustering will have similar characteristics to that of k-means, 
#       though it is not entirely equivalent
#   slightly simpler and faster model
# covariance_type="full", which 
#   allows each cluster to be modeled as an ellipse 
#       with arbitrary orientation
#   more complicated and computationally expensive model 
#       (especially as the number of dimensions grows)

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.05)

rng = np.random.RandomState(5)
X = np.dot(rng.randn(500, 2), rng.randn(2, 2))

for i, cov_type in enumerate(['diag', 'spherical', 'full']):
    model = GaussianMixture(1, covariance_type=cov_type).fit(X)
    # ax[i].axis('equal')
    ax[i].scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax[i].set_xlim(-3, 3)
    ax[i].set_title('covariance_type="{0}"'.format(cov_type),
                    size=14, family='monospace')
    draw_ellipse(model.means_[0], model.covariances_[0], ax[i], alpha=0.2)
    ax[i].xaxis.set_major_formatter(plt.NullFormatter())
    ax[i].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()


#endregion


#endregion

#region GMM as Density Estimation
# Though GMM is often categorized as a clustering algorithm, 
#   fundamentally it is an algorithm for density estimation
# result of a GMM fit to some data 
#   is technically not a clustering model, 
#   but a generative probabilistic model 
#       describing the distribution of the data


# e.g.
#   data generated from Scikit-Learn’s make_moons function
from sklearn.datasets import make_moons

Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]); plt.show()

# If we try to fit this to a two-component GMM viewed as a clustering model, 
#   the results are not particularly useful
gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
plot_gmm(gmm2, Xmoon); plt.show()

# But if we instead 
#   use many more components 
#   and ignore the cluster labels, 
# we find a fit that is much closer to the input data 
gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon, label=False); plt.show()


# Here the mixture of 16 Gaussians serves 
#   not to find separated clusters of data, 
#   but rather to model the overall distribution of the input data
# This is a generative model of the distribution, 
#   meaning that the GMM gives us the recipe to generate 
#   new random data distributed similarly to our input

# e.g. 400 new points drawn from this 16-component GMM fit to our original data
Xnew, ynew = gmm16.sample(400)
plt.scatter(Xnew[:, 0], Xnew[:, 1]); plt.show()

# GMM is convenient as a flexible means of modeling an arbitrary multidimensional distribution of data


#region How many components?
# fact that GMM is a generative model 
#   gives us a natural means of determining 
#   the optimal number of components for a given dataset
# generative model is 
#   inherently a probability distribution for the dataset, 
#   and so we can simply evaluate the likelihood of the data under the model, 
#   using cross-validation to avoid overfitting
# Another means of correcting for overfitting 
#   is to adjust the model likelihoods using some analytic criterion such as
#       the Akaike information criterion (AIC) or 
#       the Bayesian information criterion (BIC)
# Scikit-Learn’s GMM estimator actually includes built-in methods that compute both of these, 
#   and so it is very easy to operate on this approach


# look at the AIC and BIC as a function as the number of GMM components for our moon dataset
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Xmoon) for n in n_components]
plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()


# optimal number of clusters 
#   is the value that minimizes the AIC or BIC, 
#   depending on which approximation we wish to use
# AIC tells us that 
#   our choice of 16 components was probably too many: 
#   around 8–12 components would have been a better choice
# As is typical with this sort of problem, 
#   the BIC recommends a simpler model

# important point:
#   this choice of number of components 
#   measures how well GMM works as a density estimator, 
#   not how well it works as a clustering algorithm
# think of GMM primarily as a density estimator, 
#   and use it for clustering only when warranted within simple datasets


#endregion


#endregion

#region Example: GMM for Generating New Data
# generate new handwritten digits from the standard digits corpus that we have used before

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

# plot the first 100
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
plot_digits(digits.data); plt.show()

# have nearly 1,800 digits in 64 dimensions, 
#   and we can build a GMM on top of these to generate more
# GMMs can have difficulty converging in such a high dimensional space, 
#   so we will start with an invertible dimensionality reduction algorithm on the data
# will use a straightforward PCA, 
#   asking it to preserve 99% of the variance in the projected data

from sklearn.decomposition import PCA
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
data.shape

# result is 41 dimensions, a reduction of nearly 1/3 with almost no information loss
# Given this projected data, 
#   let’s use the AIC to get a gauge for the number of GMM components we should use
n_components = np.arange(50, 210, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics); plt.show()

# It appears that around 140 components minimizes the AIC; we will use this model.
#   fit this to the data and confirm that it has converged
gmm = GaussianMixture(140, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)


# draw samples of 100 new points within this 41-dimensional projected space, 
#   using the GMM as a generative model
data_new, ynew = gmm.sample(100)
data_new.shape

# use the inverse transform of the PCA object to construct the new digits
digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new); plt.show()

# results for the most part look like plausible digits from the dataset!


# Consider what we’ve done here: 
#   given a sampling of handwritten digits, 
#   we have modeled the distribution of that data in such a way that 
#   we can generate brand new samples of digits from the data: 
#       these are “handwritten digits” 
#       that do not individually appear in the original dataset, 
#       but rather capture the general features of the input data 
#       as modeled by the mixture model
# Such a generative model of digits can prove very useful 
#   as a component of a Bayesian generative classifier


#endregion
