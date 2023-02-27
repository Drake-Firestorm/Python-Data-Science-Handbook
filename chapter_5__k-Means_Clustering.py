# CHAPTER 5: In Depth: k-Means Clustering
# =======================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE

from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from scipy.stats import mode

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from sklearn.datasets import load_sample_image

# -------------------------------------------------------------

# Clustering algorithms
#   seek to learn, from the properties of the data, 
#   an optimal division or discrete labeling of groups of points

# Many clustering algorithms are available in Scikit-Learn and elsewhere, 
# perhaps the simplest to understand is an algorithm known as k-means clustering, 
#   which is implemented in sklearn.cluster.KMeans


#region Introducing k-Means
# k-means algorithm 
#   searches for a predetermined number of clusters 
#   within an unlabeled multidimensional dataset

# accomplishes this using a simple conception of what the optimal clustering looks like:
# • The “cluster center” 
#       is the arithmetic mean of all the points belonging to the cluster.
# • Each point is closer to its own cluster center than to other cluster centers

# Those two assumptions are the basis of the k-means model

# e.g.
# generate a two-dimensional dataset containing four distinct blobs
# To emphasize that this is an unsupervised algorithm, leave the labels out of the visualization

# from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50); plt.show()

# By eye, it is relatively easy to pick out the four clusters
# k-means algorithm does this automatically
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# visualize the results by plotting the data colored by these labels
# also plot the cluster centers as determined by the k-means estimator
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# how this algorithm finds these clusters so quickly! 
# typical approach to k-means involves 
#   an intuitive iterative approach known as expectation–maximization


#endregion

#region k-Means Algorithm: Expectation–Maximization
# expectation–maximization approach consists of the following procedure:
# 1. Guess some cluster centers
# 2. Repeat until converged
#   a. E-Step: assign points to the nearest cluster center
#   b. M-Step: set the cluster centers to the mean

# “E-step” or “Expectation step”
#   involves updating our expectation of which cluster each point belongs to

# “M-step” or “Maximization step” 
#   involves maximizing some fitness function that defines the location of the cluster centers—
#   in this case, that maximization is accomplished by taking a simple mean of the data in each cluster

# under typical circumstances, 
#   each repetition of the E-step and M-step 
#   will always result in a better estimate of the cluster characteristics

# k-means algorithm is simple enough that we can write it in a few lines of code.
# The following is a very basic implementation
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis'); plt.show()

# Most well-tested implementations will do a bit more than this under the hood, 
#   but the preceding function gives the gist of the expectation–maximization approach


#region Caveats of expectation–maximization
# few issues to be aware of when using the expectation–maximization algorithm

# ---------------------------

# 1. 
# globally optimal result may not be achieved
#   although the E–M procedure is guaranteed to improve the result in each step, 
#   there is no assurance that it will lead to the global best solution

# e.g.
# if we use a different random seed in our simple procedure, 
#   the particular starting guesses lead to poor results
centers, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis'); plt.show()

# Here the E–M approach has converged, 
#   but has not converged to a globally optimal configuration

# For this reason, it is common for the algorithm to be run for multiple starting guesses, 
#   as indeed Scikit-Learn does by default 
#   (set by the n_init parameter, which defaults to 10)

# ---------------------------

# 2,
# The number of clusters must be selected beforehand
#   must tell it how many clusters you expect: 
#   it cannot learn the number of clusters from the data

# e.g.
# if we ask the algorithm to identify six clusters, 
#   it will happily proceed and find the best six clusters
labels = KMeans(6, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis'); plt.show()

# Whether the result is meaningful is a question that is difficult to answer definitively

# one approach that is rather intuitive is called silhouette analysis

# Alternatively, you might use a more complicated clustering algorithm 
#   which has a better quantitative measure of the fitness per number of clusters 
#       (e.g., Gaussian mixture models)
#   or which can choose a suitable number of clusters 
#       (e.g., DBSCAN, mean-shift, or affinity propagation, 
#       all available in the sklearn.cluster submodule)

# ---------------------------

# 3.
# k-means is limited to linear cluster boundaries
# The fundamental model assumptions of k-means 
#   (points will be closer to their own cluster center than to others) 
# means that the algorithm will often be ineffective 
#   if the clusters have complicated geometries


# e.g.
# following data, along with the cluster labels found by the typical k-means approach
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)

labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis'); plt.show()


# use a kernel transformation 
#   to project the data into a higher dimension 
#   where a linear separation is possible

# One version of this kernelized k-means is implemented in Scikit-Learn within the
#   SpectralClustering estimator
# uses the graph of nearest neighbors to compute a higher-dimensional representation of the data, 
#   and then assigns labels using a k-means algorithm

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis'); plt.show()


# see that with this kernel transform approach, 
#   the kernelized k-means is able to find the more complicated nonlinear boundaries between clusters

# ---------------------------

# 4.
# k-means can be slow for large numbers of samples
# Because each iteration of k-means must access every point in the dataset, 
#   the algorithm can be relatively slow as the number of samples grows

# might wonder if this requirement to use all data at each iteration can be relaxed
# e.g.
#   might just use a subset of the data to update the cluster centers at each step

# This is the idea behind batch-based k-means algorithms, 
#   one form of which is implemented in sklearn.cluster.MiniBatchKMeans
# interface for this is the same as for standard KMeans


#endregion


#endregion

#region Examples
# Being careful about these limitations of the algorithm, 
#   can use k-means to our advantage in a wide variety of situations


#region Example 1: k-Means on digits
# will attempt to use k-means to try to identify similar digits 
# without using the original label information; 
#   this might be similar to a first step 
#   in extracting meaning from a new dataset 
#   about which you don’t have any a priori label information

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

# result is 10 clusters in 64 dimensions
# cluster centers themselves are 64-dimensional points, 
#   and can themselves be interpreted as the “typical” digit within the cluster

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()

# even without the labels, 
#   KMeans is able to find clusters whose centers are recognizable digits, 
#   with perhaps the exception of 1 and 8

# Because k-means knows nothing about the identity of the cluster, 
#   the 0–9 labels may be permuted
# can fix this 
#   by matching each learned cluster label 
#   with the true labels found in them

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
    
# check how accurate our unsupervised clustering was in finding similar digits within the data

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

# With just a simple k-means algorithm, 
#   we discovered the correct grouping for 80% of the input digits!
# check the confusion matrix for this 
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# As we might expect from the cluster centers we visualized before, 
#   the main point of confusion is between the eights and ones
# But this still shows that using k-means, 
#   we can essentially build a digit classifier 
#   without reference to any known labels!

# can use the t-distributed stochastic neighbor embedding (t-SNE) algorithm
#   to preprocess the data before performing k-means
# t-SNE is a non‐linear embedding algorithm 
#   that is particularly adept at preserving points within clusters
from sklearn.manifold import TSNE

# Project the data: this step will take several seconds
tsne =TSNE(n_components=2, init='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# Compute the accuracy
accuracy_score(digits.target, labels)


# That’s nearly 94% classification accuracy without using the labels
# This is the power of unsupervised learning when used carefully: 
#   it can extract information from the data‐set 
#   that it might be difficult to do by hand or by eye


#endregion

#region Example 2: k-means for color compression
# One interesting application of clustering is in color compression within images

# e.g.
# image with millions of colors. 
# In most images, 
#   a large number of the colors will be unused, 
#   and many of the pixels in the image will have similar or even identical colors

# e.g.
# Note: this requires the pillow package to be installed
from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
plt.show()

# image itself is stored in a three-dimensional array of size (height, width, RGB),
#   containing red/blue/green contributions as integers from 0 to 255
china.shape

# One way we can view this set of pixels 
#   is as a cloud of points in a three-dimensional color space
# will reshape the data to [n_samples x n_features], 
#   and rescale the colors so that they lie between 0 and 1

data = china / 255.0    # use 0...1 scale
data = data.reshape(427 * 640, 3)
data.shape

# visualize these pixels in this color space, 
#   using a subset of 10,000 pixels for efficiency
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color = colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    
    fig.suptitle(title, size=20)

plot_pixels(data, title='Input color space: 16 million possible colors'); plt.show()


# let’s reduce these 16 million colors to just 16 colors, 
#   using a k-means clustering across the pixel space
# Because we are dealing with a very large dataset, 
#   we will use the mini batch k-means, 
#       which operates on subsets of the data 
#       to compute the result much more quickly than the standard k-means algorithm

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors'); plt.show()

# result is a recoloring of the original pixels, 
#   where each pixel is assigned the color of its closest cluster center
# Plotting these new colors in the image space 
#   rather than the pixel space 
#   shows us the effect of this
china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
plt.show()


# Some detail is certainly lost in the rightmost panel, 
#   but the overall image is still easily recognizable
# This image on the right achieves a compression factor of around 1 million!
# While this is an interesting application of k-means, 
#   there are certainly better way to compress information in images


#endregion


#endregion
