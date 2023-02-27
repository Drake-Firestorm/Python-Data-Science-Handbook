# CHAPTER 5: In-Depth: In Depth: Principal Component Analysis
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.decomposition import PCA

from sklearn.datasets import load_digits

# -------------------------------------------------------

# principal component analysis (PCA)
#   one of the most broadly used of unsupervised algorithms
# is fundamentally a dimensionality reduction algorithm, 
#   but it can also be useful as a tool 
#       for visualization, 
#       for noise filtering, 
#       for feature extraction and engineering, 
#       and much more


#region Introducing Principal Component Analysis
# Principal component analysis 
#   is a fast and flexible unsupervised method for dimensionality reduction in data

# behavior is easiest to visualize by looking at a two-dimensional dataset
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.show()


# By eye, it is clear that there is a nearly linear relationship between the x and y variables
# but the problem setting here is slightly different: 
#   rather than attempting to predict the y values from the x values, 
#   the unsupervised learning problem attempts to learn about the relationship between the x and y values

# In principal component analysis, one quantifies this relationship 
#   by finding a list of the principal axes in the data, 
#   and using those axes to describe the dataset

# Using Scikit-Learn’s PCA estimator, we can compute this
from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(X)

# The fit learns some quantities from the data, most importantly the 
#   “components” and
#   “explained variance”
print(pca.components_); print(pca.explained_variance_)

# To see what these numbers mean, 
#   visualize them as vectors over the input data,
#   using the “components” to define the direction of the vector, and 
#   the “explained variance” to define the squared-length of the vector
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, color='black')
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()


# These vectors represent the principal axes of the data, 
#   and the length shown is an indication of 
#       how “important” that axis is in describing the distribution of the data—
#       more precisely, it is a measure of the variance of the data when projected onto that axis

# projection of each data point onto the principal axes are the “principal components” of the data


# If we plot these principal components beside the original data, we see 
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

# plot data
ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
ax[0].axis('equal')
ax[0].set(xlabels='x', ylabel='y', title='input')

# plot principal components
X_pca = pca.transform(X)
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax[1])
draw_vector([0, 0], [3, 0], ax=ax[1])
ax[1].axis('equal')
ax[1].set(xlabels='component 1', ylabel='component 2', title='principal component'
            , xlim=(-5, 5), ylim=(-3, 3.1))

plt.show()


# This transformation from data axes to principal axes is as an affine transformation,
#   which basically means it is composed of a translation, rotation, and uniform scaling

# While this algorithm to find principal components may seem like just a mathematical curiosity, 
#   it turns out to have very far-reaching applications in the world of machine learning and data exploration


#region PCA as dimensionality reduction
# Using PCA for dimensionality reduction involves
#   zeroing out one or more of the smallest principal components, 
#   resulting in a lower-dimensional projection of the data 
#   that preserves the maximal data variance

# e.g.
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape: ", X.shape)
print("transformed shape: ", X_pca.shape)

# transformed data has been reduced to a single dimension
# To understand the effect of this dimensionality reduction, 
#   we can perform the inverse transform of this reduced data 
#   and plot it along with the original data
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()

# light points are the original data, 
#   while the dark points are the projected version
# This makes clear what a PCA dimensionality reduction means: 
#   the information along the least important principal axis or axes is removed, 
#   leaving only the component(s) of the data with the highest variance
# fraction of variance that is cut out 
#       (proportional to the spread of points about the line formed above)
#   is roughly a measure of how much “information” is discarded in this reduction of dimensionality

# This reduced-dimension dataset is in some senses “good enough” 
#   to encode the most important relationships between the points: 
#       despite reducing the dimension of the data by 50%, 
#       the overall relationship between the data points is mostly preserved


#endregion

#region PCA for visualization: Handwritten digits
# usefulness of the dimensionality reduction may not be entirely apparent in only two dimensions, 
#   but becomes much more clear when we look at high-dimensional data

# e.g. application of PCA to the digits data 
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

# data consists of 8×8 pixel images, meaning that they are 64-dimensional
# To gain some intuition into the relationships between these points, 
#   we can use PCA to project them to a more manageable number of dimensions, 
#       say two
pca = PCA(2)    # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

#  plot the first two principal components of each point to learn about the data
plt.scatter(projected[:, 0], projected[:, 1]
            , c=digits.target, edgecolors='none', alpha=0.5
            , cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

# what these components mean: 
#   the full data is a 64-dimensional point cloud,
#   and these points are the projection of each data point along the directions with the largest variance
# Essentially, we have found the optimal stretch and rotation in 64-dimensional space 
#   that allows us to see the layout of the digits in two dimensions, 
#   and have done this in an unsupervised manner—
#       that is, without reference to the labels


#endregion

#region What do the components mean?
# meaning can be understood in terms of combinations of basis vectors

# e.g.
# each image in the training set is defined by a collection of 64 pixel values,
#   which we will call the vector x
# x = [x1, x2, x3⋯x64]

# One way we can think about this is in terms of a pixel basis
# That is, to construct the image, 
#   we multiply each element of the vector by the pixel it describes, 
#   and then add the results together to build the image
# image(x) = x1 · (pixel 1) + x2 · (pixel 2) + x3 · (pixel 3) ⋯x64 · (pixel 64)

# One way we might imagine reducing the dimension of this data is 
#   to zero out all but a few of these basis vectors
# e.g.
#   if we use only the first eight pixels, 
#   we get an eight-dimensional projection of the data
#   but it is not very reflective of the whole image: 
#       we’ve thrown out nearly 90% of the pixels!
# Using only eight of the pixel-basis components, 
#   we can only construct a small portion of the 64-pixel image
# Were we to continue this sequence and use all 64 pixels, we would recover the original image


# But the pixel-wise representation is not the only choice of basis
# can also use other basis functions, 
#   which each contain some predefined contribution from each pixel,
#   and write something like
# image(x) = mean + x1 · (basis 1) + x2 · (basis 2) + x3 · (basis 3) ⋯

# PCA can be thought of as a process of choosing optimal basis functions, 
#   such that adding together just the first few of them 
#   is enough to suitably reconstruct the bulk of the elements in the dataset
# principal components, 
#   which act as the low-dimensional representation of our data, 
#   are simply the coefficients 
#   that multiply each of the elements in this series


# Unlike the pixel basis, 
#   the PCA basis allows us to recover the salient features of the input image 
#   with just a mean plus eight components!
# amount of each pixel in each component 
#   is the corollary of the orientation of the vector in our two-dimensional example

# This is the sense in which PCA provides a low-dimensional representation of the data: 
#   it discovers a set of basis functions   
#   that are more efficient than the native pixel-basis of the input data


#endregion

#region Choosing the number of components
# vital part of using PCA in practice 
#   is the ability to estimate 
#   how many components are needed to describe the data

# can determine this by looking at the 
#   cumulative explained variance ratio 
#       as a function of the number of components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# This curve quantifies how much of the total, 64-dimensional variance 
#   is contained within the first N components
# e.g.
#   see that with the digits the first 10 components 
#   contain approximately 75% of the variance, 
#   while you need around 50 components to describe close to 100% of the variance

# Here we see that our two-dimensional projection loses a lot of information 
#   (as measured by the explained variance) 
#   and that we’d need about 20 components to retain 90% of the variance

# Looking at this plot for a high-dimensional dataset 
#   can help you understand the level of redundancy present in multiple observations
# or
np.arange(64)[np.cumsum(pca.explained_variance_ratio_) <= 0.9]


#endregion


#endregion

#region PCA as Noise Filtering
# idea is this: 
#   any components with variance much larger than the effect of the noise 
#   should be relatively unaffected by the noise
# So if you reconstruct the data using just the largest subset of principal components, 
#   you should be preferentially keeping the signal and throwing out the noise

# looks with the digits data
def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4)
                                , subplot_kw={'xticks':[], 'yticks':[]}
                                , gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8)
                    , cmap='binary', interpolation='nearest', clim=(0, 16))
plot_digits(digits.data); plt.show()


# add some random noise to create a noisy dataset
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy); plt.show()

# train a PCA on the noisy data, requesting that the projection preserve 50% of the variance
pca = PCA(0.5).fit(noisy)
pca.n_components_

# Here 50% of the variance amounts to 12 principal components

# compute these components, 
#   and then use the inverse of the transform to reconstruct the filtered digits
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered); plt.show()

# This signal preserving/noise filtering property makes PCA a very useful feature selection routine—
# for example, 
#   rather than training a classifier on very high-dimensional data, 
#   you might instead train the classifier on the lower-dimensional representation,
#   which will automatically serve to filter out random noise in the inputs


#endregion

#region Example: Eigenfaces
# Earlier we explored an example of using a PCA projection 
#   as a feature selector for facial recognition with a support vector machine
# Here we will take a look back and explore a bit more of what went into that

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# take a look at the principal axes that span this dataset
# Because this is a large dataset, we will use RandomizedPCA—
#   it contains a randomized method 
#   to approximate the first N principal components 
#   much more quickly than the standard PCA estimator, 
#   and thus is very useful for high-dimensional data 
# (here, a dimensionality of nearly 3,000)

# take a look at the first 150 components
pca = PCA(150, svd_solver='randomized')
pca.fit(faces.data)

# can be interesting to visualize the images associated 
#   with the first several principal components 
#   (these components are technically known as “eigenvectors,” 
#   so these types of images are often called “eigenfaces”)

fig, axes = plt.subplots(3, 8, figsize=(9, 4)
                            , subplot_kw={'xticks':[], 'yticks':[]}
                            , gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
plt.show()


# results are very interesting, and give us insight into how the images vary
# e.g.
#   first few eigenfaces (from the top left) 
#       seem to be associated with the angle of lighting on the face, 
#   and later principal vectors 
#       seem to be picking out certain features, such as eyes, noses, and lips

# look at the cumulative variance of these components 
#   to see how much of the data information the projection is preserving
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cummulative explained variance')
plt.show()

# these 150 components account for just over 90% of the variance
# That would lead us to believe that using these 150 components, 
#   we would recover most of the essential characteristics of the data

# To make this more concrete, 
#   we can compare the input images 
#   with the images reconstructed from these 150 components

# Compute the components and projected faces
pca = PCA(150, svd_solver='randomized').fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

# Plot the results
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5)
                        , subplot_kw={'xticks':[], 'yticks':[]}
                        , gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction')
plt.show()


# This visualization makes clear why the PCA feature selection used in SVM was so successful: 
#   although it reduces the dimensionality of the data by nearly a factor of 20, 
#   the projected images contain enough information 
#   that we might, by eye, recognize the individuals in the image

# What this means is that 
#   our classification algorithm needs to be trained on 150-dimensional data 
#   rather than 3,000-dimensional data, 
#   which depending on the particular algorithm we choose, 
#   can lead to a much more efficient classification


#endregion

#region Principal Component Analysis Summary
# Because of the versatility and interpretability of PCA, 
#   it has been shown to be effective in a wide variety of contexts and disciplines

# Given any high-dimensional dataset, 
#   tend to start with PCA 
#   in order to 
#       visualize the relationship between points 
#           (as we did with the digits), 
#       to understand the main variance in the data 
#           (as we did with the eigenfaces), 
#       and to understand the intrinsic dimensionality 
#           (by plotting the explained variance ratio)

# Certainly PCA is not useful for every high-dimensional dataset, 
#   but it offers a straightforward and efficient path 
#   to gaining insight into high-dimensional data

# PCA’s main weakness is that it tends to be highly affected by outliers in the data
# For this reason, 
#   many robust variants of PCA have been developed, 
#   many of which act to iteratively discard data points 
#       that are poorly described by the initial components
# While PCA is flexible, fast, and easily interpretable, 
#   it does not perform so well when there are nonlinear relationships within the data

# Scikit-Learn contains a couple interesting variants on PCA, including 
#   RandomizedPCA
#       uses a nondeterministic method 
#       to quickly approximate the first few principal components 
#       in very high-dimensional data
#   SparsePCA
#       introduces a regularization term
#       that serves to enforce sparsity of the components


#endregion
