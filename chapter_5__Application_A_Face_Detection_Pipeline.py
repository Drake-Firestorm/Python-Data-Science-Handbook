# CHAPTER 5: Application: A Face Detection Pipeline
# =================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from skimage import data, color, feature
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import skimage.data
from sklearn.datasets import fetch_lfw_people


# -------------------------------------------------------------

# chapter has explored a number of the central concepts and algorithms of machine learning
# But moving from these concepts to real-world application can be a challenge
# Real-world datasets 
#   are noisy and heterogeneous, 
#   may have missing features, 
#   and may include data in a form 
#       that is difficult to map 
#       to a clean [n_samples, n_features] matrix

# Before applying any of the methods discussed here, 
#   you must first extract these features from your data; 
#   there is no formula for how to do this 
#       that applies across all domains, 
#   and thus this is where you as a data scientist 
#       must exercise your own intuition and expertise

# One interesting and compelling application of machine learning is to images, 
# and we have already seen a few examples of this 
#   where pixel-level features are used for classification
# In the real world, 
#   data is rarely so uniform 
#   and simple pixels will not be suitable, 
# a fact that has led to 
#   a large literature on feature extraction methods for image data 

# In this section, we will take a look at one such feature extraction technique
# Histogram of Oriented Gradients (HOG)
#   which transforms image pixels 
#   into a vector representation 
#   that is sensitive to broadly informative image features 
#   regardless of confounding factors like illumination
# will use these features to develop a simple face detection pipeline, 
#   using machine learning algorithms and concepts 
#   we’ve seen throughout this chapter


#region HOG Features
# Histogram of Gradients 
#   is a straightforward feature extraction procedure 
#   that was developed in the context of identifying pedestrians within images

# HOG involves the following steps:
# 1. Optionally prenormalize images. 
#   This leads to features that resist dependence on variations in illumination.
# 2. Convolve the image with two filters 
#   that are sensitive to horizontal and vertical brightness gradients. 
#   These capture edge, contour, and texture information.
# 3. Subdivide the image into cells of a predetermined size, 
#   and compute a histogram of the gradient orientations within each cell
# 4. Normalize the histograms in each cell 
#   by comparing to the block of neighboring cells. 
#   This further suppresses the effect of illumination across the image.
# 5. Construct a one-dimensional feature vector 
#   from the information in each cell.


# A fast HOG extractor is built into the Scikit-Image project, 
#   and we can try it out relatively quickly 
#   and visualize the oriented gradients within each cell

# from skimage import data, color, feature
# import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)


fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')

ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features')
plt.show()


#endregion

#region HOG in Action: A Simple Face Detector
# Using these HOG features, 
#   we can build up a simple facial detection algorithm 
#   with any Scikit-Learn estimator
# here we will use a linear support vector machine

# steps are as follows:
# 1. Obtain a set of image thumbnails of faces to constitute “positive” training samples.
# 2. Obtain a set of image thumbnails of nonfaces to constitute “negative” training samples.
# 3. Extract HOG features from these training samples.
# 4. Train a linear SVM classifier on these samples.
# 5. For an “unknown” image, 
#   pass a sliding window across the image, 
#   using the model to evaluate whether that window contains a face or not.
# 6. If detections overlap, 
#   combine them into a single window.


# Let’s go through these steps and try it out:

# 1. Obtain a set of positive training samples.
# start by finding some positive training samples that show a variety of faces
#   one easy set of data to work with—the Labeled Faces in the Wild dataset
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape



# 2. Obtain a set of negative training samples.
# need a set of similarly sized thumbnails that do not have a face in them
# One way to do this 
#   is to take any corpus of input images, 
#   and extract thumbnails from them at a variety of scales
# can use some of the images shipped with Scikit-Image, 
#   along with Scikit-Learn’s PatchExtractor

from skimage import data, transform

# imgs_to_use = ['camera', 'text', 'coins', 'moon',
#                   'page', 'clock', 'immunohistochemistry',
#                   'chelsea', 'coffee', 'hubble_deep_field']
# print([(name, (getattr(data, name)()).shape) for name in imgs_to_use])
# already in grayscale ->  'camera', 'text', 'coins', 'moon' , 'page',  'clock', 
#   color.rgb2gray(data.camera())

imgs_to_use = ['immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]
imgs_to_use_grayscaled = ['camera', 'text', 'coins', 'moon' , 'page',  'clock']
images.append([(getattr(data, name)()) for name in imgs_to_use_grayscaled][0])



from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                                for im in images for scale in [0.5, 1.0, 2.0]])
negative_patches.shape


# take a look at a few of them to get an idea of what they look like
fig, ax = plt.subplots(6, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap='gray')
    axi.axis('off')
plt.show()


# Our hope is that these would sufficiently cover the space of “nonfaces” 
#   that our algorithm is likely to see.


# 3. Combine sets and extract HOG features.
# Now that we have these positive samples and negative samples, 
#   we can combine them and compute HOG features
# This step takes a little while, 
#   because the HOG features involve a nontrivial computation for each image

from itertools import chain
X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1
X_train.shape

# are left with 20,000 training samples in 1,215 dimensions, 
# and we now have our data in a form that we can feed into Scikit-Learn!


# 4. Train a support vector machine.
# Next we use the tools we have been exploring in this chapter 
#   to create a classifier of thumbnail patches
# For such a high-dimensional binary classification task, 
#   a linear support vector machine is a good choice
# will use Scikit-Learn’s LinearSVC, 
#   because in comparison to SVC 
#   it often has better scaling for large number of samples.


# First, though, let’s use a simple Gaussian naive Bayes to get a quick baseline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

cross_val_score(GaussianNB(), X_train, y_train)

# see that on our training data, 
#   even a simple naive Bayes algorithm 
#   gets us upward of 90% accuracy


# Let’s try the support vector machine, 
#   with a grid search over a few choices of the C parameter
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
grid.best_score_
grid.best_params_

# Let’s take the best estimator and retrain it on the full dataset:
model = grid.best_estimator_
model.fit(X_train, y_train)


# 5. Find faces in a new image.
# Now that we have this model in place, 
#   let’s grab a new image and see how the model does
# will use one portion of the astronaut image for simplicity
#   and run a sliding window over it and evaluate each patch

test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = skimage.transform.rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]

plt.imshow(test_image, cmap='gray')
plt.axis('off')
plt.show()


# Next, 
#   let’s create a window that iterates over patches of this image, 
#   and compute HOG features for each patch
def sliding_window(img, patch_size=positive_patches[0].shape, istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
patches_hog.shape


# Finally, 
#   we can take these HOG-featured patches 
#   and use our model to evaluate whether each patch contains a face
labels = model.predict(patches_hog)
labels.sum()

# see that out of nearly 2,000 patches, we have found 71 detections
# use the information we have about these patches 
#   to show where they lie on our test image, 
#   drawing them as rectangles 
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
plt.show()

# Note: The below is not true probably because of changes in the original training data set in skimage
# All of the detected patches overlap and found the face in the image! 
#   Not bad for a few lines of Python


#endregion

#region Caveats and Improvements
# If you dig a bit deeper into the preceding code and examples, 
#   you’ll see that we still have a bit of work 
#   before we can claim a production-ready face detector
# There are several issues with what we’ve done, 
#   and several improvements that could be made

# In particular:
# Our training set, especially for negative features, is not very complete
#   The central issue is that there are many face-like textures 
#   that are not in the training set, 
#   and so our current model is very prone to false positives
#   can see this if you try out the preceding algorithm on the full astronaut image: 
#   the current model leads to many false detections in other regions of the image
#   might imagine addressing this by 
#       adding a wider variety of images to the negative training set, 
#       and this would probably yield some improvement
#   Another way to address this 
#       is to use a more directed approach, 
#       such as hard negative mining
#           take a new set of images that our classifier has not seen, 
#           find all the patches representing false positives, 
#           and explicitly add them as negative instances in the training set 
#           before retraining the classifier
# Our current pipeline searches only at one scale
#   As currently written, 
#   our algorithm will miss faces that are not approximately 62×47 pixels
#   can straightforwardly address this 
#       by using sliding windows of a variety of sizes, 
#       and resizing each patch using skimage.transform.resize
#       before feeding it into the model
#   In fact, the sliding_window() utility used here
#       is already built with this in mind
# We should combine overlapped detection patches
#   For a production-ready pipeline, 
#   we would prefer not to have 30 detections of the same face, 
#   but to somehow reduce overlapping groups of detections down to a single detection
#   could be done 
#       via an unsupervised clustering approach
#           (MeanShift Clustering is one good candidate for this), 
#       or via a procedural approach 
#           such as nonmaximum suppression, 
#           an algorithm common in machine vision
# The pipeline should be streamlined
#   Once we address these issues, 
#   it would also be nice to create a more streamlined pipeline 
#   for ingesting training images 
#   and predicting sliding-window outputs
#   This is where Python as a data science tool really shines: 
#       with a bit of work, 
#       we could take our prototype code 
#       and package it with a well-designed object-oriented API 
#           that gives the user the ability to use this easily
# More recent advances, such as deep learning, should be considered
#   Finally, HOG and other procedural feature extraction methods for images 
#       are no longer state-of-the-art techniques
#   Instead, many modern object detection pipelines 
#       use variants of deep neural networks
#   One way to think of neural networks is that 
#       they are an estimator 
#       that determines optimal feature extraction strategies from the data, 
#       rather than relying on the intuition of the user


#endregion
