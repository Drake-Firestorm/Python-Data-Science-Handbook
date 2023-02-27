# CHAPTER 5: In-Depth: Support Vector Machines
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_circles
from sklearn.datasets import fetch_lfw_people

from mpl_toolkits import mplot3d
import timeit

# ----------------------------------------------------------------

# Support vector machines (SVMs) 
#   are a particularly powerful and flexible class of supervised algorithms 
#   for both classification and regression


#region Motivating Support Vector Machines
# is a discriminative classiication: 
#   rather than modeling each class, 
#   we simply find a line or curve (in two dimensions) or manifold (in multiple dimensions)
#       that divides the classes from each other


#e.g.
# from sklearn.datasets._samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn'); plt.show()


# linear discriminative classifier would attempt to draw a straight line separating the two sets of data, 
#   and thereby create a model for classification
# For two-dimensional data like that shown here, this is a task we could do by hand
# But immediately we see a problem: 
#   there is more than one possible dividing line that can perfectly discriminate between the two classes!

# can draw them as follows
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)
plt.show()


# These are three very different separators that, nevertheless, perfectly discriminate between these samples
#   Depending on which you choose, a new data point (e.g., the one marked by the “X”) will be assigned a different label! 
# Evidently our simple intuition of “drawing a line between classes” is not enough, and we need to think a bit deeper.


#endregion

#region Support Vector Machines: Maximizing the Margin
# Support vector machines offer one way to improve on this
# The intuition is this:
#   rather than simply drawing a zero-width line between the classes, 
#   we can draw around each line a margin of some width, up to the nearest point

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()


# In support vector machines, 
#   the line that maximizes this margin is the one we will choose as the optimal model
# Support vector machines are an example of such a maximum margin estimator


#region Fitting a support vector machine
# Let’s see the result of an actual fit to this data: 
#   we will use Scikit-Learn’s support vector classifier to train an SVM model on this data
# For the time being, we will use a linear kernel and set the C parameter to a very large number

# from sklearn.svm import SVC     # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)


# To better visualize what’s happening here, 
# let’s create a quick convenience function that will plot SVM decision boundaries for us
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a two-dimensional SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none', edgecolors='black')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()


# This is the dividing line that maximizes the margin between the two sets of points
# few of the training points just touch the margin
#       they are indicated by the black circles 
#   These points are the pivotal elements of this fit, 
#   and are known as the support vectors, 
#   and give the algorithm its name

# In Scikit-Learn, the identity of these points is stored in the support_vectors_ attribute of the classifier
model.support_vectors_


# A key to this classifier’s success is that for the fit, only the position of the support vectors matters; 
#   any points further from the margin that are on the correct side do not modify the fit!
# Technically, this is because these points do not contribute to the loss function used to fit the model, 
#   so their position and number do not matter so long as they do not cross the margin

# e.g. if we plot the model learned from the first 60 points and first 120 points of this dataset
def plot_svm(N=10,  ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))
plt.show()


# In the left panel, we see the model and the support vectors for 60 training points
# In the right panel, we have doubled the number of training points, but the model has not changed: 
#   the three support vectors from the left panel are still the support vectors from the right panel
# This insensitivity to the exact behavior of distant points is one of the strengths of the SVM model


#endregion

#region Beyond linear boundaries: Kernel SVM
#   https://www.quora.com/What-are-kernels-in-machine-learning-and-SVM-and-why-do-we-need-them/answer/Lili-Jiang?srid=oOgT
#   https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d

# Where SVM becomes extremely powerful is when it is combined with kernels

# seen a version of kernels before, in the basis function regressions of In Depth: Linear Regression
# There we projected our data into higher-dimensional space defined by polynomials and Gaussian basis functions, 
#   and thereby were able to fit for nonlinear relationships with a linear classifier

# In SVM models, we can use a version of the same idea

# let’s look at some data that is not linearly separable
from sklearn.datasets._samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False); plt.show()

# think about how we might project the data into a higher dimension such that a linear separator would be sufficient
#   e.g. one simple projection we could use would be to 
#       compute a radial basis function centered on the middle clump
r = np.exp(-(X ** 2).sum(1))

# can visualize this extra data dimension using a three-dimensional plot
from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plot_3D(); plt.show()

# We can see that with this additional dimension, 
#   the data becomes trivially linearly separable, 
#   by drawing a separating plane at, say, r=0.7

# had to choose and carefully tune our projection; 
#   if we had not centered our radial basis function in the right location, 
#   we would not have seen such clean, linearly separable results
# In general, the need to make such a choice is a problem: 
#   we would like to somehow automatically find the best basis functions to use

# One strategy to this end is to compute a basis function centered at every point in the dataset, 
#   and let the SVM algorithm sift through the results
# This type of basis function transformation is known as a kernel transformation, 
#   as it is based on a similarity relationship (or kernel) between each pair of points

# potential problem with this strategy—projecting N points into N dimensions—is that 
#   it might become very computationally intensive as N grows large
# However, because of a neat little procedure known as kernal trick,
#   a fit on kernel-transformed data can be done implicitly—
#   that is, without ever building the full N-dimensional representation of the kernel projection!
#   It allows us to operate in the original feature space 
#   without computing the coordinates of the data in a higher dimensional space.
# This kernel trick is built into the SVM, 
#   and is one of the reasons the method is so powerful

# In Scikit-Learn, we can apply kernelized SVM simply 
#   by changing our linear kernel 
#   to an RBF (radial basis function) kernel, 
#   using the kernel model hyperparameter
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=300, lw=1, facecolors='none', edgecolors='black')
plt.show()


# Using this kernelized support vector machine, we learn a suitable nonlinear decision boundary
# This kernel transformation strategy is used often in machine learning 
#   to turn fast linear methods into fast nonlinear methods, 
#   especially for models in which the kernel trick can be used


#endregion

#region Tuning the SVM: Softening margins
# discussion so far has centered on very clean datasets, in which a perfect decision boundary exists
# what if your data has some amount of overlap? 

# e.g.
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()

# To handle this case, the SVM implementation has a bit of a fudge-factor that “softens” the margin; 
# that is, it allows some of the points to creep into the margin if that allows a better fit
# hardness of the margin is controlled by a tuning parameter, 
#   most often known as C
# For very large C, the margin is hard, and points cannot lie in it
# For smaller C, the margin is softer, and can grow to encompass some points

# gives a visual picture of how a changing C parameter affects the final fit, via the softening of the margin
X, y = make_blobs(n_samples=100, center=2, random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1]
                , s=300, lw=1, facecolors='none', edgecolor='black')
    axi.set_title('C: {0:.1f}'.format(C), size=14)
plt.show()

# optimal value of the C parameter will depend on your dataset, 
#   and should be tuned via cross-validation or a similar procedure


#endregion


#endregion

#region Example: Face Recognition
# will use the Labeled Faces in the Wild dataset, 
#   which consists of several thousand collated photos of various public figures
# fetcher for the dataset is built into Scikit-Learn

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# plot a few of these faces to see what we’re working with
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show()

# Each image contains [62×47] or nearly 3,000 pixels
# could proceed by simply using each pixel value as a feature, 
#   but often it is more effective to use some sort of preprocessor 
#   to extract more meaningful features; 
# here we will use a principal component analysis
#   to extract 150 fundamental components to feed into our support vector machine classifier

# can do this most straightforwardly by packaging the preprocessor and the classifier into a single pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=150, whiten=True, random_state=42, svd_solver='randomized')
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# For the sake of testing our classifier output, 
#   we will split the data into a training and testing set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

# use a grid search cross-validation to explore combinations of parameters
# adjust 
#   C (which controls the margin hardness) and 
#   gamma (which controls the size of the radial basis function kernel), 
# and determine the best model
from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

timeit.timeit("grid.fit(Xtrain, ytrain)", globals=globals(), number=1)
print(grid.best_params_)

# optimal values fall toward the middle of our grid; 
#   if they fell at the edges, 
#   we would want to expand the grid to make sure we have found the true optimum

# predict the labels for the test data, which the model has not yet seen
model = grid.best_estimator_
yfit = model.predict(Xtest)

# look at a few of the test images along with their predicted values
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()


#  can get a better sense of our estimator’s performance using the classification report, 
#   which lists recovery statistics label by label
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit, target_names=faces.target_names))


# might also display the confusion matrix between these classes
#   This helps us get a sense of which labels are likely to be confused by the estimator
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
            , xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('prediceted label')
plt.show()


# For a real-world facial recognition task, 
#   in which the photos do not come precropped into nice grids, 
#   the only difference in the facial classification scheme is the feature selection: 
#       you would need to use a more sophisticated algorithm to find the faces, 
#       and extract features that are independent of the pixellation
# For this kind of application, one good option is to make use of OpenCV,
#   which among other things, includes pre‐trained implementations of state-of-the-art feature extraction tools 
#   for images in general and faces in particular


#endregion

#region Support Vector Machine Summary
# These methods are a powerful classification method for a number of reasons:
# • Their dependence on relatively few support vectors means that
#       they are very compact models, and take up very little memory.
# • Once the model is trained, the prediction phase is very fast.
# • Because they are affected only by points near the margin,
#       they work well with high-dimensional data—
#       even data with more dimensions than samples,
#       which is a challenging regime for other algorithms.
# • Their integration with kernel methods makes them very versatile,
#       able to adapt to many types of data

# SVMs have several disadvantages as well:
# • The scaling with the number of samples N is 
#           O[N**3] at worst, or 
#           O[N**2] for efficient implementations. 
#       For large numbers of training samples, this computational cost can be prohibitive.
# • The results are strongly dependent on a suitable choice for the softening parameter C. 
#       This must be carefully chosen via cross-validation, 
#       which can be expensive as datasets grow in size.
# • The results do not have a direct probabilistic interpretation.
#       This can be estimated via an internal cross-validation 
#           (see the probability parameter of SVC), 
#       but this extra estimation is costly

# With those traits in mind, I generally only turn to SVMs once other simpler, faster,
#   and less tuning-intensive methods have been shown to be insufficient for my needs
# Nevertheless, if you have the CPU cycles to commit to training and cross-validating an SVM on your data, 
#   the method can lead to excellent results


#endregion
