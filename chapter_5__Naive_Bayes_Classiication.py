# CHAPTER 5: In Depth: Naive Bayes Classiication
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # plot formatting

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix

from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets import fetch_20newsgroups

# -------------------------------------------------------------

# Naive Bayes models 
#   are a group of extremely fast and simple classification algorithms
#   that are often suitable for very high-dimensional datasets

# Because they 
#   are so fast
#   and have so few tunable parameters, 
# they end up being very useful as a quick-and-dirty baseline for a classification problem

# is an example of generative classiication
#   simple model describing the distribution of each underlying class, 
#   and used these generative models to probabilistically determine labels for new points


#region Bayesian Classiication
# Naive Bayes classifiers are built on Bayesian classification methods
# rely on Bayes’s theorem, 
#   which is an equation describing 
#   the relationship of conditional probabilities of statistical quantities

# In Bayesian classification, 
# we’re interested in finding 
#   the probability of a label 
#   given some observed features, 
# which we can write as P(L | features).

# Bayes’s theorem tells us how to express this in terms of quantities we can compute more directly
# P(L | features) = ( P(features | L) * P(L) ) / P(features)


# If we are trying to decide between two labels—
#   let’s call them L1 and L2—
# then one way to make this decision 
#   is to compute the ratio of the posterior probabilities for each label

# P(L1 | features) = P(features | L1) P(L1)
# ________________   ________________ _____
# P(L2 | features) = P(features | L2) P(L2)

# All we need now is some model by which we can compute P(features | Li) for each label
# Such a model is called a generative model 
#   because it specifies the hypothetical random process that generates the data

# Specifying this generative model for each label is the main piece of the training of such a Bayesian classifier
# general version of such a training step is a very difficult task, 
# but we can make it simpler through the use of some simplifying assumptions about the form of this model

# This is where the “naive” in “naive Bayes” comes in: 
#   if we make very naive assumptions about the generative model for each label, 
#   we can find a rough approximation of the generative model for each class, 
#   and then proceed with the Bayesian classification
# Different types of naive Bayes classifiers rest on different naive assumptions about the data


#endregion

#region Gaussian Naive Bayes
# Perhaps the easiest naive Bayes classifier to understand is Gaussian naive Bayes
# In this classifier, the assumption is that 
#   data from each label 
#   is drawn from a simple Gaussian distribution

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()


# One extremely fast way to create a simple model is to assume that the 
#   data is described by a Gaussian distribution 
#   with no covariance between dimensions

# can fit this model 
# by simply finding 
#   the mean and standard deviation of the points 
# within each label, 
# which is all you need to define such a distribution

# result of this naive Gaussian assumption
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size=14)

xlim = (-8, 8)
ylim = (-15, 5)

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
    
ax.set(xlim=xlim, ylim=ylim)

plt.show()

# ellipses here represent the Gaussian generative model for each label, 
#   with larger probability toward the center of the ellipses
# With this generative model in place for each class, 
#   we have a simple recipe to compute the likelihood P(features | L1) for any data point, 
#   and thus we can quickly compute the posterior ratio 
#   and determine which label is the most probable for a given point


# This procedure is implemented in Scikit-Learn’s sklearn.naive_bayes.GaussianNB estimator
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

# generate some new data and predict the label
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew  = model.predict(Xnew)

# plot this new data to get an idea of where the decision boundary is
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()


# see a slightly curved boundary in the classifications—
#   in general, the boundary in Gaussian naive Bayes is quadratic

# nice piece of this Bayesian formalism is that it naturally allows for probabilistic classification, 
#   which we can compute using the predict_proba method
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)

# columns give the posterior probabilities of the first and second label, respectively
# If you are looking for estimates of uncertainty in your classification, 
#   Bayesian approaches like this can be a useful approach


# Of course, 
#   the final classification will only be as good as the model assumptions that lead to it, 
#   which is why Gaussian naive Bayes often does not produce very good results
# Still, in many cases—
#       especially as the number of features becomes large—
#   this assumption is not detrimental enough to prevent Gaussian naive Bayes from being a useful method


#endregion

#region Multinomial Naive Bayes
# multinomial naive Bayes, 
#   where the features are assumed to be generated from a simple multinomial distribution

# multinomial distribution 
#   describes the probability of observing counts among a number of categories, and thus
# multinomial naive Bayes 
#   is most appropriate for features that represent counts or count rates


#region Example: Classifying text
# One place where multinomial naive Bayes is often used is in text classification, 
# where the features are related to word counts or frequencies within the documents to be classified

# e.g.
# will use the sparse word count features from the 20 Newsgroups corpus 
# to show how we might classify these short documents into categories
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names
print(data.data[1]); print('\n\n\n'); print(data.target_names[1])

# For simplicity, we will select just a few of these categories, and download the training and testing set
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

[(categories, train.target_names.index(categories)) for categories in categories] # index of each category
print(train.data[5])    # print gives a readable format as compared to direct execution
print(train.target[5])

# In order to use this data for machine learning, 
#   we need to be able to convert the content of each string into a vector of numbers
# will use the TF–IDF vectorizer
# and create a pipeline that attaches it to a multinomial naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# apply the model to the training data, and predict labels for the test data
model.fit(train.data, train.target)
labels = model.predict(test.data)

# evaluate them to learn about the performance of the estimator
# e.g.
# confusion matrix between the true and predicted labels for the test data
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# Evidently, even this very simple classifier can successfully separate space talk from computer talk, 
# but it gets confused between talk about religion and talk about Christianity. 
# This is perhaps an expected area of confusion!


# very cool thing here is that we now have the tools to determine the category for any string, 
# using the predict() method of this pipeline

# quick utility function that will return the prediction for a single string
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

# try it out
predict_category('sending a payload to the ISS')
predict_category('discussing islam vs atheism')
predict_category('determining the screen resolution')

# this is nothing more sophisticated than a simple probability model
# for the (weighted) frequency of each word in the string; 
# nevertheless, the result is striking

# Even a very naive algorithm, 
# when used carefully and trained on a large set of high-dimensional data, 
# can be surprisingly effective


#endregion


#endregion

#region When to Use Naive Bayes
# Because naive Bayesian classifiers make such stringent assumptions about data, 
# they will generally not perform as well as a more complicated model

# That said, they have several advantages:
#    • They are extremely fast for both training and prediction
#    • They provide straightforward probabilistic prediction
#    • They are often very easily interpretable
#    • They have very few (if any) tunable parameters

# These advantages mean a naive Bayesian classifier is often a good choice as an initial baseline classification
# If it performs suitably, 
#   then congratulations: 
#   you have a very fast, very interpretable classifier for your problem
# If it does not perform well, 
#   then you can begin exploring more sophisticated models, 
#   with some baseline knowledge of how well they should perform

# Naive Bayes classifiers tend to perform especially well in one of the following situations
#    • When the naive assumptions actually match the data (very rare in practice)
#    • For very well-separated categories, when model complexity is less important
#    • For very high-dimensional data, when model complexity is less important

# last two points seem distinct, but they actually are related: 
#   as the dimension of a dataset grows, 
#   it is much less likely for any two points to be found close together
#       (after all, they must be close in every single dimension to be close overall)
# This means that 
#   clusters in high dimensions tend to be more separated, on average, 
#   than clusters in low dimensions, 
#   assuming the new dimensions actually add information
# For this reason, 
#   simplistic classifiers like naive Bayes tend to work as well or better than more complicated classifiers 
#   as the dimensionality grows: 
#   once you have enough data, even a simple model can be very powerful


#endregion
