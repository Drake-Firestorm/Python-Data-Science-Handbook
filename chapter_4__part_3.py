import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cycler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import gaussian_kde
from matplotlib.legend import Legend
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.datasets import fetch_olivetti_faces
from mpl_toolkits import mplot3d
from matplotlib.tri import Triangulation
from itertools import chain

#region Three-Dimensional Plotting in Matplotlib
# enable three-dimensional plots by importing the mplot3d toolkit,
#   included with the main Matplotlib installation

# from mpl_toolkits import mplot3d

# create a three-dimensional axes by 
#   passing the keyword projection='3d' 
#   to any of the normal axes creation routines

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.show()

# With this 3D axes enabled, we can now plot a variety of three-dimensional plot types

# Three-dimensional plotting is one of the functionalities that benefits immensely from
#   viewing figures interactively rather than statically in the notebook


#region Three-Dimensional Points and Lines
# most basic three-dimensional plot is a line or scatter plot created from sets of (x, y, z) triples
# can create these using the ax.plot3D and ax.scatter3D functions
# call signature for these is nearly identical to that of their two-dimensional counterparts

# plot a trigonometric spiral, along with some points drawn randomly near the line
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

plt.show()


# by default, the scatter points have their transparency adjusted 
#   to give a sense of depth on the page
# While the three-dimensional effect is sometimes difficult to see within a static image, 
# an interactive view can lead to some nice intuition about the layout of the points


#endregion

#region Three-Dimensional Contour Plots
# Analogous to the contour plots
# mplot3d contains tools to create three-dimensional relief plots using the same inputs
# ax.contour3D requires all the input data 
#   to be in the form of two-dimensional regular grids, 
#   with the Z data evaluated at each point


# three-dimensional sinusoidal function
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


# Sometimes the default viewing angle is not optimal, 
# in which case we can use the view_init method 
#   to set the elevation and azimuthal angles

# e.g.
# use an elevation of 60 degrees 
#   (that is, 60 degrees above the x-y plane) and 
# an azimuth of 35 degrees 
#   (that is, rotated 35 degrees counter-clockwise about the z-axis)
ax.view_init(60, 35)
fig
plt.show()


# we can accomplish this type of rotation interactively 
# by clicking and dragging when using one of Matplotlib’s interactive backends


#endregion

#region Wireframes and Surface Plots
# Two other types of three-dimensional plots that work on gridded data are 
#   wireframes and 
#   surface plots

# These take a grid of values and 
# project it onto the specified threedimensional surface, and 
# can make the resulting three-dimensional forms quite easy to visualize

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe')
plt.show()


# surface plot is like a wireframe plot, 
#   but each face of the wireframe is a filled polygon
# Adding a colormap to the filled polygons 
#   can aid perception of the topology of the surface being visualized

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()


# though the grid of values for a surface plot needs to be two-dimensional, 
# it need not be rectilinear

# e.g. partial polar grid, 
# which when used with the surface3D plot 
# can give us a slice into the function we’re visualizing
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.show()


#endregion

#region Surface Triangulations
# For some applications,
# the evenly sampled grids required by the preceding routines 
# are overly restrictive and inconvenient

# In these situations, 
# the triangulation-based plots can be very useful

# What if rather than an even draw from a Cartesian or a polar grid, 
# we instead have a set of random draws?

theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)

# create a scatter plot of the points to get an idea of the surface we’re sampling from
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
plt.show()


# This leaves a lot to be desired
# function that will help us in this case is ax.plot_trisurf, 
#   which creates a surface by first finding a set of triangles formed between adjacent points

# remember that x, y, and z here are one-dimensional arrays
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
plt.show()


# result is certainly not as clean as when it is plotted with a grid, 
# but the flexibility of such a triangulation allows for some really interesting three-dimensional plots

# e.g. actually possible to plot a three-dimensional Möbius strip using this


#region Example: Visualizing a Möbius strip
# Möbius strip is similar to a strip of paper glued into a loop with a half-twist
# Topologically, it’s quite interesting because despite appearances it has only a single side!

# key to creating the Möbius strip is to think about its parameterization: 
#   it’s a two-dimensional strip, so we need two intrinsic dimensions
#       call them θ, 
#           which ranges from 0 to 2π around the loop, and 
#       w 
#           which ranges from –1 to 1 across the width of the strip
theta = np.linspace(0, 2 * np.pi, 30)
w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)

# from this parameterization, 
# we must determine the (x, y, z) positions of the embedded strip

# Thinking about it, we might realize that there are two rotations happening: 
#   one is the position of the loop about its center (what we’ve called θ), while the other is 
#   the twisting of the strip about its axis (we’ll call this ϕ)

# For a Möbius strip, we must have the strip make half a twist during a full loop, or Δϕ = Δθ/2
phi = 0.5 * theta

# use our recollection of trigonometry to derive the three-dimensional embedding
# define r, 
#   the distance of each point from the center, 
# and use this to find the embedded x, y, z coordinates
# radius in x-y plane
r = 1 + w * np.cos(phi)
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))

# Finally, to plot the object, we must make sure the triangulation is correct
# best way to do this is 
#   to define the triangulation within the underlying parameterization,
#   and then let Matplotlib project this triangulation into the three-dimensional space of the Möbius strip

# accomplished as follows
# triangulate in the underlying parameterization
from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap='viridis', linewidths=0.2)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
plt.show()


# Combining all of these techniques, 
# it is possible to create and display a wide variety of three-dimensional objects and patterns in Matplotlib


#endregion


#endregion


#endregion

#region Geographic Data with Basemap
# --------- Note: Basemap has been deprecated. Read only for reference. ---------

# One common type of visualization in data science is that of geographic data
# Matplotlib’s main tool for this type of visualization is the Basemap toolkit, 
#   which is one of several Matplotlib toolkits that live under the mpl_toolkits namespace

# Basemap feels a bit clunky to use, 
#   and often even simple visualizations take much longer to render than you might hope
# More modern solutions, 
#   such as leaflet or the Google Maps API, 
# may be a better choice for more intensive map visualizations
# Still, Basemap is a useful tool for Python users to have in their virtual toolbelts

from mpl_toolkits.basemap import Basemap

plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5)
plt.show()

# useful thing is that the globe shown here is not a mere image; 
# it is a fully functioning Matplotlib axes 
#   that understands spherical coordinates and 
#   allows us to easily over-plot data on the map!

# e.g. use a different map projection, 
#   zoom in to North America, and 
#   plot the location of Seattle. 
# use an etopo image 
#   (which shows topographical features both on land and under the ocean) 
# as the map background 


fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None, width=8E6, height=8E6, lat_0=45, lon_0=-100)
m.etopo(scale=0.5, alpha=0.5)

# Map (long, lat) to (x, y) for plotting
x, y = m(-122.3, 47.6)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Seattle', fontsize=12)
plt.show()


#region Map Projections
# first thing to decide when you are using maps is which projection to use
# Depending on the intended use of the map projection,
#   there are certain map features 
#       (e.g., direction, area, distance, shape, or other considerations) 
#   that are useful to maintain

# Basemap package implements several dozen such projections, all referenced by a short format code

# start by defining a convenience routine to draw our world map along with the longitude and latitude lines

from itertools import chain

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
    
    plt.show()


#region Cylindrical projections
# simplest of map projections are cylindrical projections, 
#   in which lines of constant latitude and longitude are mapped to horizontal and vertical lines, respectively
# This type of mapping represents equatorial regions quite well, 
#   but results in extreme distortions near the poles
# spacing of latitude lines varies between different cylindrical projections, 
#   leading to different conservation properties, 
#   and different distortion near the poles

# e.g. 
# equidistant cylindrical projection, 
#   which chooses a latitude scaling that preserves distances along meridians.
# Other cylindrical projections are 
#   the Mercator (projection='merc') and 
#   the cylindrical equal-area (projection='cea') projections

fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, )
draw_map(m)

# additional arguments to Basemap for this view specify 
#   the latitude (lat) and 
#   longitude (lon) of the 
#   lower-left corner (llcrnr) and 
#   upper-right corner (urcrnr) 
#   for the desired map, in units of degrees


#endregion

#region Pseudo-cylindrical projections
# Pseudo-cylindrical projections relax the requirement that 
#   meridians (lines of constant longitude) remain vertical; 
#   this can give better properties near the poles of the projection

# Mollweide projection (projection='moll') is one common example of this, 
#   in which all meridians are elliptical arcs
# constructed so as to preserve area across the map: 
#   though there are distortions near the poles, 
#   the area of small patches reflects the true area

# Other pseudo-cylindrical projections are 
#   the sinusoidal (projection='sinu') and 
#   Robinson (projection='robin') projections

fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='moll', resolution=None, lat_0=0, lon_0=0)
draw_map(m)

# extra arguments to Basemap here refer to 
#   the central latitude (lat_0) and 
#   longitude (lon_0) 
# for the desired map


#endregion

#region Perspective projections
# Perspective projections are constructed using a particular choice of perspective point,
#   similar to if you photographed the Earth from a particular point in space 
#   (a point which, for some projections, technically lies within the Earth!)

# orthographic projection (projection='ortho'), 
#   which shows one side of the globe as seen from a viewer at a very long distance
#   Thus, it can show only half the globe at a time

# Other perspective-based projections include the 
#   gnomonic projection (projection='gnom') and 
#   stereographic projection (projection='stere')
# These are often the most useful for showing small portions of the map

fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=0)
draw_map(m)


#endregion

#region Conic projections
# conic projection projects the map onto a single cone, which is then unrolled
# can lead to very good local properties, 
#   but regions far from the focus point of the cone may become very distorted

# can lead to very good local properties, 
#   but regions far from the focus point of the cone may become very distorted
# It projects the map onto a cone arranged in such a way that two standard parallels
#   (specified in Basemap by lat_1 and lat_2) 
#   have well-represented distances, 
#   with scale decreasing between them and increasing outside of them

# Other useful conic projections are the 
#   equidistant conic (projection='eqdc') and the 
#   Albers equal-area (projection='aea') projection

# Conic projections, like perspective projections, 
# tend to be good choices for representing small to medium patches of the globe

fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None, lon_0=0, lat_0=50, lat_1=45, lat_2=55, width=1.6E7, height=1.2E7)
draw_map(m)


#endregion

#region Other projections


#endregion


#endregion

#region Drawing a Map Background


#endregion

#region Plotting Data on Maps


#endregion

#region Example: California Cities


#endregion

#region Example: Surface Temperature Data


#endregion


#endregion

#region Visualization with Seaborn
# Seaborn provides an API on top of Matplotlib 
#   that offers sane choices for plot style and color defaults, 
#   defines simple high-level functions for common statistical plot types, and 
#   integrates with the functionality provided by Pandas DataFrames


#region Seaborn Versus Matplotlib
# simple random-walk plot in Matplotlib, using its classic plot formatting and colors
plt.style.use('classic')

# create some random walk data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
plt.show()

# Although the result contains all the information we’d like it to convey, 
# it does so in a way that is not all that aesthetically pleasing, 
# and even looks a bit old-fashioned in the context of 21st-century data visualization


# take a look at how it works with Seaborn
# Seaborn has many of its own high-level plotting routines, 
#   but it can also overwrite Matplotlib’s default parameters and 
#   in turn get even simple Matplotlib scripts to produce vastly superior output

# set the style by calling Seaborn’s set() method
# By convention, Seaborn is imported as sns

import seaborn as sns
sns.set()

# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
plt.show()


#endregion

#region Exploring Seaborn Plots
# main idea of Seaborn is that it provides high-level commands 
# to create a variety of plot types 
# useful for statistical data exploration, 
# and even some statistical model fitting

# all of the following could be done using raw Matplotlib commands 
#   (this is, in fact, what Seaborn does under the hood), 
# but the Seaborn API is much more convenient


#region Histograms, KDE, and densities
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)
plt.show()

# Rather than a histogram, 
# we can get a smooth estimate of the distribution using a kernel density estimation, 
#   which Seaborn does with sns.kdeplot
for col in 'xy':
    sns.kdeplot(data[col], shade=True)
plt.show()


# Histograms and KDE can be combined using distplot 
sns.distplot(data['x'])
sns.distplot(data['y'])
plt.show()


# If we pass the full two-dimensional dataset to kdeplot, 
#   we will get a two-dimensional visualization of the data
sns.kdeplot(data)
plt.show()


# can see the joint distribution and the marginal distributions together using sns.jointplot
# set the style to a white background
with sns.axes_style('white'):
    sns.jointplot(data=data, x='x', y='y', kind='kde')
plt.show()


# other parameters that can be passed to jointplot
# e.g. can use a hexagonally based histogram instead
with sns.axes_style('white'):
    sns.jointplot(data=data, x='x', y='y', kind='hex')
plt.show()


#endregion

#region Pair plots
# When you generalize joint plots to datasets of larger dimensions, you end up with pair plots
# useful for exploring correlations between multidimensional data, 
#   when you’d like to plot all pairs of values against each other

iris = sns.load_dataset("iris"); iris.head()

# Visualizing the multidimensional relationships among the samples is as easy as calling sns.pairplot 
sns.pairplot(iris, hue='species', size=2.5)
plt.show()


#endregion

#region Faceted histograms
# Sometimes the best way to view data is via histograms of subsets
# Seaborn’s FacetGrid makes this extremely simple

# data that shows the amount that restaurant staff receive in tips based on various indicator data 
tips = sns.load_dataset('tips'); tips.head()
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)
grid.map(plt.hist, 'tip_pct', bins=np.linspace(0, 40, 15))
plt.show()


#endregion

#region Factor plots
# Factor plots can be useful for this kind of visualization as well
# This allows you to view the distribution of a parameter within bins defined by any other parameter
with sns.axes_style(style='ticks'):
    g = sns.catplot(x='day', y='total_bill', col='sex', data=tips, kind='box')
    g.set_axis_labels('Day', 'Total Bill')
plt.show()


#endregion

#region Joint distributions
# can use sns.jointplot to show the joint distribution between different datasets, 
#   along with the associated marginal distributions
with sns.axes_style('white'):
    sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
plt.show()

# joint plot can even do some automatic kernel density estimation and regression
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
plt.show()


#endregion

#region Bar plots
# Time series can be plotted with sns.factorplot
planets = sns.load_dataset('planets'); planets.head()

with sns.axes_style('white'):
    g = sns.catplot(x='year', data=planets, aspect=2, kind='count', color='steelblue')
g.set_xticklabels(step=5)
plt.show()


# can learn more by looking at the method of discovery of each of these planets
with sns.axes_style('white'):
    g = sns.catplot(x='year', data=planets, aspect=2, kind='count', hue='method', order=range(2001, 2015))
g.set_ylabels('Number of Planets Discovered')
plt.show()


#endregion


#endregion

#region Example: Exploring Marathon Finishing Times
# help visualize and understand finishing results from a marathon
data = pd.read_csv('Data/marathon-data.csv'); data.head()

# By default, Pandas loaded the time columns as Python strings (type object); 
# we can see this by looking at the dtypes attribute of the DataFrame
data.dtypes

# Let’s fix this by providing a converter for the times:
data = pd.read_csv('Data/marathon-data.csv', converters={'split':pd.to_timedelta, 'final':pd.to_timedelta}) 
data.head(); data.dtypes


# For the purpose of our Seaborn plotting utilities, 
#   let’s next add columns that give the times in seconds
data['split_sec'] = data['split'].astype(np.int64) / 1E9
data['final_sec'] = data['final'].astype(np.int64) / 1E9
data.head()


# To get an idea of what the data looks like, we can plot a jointplot over the data
with sns.axes_style('white'):
    g = sns.jointplot(x='split_sec', y='final_sec', data=data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000), np.linspace(8000, 32000), ':k')
plt.show()


# dotted line shows where someone’s time would lie if they ran the marathon at a perfectly steady pace
# fact that the distribution lies above this indicates (as you might expect) that most people slow down over the course of the marathon
# those who do the opposite—run faster during the second half of the race—are said to have “negative-split” the race

# create another column in the data, the split fraction, 
# which measures the degree to which each runner negative-splits or positive-splits the race
#   Where this split difference is less than zero, the person negative-split the race by that fraction
data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']; data.head()

# do a distribution plot of this split fraction
sns.distplot(data['split_frac'], kde=False)
plt.axvline(0, color="k", linestyle="--")
plt.show()

sum(data.split_frac < 0)

# Out of nearly 40,000 participants, there were only 250 people who negative-split their marathon

# see whether there is any correlation between this split fraction and other variables
# do this using a pairgrid, 
#   which draws plots of all these correlations
g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'], hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend()
plt.show()

# looks like the split fraction does not correlate particularly with age, 
# but does correlate with the final time: 
#   faster runners tend to have closer to even splits on their marathon time

# see here that Seaborn is no panacea for Matplotlib’s ills when it comes to plot styles: 
#   in particular, the x-axis labels overlap

# difference between men and women here is interesting
# look at the histogram of split fractions for these two groups 
sns.kdeplot(data.split_frac[data.gender=='M'], label='men', shade=True)
sns.kdeplot(data.split_frac[data.gender=='W'], label='women', shade=True)
plt.xlabel('split_frac')
plt.legend()
plt.show()


# interesting thing here is that there are many more men than women who are running close to an even split!
# almost looks like some kind of bimodal distribution among the men and women

# see if we can suss out what’s going on by looking at the distributions as a function of age
# nice way to compare distributions is to use a violin plot
sns.violinplot(x='gender', y='split_frac', data=data, palette=["lightblue", "lightpink"])
plt.show()


# look a little deeper, and compare these violin plots as a function of age
# start by creating a new column in the array that specifies the decade of age that each person is in
data['age_dec'] = data.age.map(lambda age: 10 * (age // 10)); data.head()
men = (data.gender == 'M')
women = (data.gender == 'W')

with sns.axes_style(style=None):
    sns.violinplot(x='age_dec', y='split_frac', hue='gender', data=data, split=True, inner='quartile', palette=["lightblue", "lightpink"])
plt.show()

# Looking at this, we can see where the distributions of men and women differ: 
#   the split distributions of men in their 20s to 50s show a pronounced over-density toward
#   lower splits when compared to women of the same age (or of any age, for that matter)

# Also surprisingly, the 80-year-old women seem to outperform everyone in terms of their split time. 
#   This is probably due to the fact that we’re estimating the distribution from small numbers, 
#   as there are only a handful of runners in that range
(data.age > 80).sum()


# Back to the men with negative splits: 
#   who are these runners? 
#   Does this split fraction correlate with finishing quickly? 
# use regplot, 
#   which will automatically fit a linear regression to the data
g = sns.lmplot(x='final_sec', y='split_frac', col='gender', data=data, markers='.', scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.1, color="k", ls=":")
plt.show()

# Apparently the people with fast splits 
#   are the elite runners who are finishing within ~15,000 seconds, or about 4 hours. 
# People slower than that are much less likely to have a fast second split


#endregion


#endregion

#region Further Resources


#region Matplotlib Resources


#endregion

#region Other Python Graphics Libraries


#endregion


#endregion
