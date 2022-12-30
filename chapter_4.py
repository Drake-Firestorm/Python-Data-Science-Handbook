# CHAPTER 4: Visualization with Matplotlib
# ========================================
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

# One of Matplotlib’s most important features 
# is its ability to play well with many operating systems and graphics backends
# Matplotlib supports dozens of backends and output types, 
# which means you can count on it to work regardless of 
#   which operating system you are using or 
#   which output format you wish


#region General Matplotlib Tips

#region Importing matplotlib
# some standard shorthands for Matplotlib imports
# import matplotlib as mpl
# import matplotlib.pyplot as plt


# plt interface is what we will use most often


#endregion

#region Setting Styles
# use the plt.style directive to choose appropriate aesthetic styles for our figures
#   classic style, ensures that the plots we create use the classic Matplotlib style

plt.style.use('classic')


#endregion

#region show() or No show()? How to Display Your Plots
# best use of Matplotlib differs depending on how you are using it; 
# roughly, the three applicable contexts are using 
#   Matplotlib in a script, 
#   in an IPython terminal, or 
#   in an IPython notebook


#region Plotting from a script
# If you are using Matplotlib from within a script, the function plt.show() is your friend
# plt.show() 
#   starts an event loop, 
#   looks for all currently active figure objects, and 
#   opens one or more interactive windows that display your figure or figures

# plt.show() command should be used only once per Python session, 
#   and is most often seen at the very end of the script
# Multiple show() commands can lead to unpredictable backend-dependent behavior, 
#   and should mostly be avoided


import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()


#endregion

#region Plotting from an IPython shell


#endregion

#region Plotting from an IPython notebook


#endregion


#endregion

#region Saving Figures to File
# One nice feature of Matplotlib is the ability to save figures in a wide variety of formats
# can save a figure using the savefig() command in the current working directory
# when saving your figure, it’s not necessary to use plt.show() or related commands

# e.g. 
x = np.linspace(0, 10, 100)
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');

fig.savefig('my_figure.png')


# In savefig(), the file format is inferred from the extension of the given filename
# find the list of supported file types for your system by using the
#   following method of the figure canvas object
fig.canvas.get_supported_filetypes()


#endregion


#endregion

#region Two Interfaces for the Price of One
# potentially confusing feature of Matplotlib is its dual interfaces: 
#   a convenient MATLAB-style state-based interface, and 
#   a more powerful object-oriented interface


#region MATLAB-style interface
# MATLAB-style tools are contained in the pyplot (plt) interface
# important to note that this interface is stateful: 
#   it keeps track of the “current” figure and axes, 
#   which are where all plt commands are applied
# can get a reference to these using the 
#   plt.gcf() (get current figure) and 
#   plt.gca() (get current axes) routines


plt.figure()    # create a plot figure
# create the first of two panels and set current axis
plt.subplot(2, 1, 1)    # (rows, columns, panel number)
plt.plot(x, np.sin(x))
# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()


# While this stateful interface is fast and convenient for simple plots, it is easy to run into problems
# e.g. once the second panel is created, how can we go back and add something to the first?


#endregion

#region Object-oriented interface
# object-oriented interface is available 
#   for these more complicated situations, and
#   for when you want more control over your figure

# in the object-oriented interface 
#   the plotting functions are methods of explicit Figure and Axes objects

# e.g. To re-create the previous plot using this style of plotting

# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

plt.show()


# For more simple plots, the choice of which style to use is largely a matter of preference, 
# but the object-oriented approach can become a necessity as plots become more complicated


#endregion


#endregion

#region Simple Line Plots
plt.style.use('seaborn-whitegrid')


# For all Matplotlib plots, we start by creating a figure and an axes
# In their simplest form, a figure and axes can be created as follows
fig = plt.figure()
ax = plt.axes()
plt.show()

# In Matplotlib, 
#   the figure (an instance of the class plt.Figure) 
#       can be thought of as a single container that contains all the objects representing axes, graphics, text, and labels
#    axes (an instance of the class plt.Axes): 
#       a bounding box with ticks and labels, which will eventually contain the plot elements that make up our visualization


# Once we have created an axes, we can use the ax.plot function to plot some data
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))
plt.show()

# Alternatively, we can use the pylab interface and let the figure and axes be created for us in the background
plt.plot(x, np.sin(x))
plt.show()


# If we want to create a single figure with multiple lines, 
#   we can simply call the plot function multiple times
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()


#region Adjusting the Plot: Line Colors and Styles
# plt.plot() function takes additional arguments that can be used to specify these

# To adjust the color, 
# you can use the color keyword, 
#   which accepts a string argument representing virtually any imaginable color
# If no color is specified, 
#   Matplotlib will automatically cycle through a set of default colors for multiple lines
# color can be specified in a variety of ways
plt.plot(x, np.sin(x - 0), color='blue') # specify color by name
plt.plot(x, np.sin(x - 1), color='g') # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75') # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44') # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 and 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
plt.show()


# adjust the line style using the linestyle keyword 
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-') # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':'); # dotted

plt.show()


# If you would like to be extremely terse, 
# these linestyle and color codes can be combined into a single nonkeyword argument to the plt.plot() function
plt.plot(x, x + 0, '-g') # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r'); # dotted red
plt.show()

# single-character color codes reflect the standard abbreviations in the 
#   RGB (Red/Green/Blue) and 
#   CMYK (Cyan/Magenta/Yellow/blacK) color systems, 
# commonly used for digital color graphics


#endregion

#region Adjusting the Plot: Axes Limits
# most basic way to adjust axis limits is to use the plt.xlim() and plt.ylim() methods
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.show()


# If for some reason you’d like either axis to be displayed in reverse, 
# you can simply reverse the order of the arguments
plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2)
plt.show()


# useful related method is plt.axis() 
#   (note here the potential confusion between axes with an e, and axis with an i)
# plt.axis() method 
#   allows you to set the x and y limits with a single call, 
#   by passing a list that specifies [xmin, xmax, ymin, ymax]
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])
plt.show()

# plt.axis() method goes even beyond this, 
#   allowing you to do things like automatically tighten the bounds around the current plot
plt.plot(x, np.sin(x))
plt.axis('tight')
plt.show()

# It allows even higher-level specifications, 
#   such as ensuring an equal aspect ratio 
#       so that on your screen, one unit in x is equal to one unit in y
plt.plot(x, np.sin(x))
plt.axis('equal')
plt.show()


#endregion

#region Labeling Plots
# Titles and axis labels are the simplest such labels
#   there are methods that can be used to quickly set them 
# can adjust the position, size, and style of these labels using optional arguments to the function
plt.plot(x, np.sin(x))
plt.title('A Sine Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()


# When multiple lines are being shown within a single axes, 
#   it can be useful to create a plot legend that labels each line type
# It is done via the plt.legend() method
# Though there are several valid ways of using this, 
#   find it easiest to specify the label of each line using the label keyword of the plot function
# plt.legend() function keeps track of the line style and color, and matches these with the correct label
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.axis('equal')
plt.legend()
plt.show()


#endregion

#region Matplotlib Gotchas
# While most plt functions translate directly to ax methods 
#       (such as plt.plot() → ax.plot(), plt.legend() → ax.legend(), etc.), 
#   this is not the case for all commands
# functions to set limits, labels, and titles are slightly modified
# For transitioning between MATLAB-style functions and object-oriented methods, make the following changes
# • plt.xlabel() → ax.set_xlabel()
# • plt.ylabel() → ax.set_ylabel()
# • plt.xlim() → ax.set_xlim()
# • plt.ylim() → ax.set_ylim()
# • plt.title() → ax.set_title()


# In the object-oriented interface to plotting, 
# rather than calling these functions individually, 
# it is often more convenient to use the ax.set() method to set all these properties at once 
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
        xlabel='x', ylabel='y',
        title='A Simple Plot')
plt.show()


#endregion


#endregion

#region Simple Scatter Plots

#region Scatter Plots with plt.plot
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')
plt.show()

# third argument in the function call is a character that represents the type of symbol used for the plotting
# marker style has its own set of short string codes
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)
plt.show()


# For even more possibilities, 
# these character codes can be used together with line and color codes 
#   to plot points along with a line connecting them
plt.plot(x, y, '-ok')   # line (-), circle marker (o), black (k)
plt.show()


# Additional keyword arguments to plt.plot specify a wide range of properties of the lines and markers
plt.plot(x, y, '-p', color='gray',
            markersize=15, linewidth=4,
            markerfacecolor='white',
            markeredgecolor='gray',
            markeredgewidth=2)
plt.ylim(-1.2, 1.2)
plt.show()


#endregion

#region Scatter Plots with plt.scatter
# second, more powerful method of creating scatter plots is the 
#   plt.scatter function, 
#       which can be used very similarly to the plt.plot function
# primary difference of plt.scatter from plt.plot is that 
#   it can be used to create scatter plots 
#   where the properties of each individual point (size, face color, edge color, etc.) 
#   can be individually controlled or mapped to data
plt.scatter(x, y, marker='o')
plt.show()


# use the alpha keyword to adjust the transparency level
# color argument is automatically mapped to a color scale 
#   (shown here by the colorbar() command), and 
# the size argument is given in pixels
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
plt.colorbar()  # show color scale
plt.show()

# e.g. Iris data from Scikit-Learn
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()


#endregion

#region plot Versus scatter: A Note on Eiciency
# While it doesn’t matter as much for small amounts of data, 
# as datasets get larger than a few thousand points, 
#   plt.plot can be noticeably more efficient than plt.scatter
# For large datasets, 
#   the difference between these two can lead to vastly different performance,
#   and for this reason, plt.plot should be preferred over plt.scatter for large datasets
# reason is that 
#   plt.scatter 
#       has the capability to render a different size and/or color for each point, 
#       so the renderer must do the extra work of constructing each point individually
#   plt.plot, on the other hand, 
#       the points are always essentially clones of each other, 
#       so the work of determining the appearance of the points is done only once for the entire set of data


#endregion


#endregion

#region Visualizing Errors
# For any scientific measurement, accurate accounting for errors is nearly as important,
#   if not more important, than accurate reporting of the number itself
# In visualization of data and results, showing these errors effectively 
#   can make a plot convey much more complete information


#region Basic Errorbars
# basic errorbar can be created with a single Matplotlib function call 
#   fmt 
#       is a format code controlling the appearance of lines and points, 
#       and has the same syntax as the shorthand used in plt.plot
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.show()


# errorbar function has many options to finetune the outputs
# Using these additional options you can easily customize the aesthetics of your errorbar plot
# find it helpful, especially in crowded plots, to make the errorbars lighter than the points themselves
#   can also specify 
#       horizontal errorbars (xerr), 
#       onesided errorbars, and 
#       many other variants
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
                ecolor='lightgray', elinewidth=3, capsize=0)
plt.show()


#endregion

#region Continuous Errors
# In some situations it is desirable to show errorbars on continuous quantities
# Though Matplotlib does not have a built-in convenience routine for this type of application,
#   it’s relatively easy to combine primitives like plt.plot and plt.fill_between for a useful result


# perform a simple Gaussian process regression (GPR), using the Scikit-Learn API 
#   This is a method 
#       of fitting a very flexible nonparametric function 
#       to data with a continuous measure of the uncertainty

from sklearn.gaussian_process import GaussianProcessRegressor

# define the model and draw some data
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# Compute the Gaussian process fit
gp = GaussianProcessRegressor()
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], return_std=True)
dyfit = 2 * np.sqrt(MSE)    # 2*sigma ~ 95% confidence region

# now have xfit, yfit, and dyfit, which sample the continuous fit to our data
# use the plt.fill_between function with a light color to visualize this continuous error

# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
plt.xlim(0, 10)
plt.show()


# with the fill_between function: 
#   we pass an x value, 
#   then the lower y-bound, 
#   then the upper y-bound, and 
#   the result is that the area between these regions is filled


# resulting figure gives a very intuitive view into what the Gaussian process regression algorithm is doing: 
#   in regions near a measured data point, 
#       the model is strongly constrained and 
#       this is reflected in the small model errors. 
#   In regions far from a measured data point, 
#       the model is not strongly constrained, and 
#       the model errors increase


#endregion


#endregion

#region Density and Contour Plots
# Sometimes it is useful to display three-dimensional data in two dimensions using contours or color-coded regions
# three Matplotlib functions that can be helpful for this task: 
#   plt.contour for contour plots, 
#   plt.contourf for filled contour plots, and 
#   plt.imshow for showing images
# combination of these three functions
#       plt.contour, 
#       plt.contourf, and
#       plt.imshow
#   gives nearly limitless possibilities for displaying this sort of threedimensional data within a two-dimensional plot


#region Visualizing a Three-Dimensional Function
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * y) * np.cos(x)


# contour plot can be created with the plt.contour function
# takes three arguments: 
#   a grid of x values, 
#   a grid of y values, and 
#   a grid of z values
#       x and y values represent positions on the plot, and the 
#       z values will be represented by the contour levels


# most straightforward way to prepare such data is to use the np.meshgrid function, 
#   which builds two-dimensional grids from one-dimensional arrays

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# look at this with a standard line-only contour plot
plt.contour(X, Y, Z, colors='black')
plt.show()


# by default when a single color is used, 
#   negative values are represented by dashed lines, and 
#   positive values by solid lines
# can color-code the lines 
#   by specifying a colormap with the cmap argument

# also specify that we want more lines to be drawn
#   20 equally spaced intervals within the data range
# chose the RdGy (short for Red-Gray) colormap, 
#   which is a good choice for centered data
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.show()


# spaces between the lines may be a bit distracting
# change this by switching to a filled contour plot using the plt.contourf() function 
#   (notice the f at the end), 
#   which uses largely the same syntax as plt.contour()

# add a plt.colorbar() command, 
#   which automatically creates an additional axis with labeled color information for the plot

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.show()
# colorbar makes it clear that 
#   the black regions are “peaks,” while 
#   the red regions are “valleys.


# One potential issue with this plot is that it is a bit “splotchy.”
#   That is, the color steps are discrete rather than continuous, 
#       which is not always what is desired
# use the plt.imshow() function, 
#   which interprets a two-dimensional grid of data as an image
# few potential gotchas with imshow(), however:
#   • plt.imshow() doesn’t accept an x and y grid, 
#       so you must manually specify the extent [xmin, xmax, ymin, ymax] of the image on the plot.
# • plt.imshow() by default follows the standard image array definition where the
#       origin is in the upper left, not in the lower left as in most contour plots. 
#       This must be changed when showing gridded data
# • plt.imshow() will automatically adjust the axis aspect ratio to match the input data; 
#       you can change this by setting, 
#           for example, plt.axis(aspect='image') 
#       to make x and y units match

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image')
plt.show()


# sometimes be useful to combine contour plots and image plots
# use a 
#   partially transparent background image 
#       (with transparency set via the alpha parameter) and 
#   over-plot contours with labels on the contours themselves 
#       (using the plt.clabel() function)

contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
plt.colorbar()

plt.show()


#endregion


#endregion

#region Histograms, Binnings, and Density
# simple histogram can be a great first step in understanding a dataset

data = np.random.randn(1000)
plt.hist(data)
plt.show()

# hist() function has many options to tune both the calculation and the display
plt.hist(data, bins=30, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
plt.show()


# combination of 
#   histtype='stepfilled' along with some 
#   transparency alpha 
# to be very useful when comparing histograms of several distributions

x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)
plt.show()


# to simply compute the histogram 
#   (that is, count the number of points in a given bin) 
# and not display it, the np.histogram() function is available

counts, bin_edge = np.histogram(data, bins=5); counts


#region Two-Dimensional Histograms and Binnings
# create histograms in two dimensions by dividing points among twodimensional bins

mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T


#region plt.hist2d: Two-dimensional histogram
# One straightforward way to plot a two-dimensional histogram is to use Matplotlib’s plt.hist2d function

plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bins')
plt.show()


# plt.hist2d has a counterpart in np.histogram2d, which can be used as follows
counts, xedges, yedges = np.histogram2d(x, y, bins=30)


#endregion

#region plt.hexbin: Hexagonal binnings
# two-dimensional histogram creates a tessellation of squares across the axes
# Another natural shape for such a tessellation is the regular hexagon

# Matplotlib provides the plt.hexbin routine, 
#   which represents a two-dimensional dataset binned within a grid of hexagons

plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
plt.show()

# plt.hexbin has a number of interesting options, including the 
#   ability to specify weights for each point, and 
#   to change the output in each bin to any NumPy aggregate 
#       (mean of weights, standard deviation of weights, etc.)


#endregion

#region Kernel density estimation
# Another common method of evaluating densities in multiple dimensions is kernel density estimation (KDE)
# KDE can be thought of as a way to 
#   “smear out” the points in space and 
#   add up the result to obtain a smooth function
# One extremely quick and simple KDE implementation exists in the scipy.stats package

from scipy.stats import gaussian_kde
# fit an array of size [Ndim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)
# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent=[-3.5, 3.5, -6, 6], cmap='Blues')
cb = plt.colorbar()
cb.set_label('density')
plt.show()


# KDE has a smoothing length that effectively slides the knob between detail and smoothness 
#   (one example of the ubiquitous bias–variance trade-off)
#   gaussian_kde uses a rule of thumb to attempt to find a nearly optimal smoothing length for the input data
# Other KDE implementations are available within the SciPy ecosystem, 
#   each with its own various strengths and weaknesses
# For visualizations based on KDE, using Matplotlib tends to be overly verbose
#   Seaborn library provides a much more terse API for creating KDE-based visualizations


#endregion


#endregion

#endregion

#region Customizing Plot Legends
# Plot legends give meaning to a visualization, assigning labels to the various plot elements

# simplest legend can be created with the plt.legend() command, 
#   which automatically creates a legend for any labeled plot elements

plt.style.use('classic')

x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()
plt.show()



# many ways we might want to customize such a legend
# e.g. specify the location and turn off the frame
ax.legend(loc='upper left', frameon=False)
fig     # need to rerun the above code in IDE
plt.show()


# use the ncol command to specify the number of columns in the legend
ax.legend(frameon=False, loc='lower center', ncol=2)
plt.show()


# use a rounded box (fancybox) or 
# add a shadow, 
# change the transparency (alpha value) of the frame, or 
# change the padding around the text 
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()


#region Choosing Elements for the Legend
# legend includes all labeled elements by default

# can fine-tune which elements and labels appear in the legend 
#   by using the objects returned by plot commands
# plt.plot() command is able to create multiple lines at once, 
#   and returns a list of created line instances
# Passing any of these to plt.legend() 
#   will tell it which to identify, 
#   along with the labels we’d like to specify

y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)
# lines is a list of plt.Line2D instances
plt.legend(lines[:2], ['first', 'second'])
plt.show()


# by default, the legend ignores all elements without a label attribute set
# clearer to use the first method, 
#   applying labels to the plot elements you’d like to show on the legend 
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True)
plt.show()


#endregion

#region Legend for Size of Points
# legend that specifies the scale of the sizes of the points, and 
# we’ll accomplish this by plotting some labeled data with no entries
# legend will always reference some object that is on the plot, 
#   so if we’d like to display a particular shape we need to plot it
# In this case, the objects we want (gray circles) are not on the plot, 
#   so we fake them by plotting empty lists
# legend only lists plot elements that have a label specified
# By plotting empty lists, 
#   we create labeled plot objects that are picked up by the legend,
#   and now our legend tells us some useful information
# This strategy can be useful for creating more sophisticated visualizations


cities = pd.read_csv('Data/california_cities.csv'); cities.head()

# Extract the data we're interested in
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size and color but no label
plt.scatter(lon, lat, label=None,
            c=np.log10(population), cmap='viridis',
            s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Here we create a legend:
# we'll plot empty lists with the desired size and label
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False,
                labelspacing=1, title='City Area')

plt.title('California Cities: Area and Population')
plt.show()


#endregion

#region Multiple Legends
# to add multiple legends to the same axes
# Matplotlib does not make this easy: 
#   via the standard legend interface,
#   it is only possible to create a single legend for the entire plot
# If you try to create a second legend using plt.legend() or ax.legend(), 
#   it will simply override the first one

# work around this 
#   by creating a new legend artist from scratch, and 
#   then using the lower-level ax.add_artist() method 
#       to manually add the second artist to the plot

fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                    styles[i], color='black')
ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
            loc='upper right', frameon=False)

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
                loc='lower right', frameon=False)
ax.add_artist(leg)

plt.show()


# This is a peek into the low-level artist objects that compose any Matplotlib plot


#endregion


#endregion

#region Customizing Colorbars
# Plot legends 
#   identify discrete labels of discrete points
# For continuous labels 
#   based on the color of points, lines, or regions, 
#   a labeled colorbar can be a great tool

# In Matplotlib,    
#   a colorbar is a separate axes 
#   that can provide a key for the meaning of colors in a plot

# simplest colorbar can be created with the plt.colorbar function
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar()
plt.show()


#region Customizing Colorbars
# specify the colormap using the cmap argument to the plotting function that is creating the visualization
# All the available colormaps are in the plt.cm namespace
plt.imshow(I, cmap='gray')
plt.show()


# being able to choose a colormap is just the first step: 
# more important is how to decide among the possibilities!
# choice turns out to be much more subtle than you might initially expect


#region Choosing the colormap
# Broadly, you should be aware of three different categories of colormaps:
#     Sequential colormaps
#         These consist of one continuous sequence of colors (e.g., binary or viridis).
#     Divergent colormaps
#         These usually contain two distinct colors, which show positive and negative
#         deviations from a mean (e.g., RdBu or PuOr).
#     Qualitative colormaps
#         These mix colors with no particular sequence (e.g., rainbow or jet)

# qualitative maps are often a poor choice for representing quantitative data
# Among the problems is the fact that 
#   qualitative maps usually do not display any uniform progression in brightness as the scale increases

from matplotlib.colors import LinearSegmentedColormap

def greyscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    cmap = greyscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                            subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    plt.show()

view_colormap('jet')

# Notice the bright stripes in the grayscale image. 
# Even in full color, this uneven brightness means that 
#   the eye will be drawn to certain portions of the color range, 
#   which will potentially emphasize unimportant parts of the dataset
# It’s better to use a colormap such as viridis
#   which is specifically constructed to have an even brightness variation across the range
#   it 
#       not only plays well with our color perception, but 
#       also will translate well to grayscale printing

view_colormap('viridis')


# If you favor rainbow schemes, 
#   another good option for continuous data is the cubehelix colormap 
view_colormap('cubehelix')


# For other situations, 
#   such as showing positive and negative deviations from some mean, 
#   dual-color colorbars such as RdBu (short for Red-Blue) can be useful
# it’s important to note that the positive-negative information will be lost upon translation to grayscale!

view_colormap('RdBu')


#endregion

#region Color limits and extensions
# colorbar itself is simply an instance of plt.Axes, 
#   so all of the axes and tick formatting tricks we’ve learned are applicable

# colorbar has some interesting flexibility
# e.g.
#   can narrow the color limits and 
#   indicate the out-of-bounds values 
#       with a triangular arrow at the top and bottom 
#   by setting the extend property
# might come in handy, for example, 
#   if you’re displaying an image that is subject to noise

# make noise in 1% of the image pixels
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1)

plt.show()


# in the left panel, 
#   the default color limits respond to the noisy pixels, and
#   the range of the noise completely washes out the pattern we are interested in
# In the right panel, 
#   we manually set the color limits, and 
#   add extensions to indicate values that are above or below those limits
#   result is a much more useful visualization of our data


#endregion

#region Discrete colorbars
# Colormaps are by default continuous, 
# but sometimes you’d like to represent discrete values

# easiest way to do this is 
#   to use the plt.cm.get_cmap() function, and 
#   pass 
#       the name of a suitable colormap along with 
#       the number of desired bins

plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1)
plt.show()


#endregion


#endregion

#region Example: Handwritten Digits
# interesting visualization of some handwritten digits data

# visualizing several of the example images with plt.imshow()

# load images of the digits 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])
plt.show()


# Because each digit is defined by the hue of its 64 pixels,
# we can consider each digit to be a point lying in 64-dimensional space: 
#   each dimension represents the brightness of one pixel
# But visualizing relationships in such high-dimensional spaces can be extremely difficult
# One way to approach this is to use a dimensionality reduction technique 
#   such as manifold learning 
#   to reduce the dimensionality of the data while maintaining the relationships of interest

# let’s take a look at a two-dimensional manifold learning projection of this digits data
# use our discrete colormap to view the results, 
# setting the ticks and clim to improve the aesthetics of the resulting colorbar

# project the digits into 2 dimensions using IsoMap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)

# plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)
plt.show()

# projection also gives us some interesting insights on the relationships within the dataset:
# for example, 
#   the ranges of 5 and 3 nearly overlap in this projection, 
#       indicating that some handwritten fives and threes are difficult to distinguish, and 
#       therefore more likely to be confused by an automated classification algorithm. 
#   Other values, like 0 and 1, 
#       are more distantly separated, and 
#       therefore much less likely to be confused. 
# This observation agrees with our intuition, 
#   because 5 and 3 look much more similar than do 0 and 1


#endregion


#endregion

#region Multiple Subplots
# Sometimes it is helpful to compare different views of data side by side
# To this end, Matplotlib has the concept of subplots: 
#   groups of smaller axes that can exist together within a single figure
# These subplots might be insets, grids of plots, or other more complicated layouts

#region plt.axes: Subplots by Hand
# most basic method of creating an axes is to use the plt.axes function
# by default this creates a standard axes object that fills the entire figure
# plt.axes also takes an optional argument 
#   that is a list of four numbers in the figure coordinate system
#   These numbers represent 
#       [bottom, left, width, height] 
#   in the figure coordinate system, 
#       which ranges from 
#           0 at the bottom left of the figure to 
#           1 at the top right of the figure

# e.g. might create an inset axes at the top-right corner of another axes 
#   by setting the x and y position to 0.65 
#       (that is, starting at 65% of the width and 65% of the height of the figure) and 
#   the x and y extents to 0.2 
#       (that is, the size of the axes is 20% of the width and 20% of the height of the figure)

ax1 = plt.axes()    # standard axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
plt.show()

# equivalent of this command within the object-oriented interface is fig.add_axes()

# e.g. to create two vertically stacked axes
# now have two axes 
#   (the top with no tick labels) 
#   that are just touching: 
#       the bottom of the upper panel (at position 0.5) matches 
#       the top of the lower panel (at position 0.1 + 0.4)
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                    xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                    ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
plt.show()


#endregion

#region plt.subplot: Simple Grids of Subplots
# Aligned columns or rows of subplots are a common enough need 
# that Matplotlib has several convenience routines that make them easy to create

# lowest level of these is plt.subplot(), 
#   which creates a single subplot within a grid
# command takes three integer arguments—
#   the number of rows, 
#   the number of columns, and 
#   the index of the plot to be created in this scheme, 
# which runs from the upper left to the bottom right

for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
plt.show()


# command plt.subplots_adjust 
#   can be used to adjust the spacing between these plots
# used the hspace and wspace arguments of plt.subplots_adjust, 
#   which specify the spacing along the height and width of the figure, 
#   in units of the subplot size 
#       (in the below case, the space is 40% of the subplot width and height)

# equivalent object-oriented command, fig.add_subplot()
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
plt.show()


#endregion

#region plt.subplots: The Whole Grid in One Go
# approach just described can become quite tedious 
# when you’re creating a large grid of subplots, 
# especially if you’d like to hide the x- and y-axis labels on the inner plots

# For this purpose, 
#   plt.subplots() is the easier tool to use (note the s at the end of subplots)
# this function creates a full grid of subplots in a single line, 
#   returning them in a NumPy array
# arguments are the
#   number of rows and 
#   number of columns, along with 
#   optional keywords sharex and sharey, 
#       which allow you to specify the relationships between different axes

# create a 2×3 grid of subplots, where 
#   all axes in the same row share their y-axis scale, and 
#   all axes in the same column share their x-axis scale
# by specifying sharex and sharey, 
#   we’ve automatically removed inner labels on the grid 
#   to make the plot cleaner
# resulting grid of axes instances is returned within a NumPy array, 
#   allowing for convenient specification of the desired axes 
#   using standard array indexing notation

fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
plt.show()

# axes are in a two-dimensional array, indexed by [row, col]
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')
plt.show()


# In comparison to plt.subplot(), 
# plt.subplots() is more consistent with Python’s conventional 0-based indexing


#endregion

#region plt.GridSpec: More Complicated Arrangements
# To go beyond a regular grid to subplots that span multiple rows and columns,
# plt.GridSpec() is the best tool

# plt.GridSpec() object does not create a plot by itself; 
# it is simply a convenient interface that is recognized by the plt.subplot() command

# e.g. gridspec for a grid of two rows and three columns
#   with some specified width and height space 
#   looks like this
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

# From this we can specify subplot locations and extents 
# using the familiar Python slic‐ ing syntax 
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])
plt.show()


# This type of flexible grid alignment has a wide range of uses
# e.g. most often use it when creating multi-axes histogram plots

# Create some normally distributed data
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
y_hist.invert_yaxis()
plt.show()


#endregion


#endregion

#region Text and Annotation
# Creating a good visualization involves guiding the reader so that the figure tells a story
# In some cases, 
#   this story can be told in an entirely visual manner, 
#   without the need for added text, 
# but in others, 
#   small textual cues and labels are necessary

# most basic types of annotations you will use are axes labels and titles

plt.style.use('seaborn-whitegrid')


#region Example: Efect of Holidays on US Births
births = pd.read_csv('Data/births.csv')
births.head(); births.shape

quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format='%Y%m%d')

births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]


fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax=ax)
plt.show()


# When we’re communicating data like this, 
# it is often useful to annotate certain features of the plot to draw the reader’s attention
# can be done manually with the
#   plt.text/ax.text command, 
#       which will place text at a particular x/y value
# ax.text method takes 
#   an x position, 
#   a y position, 
#   a string, and then 
#   optional keywords specifying the color, size, style, alignment, and other properties of the text
# ha - is short for horizonal alignment


fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
style = dict(size=10, color='gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

# Label the axes
ax.set(title='USA births by day of year (1969-1988)', ylabel='average daily births')

# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

plt.show()


#endregion

#region Transforms and Text Position
# In the previous example, we anchored our text annotations to data locations
# Sometimes it’s preferable to anchor the text to a position on the axes or figure, independent of the data
#   do this by modifying the transform

# Any graphics display framework needs some scheme for translating between coordinate systems
# Matplotlib has a well-developed set of tools that it uses internally to perform them 
#   (the tools can be explored in the matplotlib.transforms submodule)

# rarely needs to worry about the details of these transforms
# it is helpful knowledge to have when considering the placement of text on a figure

# three predefined transforms that can be useful in this situation:
# ax.transData
#   Transform associated with data coordinates
# ax.transAxes
#   Transform associated with the axes (in units of axes dimensions)
# fig.transFigure
#   Transform associated with the figure (in units of figure dimensions)
# by default, the text is aligned above and to the left of the specified coordinates; 
#   below, the “.” at the beginning of each string will approximately mark the given coordinate location

# transData coordinates 
#   give the usual data coordinates associated with the x- and y-axis labels
# transAxes coordinates 
#   give the location from the bottom-left corner of the axes (below, the white box) as a fraction of the axes size
# transFigure coordinates are similar, 
#   but specify the position from the bottom left of the figure (below, the gray box) as a fraction of the figure size
# if we change the axes limits, 
#   it is only the transData coordinates that will be affected, 
#   while the others remain stationary

# e.g. drawing text at various locations using these transforms

fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])
# transform=ax.transData is the default, but we'll specify it anyway
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure)
plt.show()

#change limits
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])
# transform=ax.transData is the default, but we'll specify it anyway
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure)
ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
plt.show()


#endregion

#region Arrows and Annotation
# Drawing arrows in Matplotlib is often much harder than you might hope

# While there is a plt.arrow() function available, 
#   wouldn’t suggest using it; 
#   the arrows it creates are SVG objects 
#       that will be subject to the varying aspect ratio of your plots,
#       and the result is rarely what the user intended

# suggest using the plt.annotate() function. 
#   This function creates some text and an arrow, 
#   and the arrows can be very flexibly specified
# arrow style is controlled through the arrowprops dictionary, 
#   which has numerous options available

fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6), arrowprops=dict(arrowstyle="->", connectionstyle='angle3, angleA=0, angleB=-90'))
plt.show()

# demonstrate several of the possible options using the birthrate plot from before 

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
# Add labels to the plot
ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data', xytext=(50, -30)
            , textcoords='offset points', arrowpropss=dict(arrowstyle="->", connectionstyle='arc3, rad=-0.2'))
ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data',
            bbox=dict(boxstyle='round', fc='none', ec='gray'),
            xytext=(10, -40), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="->"))
ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center', xytext=(0, -20), textcoords='offset points')
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850), xycoords='data'
            , textcoords='data', arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data', xytext=(-80, -40)
            , textcoords='offset points', arrowprops=dict(arrowstyle="fancy", fc="0.6", ec="none", connectionstyle="angle3,angleA=0,angleB=-90"))
ax.annotate('Thanksgiving', xy=('2012-11-25', 4500), xycoords='data', xytext=(-120, -60)
            , textcoords='offset points', bbox=dict(boxstyle="round4,pad=.5", fc="0.9")
            , arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=80,rad=20"))
ax.annotate('Christmas', xy=('2012-12-25', 3850), xycoords='data', xytext=(-30, 0)
            , textcoords='offset points', size=13, ha='right', va="center", bbox=dict(boxstyle="round", alpha=0.1)
            , arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));

# Label the axes
ax.set(title='USA births by day of year (1969-1988)', ylabel='average daily births')

# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));

ax.set_ylim(3600, 5400);

plt.show()



# specifications of the arrows and text boxes are very detailed: 
#   this gives you the power to create nearly any arrow style you wish
# Unfortunately, it also means that these sorts of features often must be manually tweaked, 
#   a process that can be very time-consuming when one is producing publication-quality graphics!
# preceding mix of styles is by no means best practice for presenting data, 
#   but rather included as a demonstration of some of the available options


#endregion


#endregion

#region Customizing Ticks
# Matplotlib’s default tick locators and formatters 
# are designed to be generally sufficient in many common situations, 
# but are in no way optimal for every plot

# understand further the object hierarchy of Matplotlib plots
# Matplotlib aims to have a Python object representing everything that appears on the plot
# Each Matplotlib object can also act as a container of sub-objects

# tick marks are no exception. 
# Each axes has attributes xaxis and yaxis, 
#   which in turn have attributes 
#       that contain all the properties of the lines, ticks, and labels that make up the axes


#region Major and Minor Ticks
# Within each axis, there is the concept of a major tick mark and a minor tick mark
#   major ticks are usually bigger or more pronounced, while
#   minor ticks are usually smaller

# By default, Matplotlib rarely makes use of minor ticks, 

ax = plt.axes(xscale='log', yscale='log')
plt.show()

# can customize these tick properties—
#   that is, locations and labels—
# by setting the formatter and locator objects of each axis

# locator
print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())
# formatter
print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())

# see that both major and minor tick labels have their locations specified by a LogLocator 
#   (which makes sense for a logarithmic plot)


#endregion

#region Hiding Ticks or Labels
# most common tick/label formatting operation is the act of hiding ticks or labels
# can do this using plt.NullLocator() and plt.NullFormatter()

ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())

plt.show()

# removed the labels (but kept the ticks/gridlines) from the x axis, and
# removed the ticks (and thus the labels as well) from the y axis

# Having no ticks at all can be useful in many situations
#   e.g. when you want to show a grid of images
fig, ax = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=0, wspace=0)

# Get some face data from scikit-learn
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap='bone')
plt.show()

# each image has its own axes, and 
# we’ve set the locators to null because 
#   the tick values (pixel number in this case) do not convey relevant information for this particular visualization


#endregion

#region Reducing or Increasing the Number of Ticks
# One common problem with the default settings is that 
#   smaller subplots can end up with crowded labels

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
plt.show()

# Particularly for the x ticks, 
#   the numbers nearly overlap, 
#   making them quite difficult to decipher

# can fix this with the plt.MaxNLocator(), 
#   which allows us to specify the maximum number of ticks that will be displayed
# Given this maximum number, 
#   Matplotlib will use internal logic to choose the particular tick locations

# For every axis, set the x and y major locator
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
plt.show()


# If you want even more control over the locations of regularly spaced ticks, 
# you might also use plt.MultipleLocator


#endregion

#region Fancy Tick Formats
# Matplotlib’s default tick formatting can leave a lot to be desired; 
#   it works well as a broad default, 
#   but sometimes you’d like to do something more

# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')
# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi)
plt.show()


# There are a couple changes we might like to make
# First, it’s more natural for this data to space the ticks and grid lines in multiples of π
#   can do this by setting a MultipleLocator, 
#       which locates ticks at a multiple of the number you provide

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig
plt.show()

# But now these tick labels look a little bit silly: 
#   we can see that they are multiples of π,
#   but the decimal representation does not immediately convey this
# To fix this, we can change the tick formatter
# There’s no built-in formatter for what we want to do, 
# so we’ll instead use plt.FuncFormatter, 
#   which accepts a user-defined function giving fine-grained control over the tick outputs

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))    
fig
plt.show()


# we’ve made use of Matplotlib’s LaTeX support, 
#   specified by enclosing the string within dollar signs
# This is very convenient for display of mathematical symbols and formulae; 
#   in this case, "$\pi$" is rendered as the Greek character π


# plt.FuncFormatter() offers extremely fine-grained control over the appearance of your plot ticks, 
# and comes in very handy when you’re preparing plots for presentation or publication


#endregion

#region Summary of Formatters and Locators


#endregion


#endregion

#region Customizing Matplotlib: Conigurations and Stylesheets
# Matplotlib’s runtime configuration (rc) options


#region Plot Customization by Hand
# It’s possible to do these customizations for each individual plot

# e.g. 
x = np.random.randn(1000)
plt.hist(x)
plt.show()

# adjust this by hand to make it a much more visually pleasing plot

# use a gray background
ax = plt.axes(fc='#E6E6E6')
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)

# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')

# control face and edge color of histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')
plt.show()


# But this took a whole lot of effort!
# We definitely do not want to have to do all that tweaking each time we create a plot
# Fortunately, there is a way to adjust these defaults once in a way that will work for all plots


#endregion

#region Changing the Defaults: rcParams
# Each time Matplotlib loads, 
# it defines a runtime configuration (rc) 
#   containing the default styles for every plot element you create
# can adjust this configuration at any time using the plt.rc convenience routine

# start by saving a copy of the current rcParams dictionary, 
# so we can easily reset these changes in the current session
py_default = plt.rcParams.copy()
# reset rcParams
plt.rcParams.update(py_default)

# use the plt.rc function to change some of these settings
from matplotlib import cycler

colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

# With these settings defined, we can now create a plot and see our settings in action
plt.hist(x)
plt.show()

# simple line plots look like with these rc parameters
for i in range(4):
    plt.plot(np.random.rand(10))
plt.show()


# These settings can be saved in a .matplotlibrc file

# prefer to customize Matplotlib using its stylesheets instead


#endregion

#region Stylesheets
# stylesheets are formatted similarly to the .matplotlibrc files mentioned earlier, 
#   but must be named with a .mplstyle extension

# Even if you don’t create your own style, 
# the stylesheets included by default are extremely useful

# available styles are listed in plt.style.available
plt.style.available[:5]

# basic way to switch to a stylesheet is to call
# this will change the style for the rest of the session!
plt.style.use('stylename')

# Alternatively, you can use the style context manager, 
# which sets a style temporarily
with plt.style.context('stylename'):
    make_a_plot()


# function that will make two basic types of plot
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')
    plt.show()


#region Default style
# reset our runtime configuration to the notebook default

# reset rcParams
plt.rcParams.update(py_default)

hist_and_lines()


#endregion

#region FiveThirtyEight style
# FiveThirtyEight style mimics the graphics found on the popular FiveThirtyEight website
# it is typified by bold colors, thick lines, and transparent axes

with plt.style.context('fivethirtyeight'):
    hist_and_lines()


#endregion

#region ggplot
# ggplot package in the R language is a very popular visualization tool
# Matplotlib’s ggplot style mimics the default styles from that package

with plt.style.context('ggplot'):
    hist_and_lines()


#endregion

#region Bayesian Methods for Hackers style
with plt.style.context('bmh'):
    hist_and_lines()


#endregion

#region Dark background
# For figures used within presentations, it is often useful to have a dark rather than light background
# dark_background style provides this
with plt.style.context('dark_background'):
    hist_and_lines()


#endregion

#region Grayscale
# for a print publication that does not accept color figures
# grayscale style can be very useful
with plt.style.context('grayscale'):
    hist_and_lines()


#endregion

#region Seaborn style
# Matplotlib also has stylesheets inspired by the Seaborn library
# these styles are loaded automatically when Seaborn is imported into a notebook
# tend to use them as defaults in my own data exploration

import seaborn
hist_and_lines()


# With all of these built-in options for various plot styles, 
# Matplotlib becomes much more useful for both interactive visualization and creation of figures for publication


#endregion


#endregion


#endregion

#region Three-Dimensional Plotting in Matplotlib
# enable three-dimensional plots by importing the mplot3d toolkit,
#   included with the main Matplotlib installation

from mpl_toolkits import mplot3d

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
