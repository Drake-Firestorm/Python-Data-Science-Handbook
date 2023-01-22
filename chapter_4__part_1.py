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
