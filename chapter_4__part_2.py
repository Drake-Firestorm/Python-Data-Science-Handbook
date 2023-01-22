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
