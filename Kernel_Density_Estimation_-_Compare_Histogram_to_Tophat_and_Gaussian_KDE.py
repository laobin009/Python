import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


#----------------------------------------------------------------------
# Plot the progression of histograms to kernels
np.random.seed(1)
N = 20

# np.random.normal(0, 1, 0.3 * N): draw sample from normal distribution
# 0 is mean, 1 is standard deviation, 0.3 * N is the size of sample
X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

# sharex=True, sharey=True : x- or y-axis will be shared among all subplots
# fig : matplotlib.figure.Figure object
# ax : Axes object or array of Axes objects.
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# histogram 1
# normed = True
# If True, the first element of the return tuple will be the counts normalized
# to form a probability density,
# For ax[0, 0].hist, check "matplotlib.axes.Axes.hist" online
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# histogram 2
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# tophat KDE
kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
# kde.score computes the total log porbability under the model.
# We can infer that kde.score_samples computes the each sample's log probability
# under that trained model.
# so we have to use function np.exp() to get the porbability of each sample.
log_dens = kde.score_samples(X_plot)
# For ax[1, 0].fill, check "matplotlib.axes.Axes.fill" online
ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

for axi in ax.ravel():
    # check matplotlib.axes.Axes.plot for information
    # axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
    # It is for adding the '+' mark show the position of trained data sample
    # the mark is (x,y), x is from X and y is minus 0.01
    # which locate the mark under the 0.00 line.
    axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')

for axi in ax[1, :]:
    axi.set_xlabel('x')

plt.show()
