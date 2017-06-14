import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# Plot all available kernels
# [:, None] is just like [:, np.newaxis]
X_plot = np.linspace(-6, 6, 1000)[:, None]
X_src = np.zeros((1, 1))

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)

# this user-defined function is for the plt.FuncFormatter function
def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x

for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
                            'exponential', 'linear', 'cosine']):
    # ax.ravel() is a one-D array
    axi = ax.ravel()[i]
    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
    axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
    axi.text(-2.6, 0.95, kernel)
    # Go to https://matplotlib.org/api/ticker_api.html
    # to check out plt.FuncFormatte(), plt.MultipleLocato(), plt.NullLocator()
    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    # only show 1, 2, 3 ect that is result of muliply 1
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    # no locator is shon on y axis.
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title('Available Kernels')
plt.show()
