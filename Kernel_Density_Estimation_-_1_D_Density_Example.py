import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# Plot a 1D density example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
"""
scipy.stats.norm(0,1).pdf(0) calculates the probability on 0
in normal distribution whose mean is 0 and standard deviation is 1
and the probability is 0.3989422804014327

This query is to compare the pdf from KDE in serveral kernel to the true pdf,
and because we draw 100 samples from normal distribution, we already know that
true pdf are two normal distributions together on the x, y axises, and kde
pdf will be similar to two normal distributions together because we train kde
pdf with that 100 samples.
the reason we need to multiple 0.3 and 0.7 is that there are
only 30% sample from norm(0,1) and 70% samples from norm(0,1),
so the probability we got dat from norm(0,1) area
should be only 30% of the original norm(0,1)

"""
true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')

for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()
