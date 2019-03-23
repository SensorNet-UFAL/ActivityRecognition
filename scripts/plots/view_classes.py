import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scripts.plots.commons_plot_functions import plot_svm_boundary

def plot_with_two_dim(clf,values, labels):
    colors = ['green', 'red']
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(values[:,0], values[:,1], c=labels, cmap=matplotlib.colors.ListedColormap(colors), s=20, edgecolors='k')
    cb = plt.colorbar()
    loc = np.arange(0, max(labels), max(labels) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)
    plt.show()

    '''X = values[:,0]
    Y = values[:,1]
    plot_svm_boundary(clf, X, Y, labels)'''


