from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from os.path import expanduser
import statsmodels.api as sm
from scipy.stats.kde import gaussian_kde


def get_kdens_choose_kernel(_list,kernel):
    """ Finds the kernel density function across a sample of SADs """
    density = gaussian_kde(_list)
    n = len(_list)
    xs = np.linspace(min(_list), max(_list), n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D


def figplot(x, xpt, xlab, ylab, fig, n):

        fs = 8
        x = x.tolist()

        fig.add_subplot(2, 2, n)

        D = get_kdens_choose_kernel(x, 0.5)
        plt.plot(D[0],D[1],color = 'k', lw=2, alpha = 0.99)

        plt.xlabel(xlab, fontsize=fs)
        plt.ylabel(ylab, fontsize=fs)
        plt.tick_params(axis='both', labelsize=fs)
        plt.axvline(xpt, color='k', ls=':', lw=1)

        return fig


def figfunction():

    p, fr, _lw, w, fs, sz = 2, 0.25, 0.5, 1, 6, 4
    ws, hs = 0.45, 0.5

    mydir = expanduser("~/GitHub/DistDecay")

    mets = ['Bray', 'Dice', 'Canb']
    fits = [0, 1]
    for fit in fits:
        for met in mets:

            if met == 'Bray': xpt = [-0.325, -0.248, -0.182, -0.132]
            elif met == 'Dice': xpt = [-0.101, -0.054, -0.052, -0.032]
            elif met == 'Canb': xpt = [-0.054, -0.029, -0.028, -0.016]

            df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
            if fit == 1: df = df[df['fit'] == fit]

            fig = plt.figure()
            ylab = 'kernel density'

            xlab = 'slope, Act-Env'
            fig = figplot(df[met+'-ActEnv-m'], xpt[0], xlab, ylab, fig, 1)

            xlab = 'slope, All-Env'
            fig = figplot(df[met+'-AllEnv-m'], xpt[1], xlab, ylab, fig, 2)

            xlab = 'slope, Act-Geo'
            fig = figplot(df[met+'-ActGeo-m'], xpt[2], xlab, ylab, fig, 3)

            xlab = 'slope, All-Geo'
            fig = figplot(df[met+'-AllGeo-m'], xpt[3], xlab, ylab, fig, 4)


            #### Final Format and Save #####################################################
            plt.subplots_adjust(wspace=ws, hspace=hs)
            if fit == 0: plt.savefig(mydir+'/figs/FromSims/temp/'+met+'-kdens.png', dpi=200, bbox_inches = "tight")
            elif fit == 1: plt.savefig(mydir+'/figs/FromSims/temp/'+met+'-kdens-fit.png', dpi=200, bbox_inches = "tight")
            plt.close()



figfunction()
