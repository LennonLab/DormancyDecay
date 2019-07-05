from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from os.path import expanduser
import statsmodels.api as sm


def xfrm(X, _max): return _max-np.array(X)

def figplot(dat, y, x, seed, xlab, ylab, fig, fit, n):

    fitline=False
    b = 2000
    ci = 99
    p, fr, _lw, w, fs, sz = 2, 0.25, 0.5, 1, 6, 3
    a = 1.0

    dat = dat.tolist()
    y = y.tolist()
    x = x.tolist()
    seed = seed.tolist()

    fig.add_subplot(2, 2, n)

    clrs = []

    if fit == 0:
        for i, val in enumerate(dat):
            sd = seed[i]
            clr = str()
            if sd <= 1: clr = 'darkred'
            elif sd < 1.5: clr = 'red'
            elif sd < 2: clr = 'orange'
            elif sd < 2.5: clr = 'yellow'
            elif sd < 3: clr = 'lawngreen'
            elif sd < 3.5: clr = 'green'
            elif sd < 4: clr = 'deepskyblue'
            elif sd < 4.5: clr = 'blue'
            elif sd < 5: clr = 'blueviolet'
            else: clr = 'purple'
            clrs.append(clr)

    elif fit == 1:
        for i, val in enumerate(dat):
            sd = seed[i]
            clr = str()
            if sd <= 2: clr = 'darkred'
            elif sd < 2.4: clr = 'red'
            elif sd < 2.8: clr = 'orange'
            elif sd < 3.2: clr = 'yellow'
            elif sd < 3.6: clr = 'lawngreen'
            elif sd < 4.0: clr = 'green'
            elif sd < 4.4: clr = 'deepskyblue'
            elif sd < 4.8: clr = 'blue'
            elif sd < 5: clr = 'blueviolet'
            else: clr = 'purple'
            clrs.append(clr)

    #if xlab == 'dormant death': x = np.array(x)**10
    plt.scatter(x, y, s = sz, c=clrs, linewidths=0.0, alpha=a, edgecolor=None)

    if fitline:
        x, y = (np.array(t) for t in zip(*sorted(zip(x, y))))

        Xi = xfrm(x, max(x))
        bins = np.linspace(np.min(Xi), np.max(Xi)+1, b)
        ii = np.digitize(Xi, bins)

        pcts = np.array([np.percentile(y[ii==i], ci) for i in range(1, len(bins)) if len(y[ii==i]) > 0])
        xran = np.array([np.mean(x[ii==i]) for i in range(1, len(bins)) if len(y[ii==i]) > 0])

        lowess = sm.nonparametric.lowess(pcts, xran, frac=fr)
        x, y = lowess[:, 0], lowess[:, 1]
        plt.plot(x, y, lw=0.5, color='k')

    plt.xlabel(xlab, fontsize=fs)
    plt.ylabel(ylab, fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    #plt.ylim(min(y), max(y))
    #plt.xlim(min(x), max(x))

    return fig



def figfunction(met2, allfig=True, fitfig=True):

    ws, hs = 0.45, 0.5
    met = 'm'
    mydir = expanduser("~/GitHub/DistDecay")

    if allfig:
        fit = 0
        df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
        fig = plt.figure()

        if met2 == 'perr': ylab = 'Percent error'
        elif met2 == 'pdif': ylab = 'Percent difference'
        elif met2 == 'adif': ylab = 'Difference'

        xlab = 'environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        xlab = 'active spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'dormant spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)


        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Bray-2x2.png',
            dpi=200, bbox_inches = "tight")
        plt.close()


        fig = plt.figure()

        xlab = 'environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        xlab = 'active spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'dormant spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)


        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Sorensen-2x2.png',
            dpi=200, bbox_inches = "tight")
        plt.close()



        fig = plt.figure()

        xlab = 'environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        xlab = 'active spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'dormant spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)



        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Canberra-2x2.png',
            dpi=200, bbox_inches = "tight")
        plt.close()


    if fitfig:
        fit = 1
        df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
        df = df[df['fit'] == 1]

        fig = plt.figure()

        if met2 == 'perr': ylab = 'Percent error'
        elif met2 == 'pdif': ylab = 'Percent difference'
        elif met2 == 'adif': ylab = 'Difference'

        xlab = 'environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        xlab = 'active spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'dormant spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)

        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Bray-2x2-fit.png',
            dpi=200, bbox_inches = "tight")
        plt.close()


        fig = plt.figure()

        xlab = 'environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        xlab = 'active spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'dormant spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)


        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Sorensen-2x2-fit.png',
            dpi=200, bbox_inches = "tight")
        plt.close()



        fig = plt.figure()

        xlab = 'environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        xlab = 'active spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'dormant spatial dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)

        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Canberra-2x2-fit.png',
            dpi=200, bbox_inches = "tight")
        plt.close()



met2 = 'perr'
figfunction(met2)

met2 = 'adif'
figfunction(met2)

met2 = 'pdif'
figfunction(met2)
