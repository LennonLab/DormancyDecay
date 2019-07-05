from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from os.path import expanduser
import statsmodels.api as sm


def figfunction(met2):

    def xfrm(X, _max): return _max-np.array(X)

    met = 'm'
    b = 20
    ci = 99
    p, fr, _lw, w, fs, sz = 2, 0.75, 0.5, 1, 6, 2
    a = 1.0

    mydir = expanduser("~/GitHub/DistDecay")
    df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
    #df = pd.read_csv(mydir+'/model/ModelData/saved/modelresults.txt')

    print 100 * len(df[df['fit'] == 1])/len(df['fit'])
    df = df[df['fit'] == 0]

    #df = df[df[met+'-mean-Bray'] < 200]
    #df = df[df[met+'-mean-Dice'] < 200]
    #df = df[df[met+'-mean-Canb'] < 200]

    def figplot(fit, y, x, xlab, ylab, fig, n):

        #x = np.log10(x)
        fig.add_subplot(3, 3, n)

        clrs = []
        for i in fit:
            if i == 1: clrs.append('m')
            else: clrs.append('c')

        plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, alpha=a, edgecolor=None)

        '''
        x, y = (np.array(t) for t in zip(*sorted(zip(x, y))))

        Xi = xfrm(x, max(x))
        bins = np.linspace(np.min(Xi), np.max(Xi)+1, b)
        ii = np.digitize(Xi, bins)

        pcts = np.array([np.percentile(y[ii==i], ci) for i in range(1, len(bins)) if len(y[ii==i]) > 0])
        xran = np.array([np.mean(x[ii==i]) for i in range(1, len(bins)) if len(y[ii==i]) > 0])

        lowess = sm.nonparametric.lowess(pcts, xran, frac=fr)
        x, y = lowess[:, 0], lowess[:, 1]
        plt.plot(x, y, lw=1, color='k')
        '''


        plt.xlabel(xlab, fontsize=6)
        plt.ylabel(ylab, fontsize=6)
        plt.tick_params(axis='both', labelsize=6)

        return fig


    '''
    fig = plt.figure()

    ylab = 'Percent error'

    xlab = 'active death'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['aded'], xlab, ylab, fig, 1)

    xlab = 'dormant death'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dded'], xlab, ylab, fig, 2)

    xlab = 'active random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_r'], xlab, ylab, fig, 4)

    xlab = 'dormant random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_r'], xlab, ylab, fig, 5)

    xlab = 'active spatial dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_s'], xlab, ylab, fig, 7)

    xlab = 'dormant spatial dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_s'], xlab, ylab, fig, 8)


    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Bray-2x2.png',
        dpi=200, bbox_inches = "tight")
    plt.close()


    fig = plt.figure()

    xlab = 'active death'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['aded'], xlab, ylab, fig, 1)

    xlab = 'dormant death'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dded'], xlab, ylab, fig, 2)

    xlab = 'active random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_r'], xlab, ylab, fig, 4)

    xlab = 'dormant random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_r'], xlab, ylab, fig, 5)

    xlab = 'active spatial dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_s'], xlab, ylab, fig, 7)

    xlab = 'dormant spatial dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_s'], xlab, ylab, fig, 8)


    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Sorensen-2x2.png',
        dpi=200, bbox_inches = "tight")
    plt.close()



    fig = plt.figure()

    xlab = 'active death'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['aded'], xlab, ylab, fig, 1)

    xlab = 'dormant death'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dded'], xlab, ylab, fig, 2)

    xlab = 'active random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_r'], xlab, ylab, fig, 4)

    xlab = 'dormant random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_r'], xlab, ylab, fig, 5)

    xlab = 'active spatial dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_s'], xlab, ylab, fig, 7)

    xlab = 'dormant spatial dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_s'], xlab, ylab, fig, 8)


    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Canberra-2x2.png',
        dpi=200, bbox_inches = "tight")
    plt.close()
    '''


    df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
    #df = pd.read_csv(mydir+'/model/ModelData/saved/modelresults.txt')
    df = df[df['fit'] == 1]


    fig = plt.figure()

    ylab = 'Percent error'

    #xlab = 'active death'
    #fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['aded'], xlab, ylab, fig, 1)

    xlab = 'dormant death'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dded'], xlab, ylab, fig, 1)

    xlab = 'active random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_r'], xlab, ylab, fig, 2)

    xlab = 'dormant random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_r'], xlab, ylab, fig, 3)

    #xlab = 'active spatial dispersal'
    #fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_s'], xlab, ylab, fig, 7)

    #xlab = 'dormant spatial dispersal'
    #fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_s'], xlab, ylab, fig, 8)


    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Bray-2x2-fit.png',
        dpi=200, bbox_inches = "tight")
    plt.close()


    fig = plt.figure()

    #xlab = 'active death'
    #fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['aded'], xlab, ylab, fig, 1)

    xlab = 'dormant death'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dded'], xlab, ylab, fig, 1)

    xlab = 'active random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_r'], xlab, ylab, fig, 2)

    xlab = 'dormant random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_r'], xlab, ylab, fig, 3)

    #xlab = 'active spatial dispersal'
    #fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_s'], xlab, ylab, fig, 7)

    #xlab = 'dormant spatial dispersal'
    #fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_s'], xlab, ylab, fig, 8)


    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Sorensen-2x2-fit.png',
        dpi=200, bbox_inches = "tight")
    plt.close()



    fig = plt.figure()

    #xlab = 'active death'
    #fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['aded'], xlab, ylab, fig, 1)

    xlab = 'dormant death'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dded'], xlab, ylab, fig, 1)

    xlab = 'active random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_r'], xlab, ylab, fig, 2)

    xlab = 'dormant random dispersal'
    fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_r'], xlab, ylab, fig, 3)

    #xlab = 'active spatial dispersal'
    #fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_s'], xlab, ylab, fig, 7)

    #xlab = 'dormant spatial dispersal'
    #fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_s'], xlab, ylab, fig, 8)


    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Canberra-2x2-fit.png',
        dpi=200, bbox_inches = "tight")
    plt.close()


met2 = 'perr'
figfunction(met2)

met2 = 'pdif'
figfunction(met2)
