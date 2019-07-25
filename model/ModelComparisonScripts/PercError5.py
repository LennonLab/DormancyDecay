from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import sys
from os.path import expanduser
#import statsmodels.api as sm


def xfrm(X, _max): return _max-np.array(X)

def figplot(dat, y, x, seed, xlab, ylab, fig, fit, n):

    fs, sz = 8, 2
    a = 1.0

    e = max(seed)
    dat = dat.tolist()
    y = y.tolist()
    x = x.tolist()
    seed = seed.tolist()

    fig.add_subplot(2, 2, n)

    clrs = []

    if fit >= 0:
        for i, val in enumerate(dat):
            sd = seed[i]
            clr = str()
            if sd <= e*0.1: clr = 'darkred'
            elif sd < e*0.2: clr = 'red'
            elif sd < e*0.3: clr = 'orange'
            elif sd < e*0.4: clr = 'yellow'
            elif sd < e*0.5: clr = 'lawngreen'
            elif sd < e*0.6: clr = 'green'
            elif sd < e*0.7: clr = 'deepskyblue'
            elif sd < e*0.8: clr = 'blue'
            elif sd < e*0.9: clr = 'blueviolet'
            else: clr = 'purple'
            clrs.append(clr)

    elif fit == 1:
        for i, val in enumerate(dat):
            sd = seed[i]
            clr = str()
            if sd <= e*0.1: clr = 'darkred'
            elif sd < e*0.2: clr = 'red'
            elif sd < e*0.3: clr = 'orange'
            elif sd < e*0.4: clr = 'yellow'
            elif sd < e*0.5: clr = 'lawngreen'
            elif sd < e*0.6: clr = 'green'
            elif sd < e*0.7: clr = 'deepskyblue'
            elif sd < e*0.8: clr = 'blue'
            elif sd < e*0.9: clr = 'blueviolet'
            else: clr = 'purple'
            clrs.append(clr)

    #if xlab == 'Dormant death': x = np.array(x)**5
    plt.scatter(x, y, s = sz, c=clrs, linewidths=0.0, alpha=a, edgecolor=None)

    '''
    if fitline:
        fr = 0.25
        fitline = False
        b = 2000
        ci = 99
        x1, y1 = (np.array(t) for t in zip(*sorted(zip(x, y))))

        Xi = xfrm(x1, max(x1))
        bins = np.linspace(np.min(Xi), np.max(Xi)+1, b)
        ii = np.digitize(Xi, bins)

        pcts = np.array([np.percentile(y[ii==i], ci) for i in range(1, len(bins)) if len(y[ii==i]) > 0])
        xran = np.array([np.mean(x[ii==i]) for i in range(1, len(bins)) if len(y[ii==i]) > 0])

        lowess = sm.nonparametric.lowess(pcts, xran, frac=fr)
        x1, y1 = lowess[:, 0], lowess[:, 1]
        plt.plot(x1, y1, lw=0.5, color='k')
    '''

    #if xlab == 'Active dispersal' or xlab == 'Dormant dispersal': 
    #    plt.xscale('log')
    #    plt.xlim(min(x),1)
    
    #plt.ylim(min(y), max(y))
    plt.xlabel(xlab, fontsize=fs+2)
    plt.ylabel(ylab, fontsize=fs+2)
    plt.tick_params(axis='both', labelsize=fs)

    return fig



def figfunction(met2, allfig=False, fitfig=True):

    ws, hs = 0.4, 0.4
    met = 'm'
    mydir = expanduser("~/GitHub/DistDecay")

    if allfig:
        fit = 0
        #df = pd.read_csv(mydir+'/model/ModelData/saved/modelresults_nodispersal.txt')
        df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
        #print 'shape', df.shape
        df = df[df['fit'] == 0]
        #print df.shape
        #sys.exit()
        
        fig = plt.figure()
        #plt.style.use('classic')

        if met2 == 'perr': ylab = 'Percent error'
        elif met2 == 'pdif': ylab = 'Percent difference'
        elif met2 == 'adif': ylab = 'Difference'

        xlab = 'Environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'Dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        

        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Bray-2x2.png',
            dpi=400, bbox_inches = "tight")
        plt.close()


        fig = plt.figure()

        xlab = 'Environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'Dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        

        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Sorensen-2x2.png',
            dpi=400, bbox_inches = "tight")
        plt.close()



        fig = plt.figure()

        xlab = 'Environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'Dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)
        
        
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        

        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Canberra-2x2.png',
            dpi=400, bbox_inches = "tight")
        plt.close()


    if fitfig:
        fit = 1
        df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
        df = df[df['fit'] == 1]

        fig = plt.figure()

        if met2 == 'perr': ylab = 'Percent error'
        elif met2 == 'pdif': ylab = 'Percent difference'
        elif met2 == 'adif': ylab = 'Difference'

        xlab = 'Environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'Dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Bray-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        
        
        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Bray-2x2-fit.png',
            dpi=400, bbox_inches = "tight")
        plt.close()


        fig = plt.figure()

        xlab = 'Environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'Dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)
        
        
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Dice-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        

        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Sorensen-2x2-fit.png',
            dpi=400, bbox_inches = "tight")
        plt.close()



        fig = plt.figure()

        xlab = 'Environmental filtering'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

        xlab = 'Dormant death'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

        
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], df[met+'-mean-Canb-'+met2], df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        
        
        #### Final Format and Save #####################################################
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/'+met2+'-Canberra-2x2-fit.png',
            dpi=400, bbox_inches = "tight")
        plt.close()



met2 = 'perr'
figfunction(met2)

#met2 = 'adif'
#figfunction(met2)

#met2 = 'pdif'
#figfunction(met2)
