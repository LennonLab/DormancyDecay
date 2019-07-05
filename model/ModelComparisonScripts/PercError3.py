from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import expanduser


p, fr, _lw, w, fs, sz = 2, 0.75, 0.5, 1, 6, 2
a = 1

mydir = expanduser("~/GitHub/DistDecay")
df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
#df = pd.read_csv(mydir+'/model/ModelData/saved/modelresults.txt')

print 100 * len(df[df['fit'] == 1])/len(df['fit'])


def figplot(fit, y, x, xlab, ylab, fig, n):
    fig.add_subplot(3, 3, n)

    clrs = []
    for i in fit:
        if i == 1: clrs.append('m')
        else: clrs.append('c')

    #x = np.log10(x)
    #y = np.log10(y)

    plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, alpha=a, edgecolor=None)
    plt.xlabel(xlab, fontsize=6)
    plt.ylabel(ylab, fontsize=6)
    plt.tick_params(axis='both', labelsize=6)

    return fig



fig = plt.figure()

ylab = 'Percent error'

xlab = 'active death'
fig = figplot(df['fit'], df['m-mean-Bray'], df['aded']-df['dded'], xlab, ylab, fig, 1)

xlab = 'dormant death'
fig = figplot(df['fit'], df['m-mean-Bray'], df['dded'], xlab, ylab, fig, 2)

xlab = 'active random dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['ad_r']-df['dd_r'], xlab, ylab, fig, 4)

xlab = 'dormant random dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['dd_r'], xlab, ylab, fig, 5)

xlab = 'active spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['ad_s']-df['dd_s'], xlab, ylab, fig, 7)

xlab = 'dormant spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['dd_s'], xlab, ylab, fig, 8)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Params-Bray-2x2.png',
    dpi=200, bbox_inches = "tight")
plt.close()


fig = plt.figure()

xlab = 'active death'
fig = figplot(df['fit'], df['m-mean-Dice'], df['aded']-df['dded'], xlab, ylab, fig, 1)

xlab = 'dormant death'
fig = figplot(df['fit'], df['m-mean-Dice'], df['dded'], xlab, ylab, fig, 2)

xlab = 'active random dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['ad_r']-df['dd_r'], xlab, ylab, fig, 4)

xlab = 'dormant random dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['dd_r'], xlab, ylab, fig, 5)

xlab = 'active spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['ad_s']-df['dd_s'], xlab, ylab, fig, 7)

xlab = 'dormant spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['dd_s'], xlab, ylab, fig, 8)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Params-Sorensen-2x2.png',
    dpi=200, bbox_inches = "tight")
plt.close()



fig = plt.figure()

xlab = 'active death'
fig = figplot(df['fit'], df['m-mean-Canb'], df['aded']-df['dded'], xlab, ylab, fig, 1)

xlab = 'dormant death'
fig = figplot(df['fit'], df['m-mean-Canb'], df['dded'], xlab, ylab, fig, 2)

xlab = 'active random dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['ad_r']-df['dd_r'], xlab, ylab, fig, 4)

xlab = 'dormant random dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['dd_r'], xlab, ylab, fig, 5)

xlab = 'active spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['ad_s']-df['dd_s'], xlab, ylab, fig, 7)

xlab = 'dormant spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['dd_s'], xlab, ylab, fig, 8)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Params-Canberra-2x2.png',
    dpi=200, bbox_inches = "tight")
plt.close()




df = df[df['fit'] == 1]


fig = plt.figure()

ylab = 'Percent error'

xlab = 'active death'
fig = figplot(df['fit'], df['m-mean-Bray'], df['aded']-df['dded'], xlab, ylab, fig, 1)

xlab = 'dormant death'
fig = figplot(df['fit'], df['m-mean-Bray'], df['dded'], xlab, ylab, fig, 2)

xlab = 'active random dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['ad_r']-df['dd_r'], xlab, ylab, fig, 4)

xlab = 'dormant random dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['dd_r'], xlab, ylab, fig, 5)

xlab = 'active spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['ad_s']-df['dd_s'], xlab, ylab, fig, 7)

xlab = 'dormant spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Bray'], df['dd_s'], xlab, ylab, fig, 8)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Params-Bray-2x2-fit.png',
    dpi=200, bbox_inches = "tight")
plt.close()


fig = plt.figure()

xlab = 'active death'
fig = figplot(df['fit'], df['m-mean-Dice'], df['aded']-df['dded'], xlab, ylab, fig, 1)

xlab = 'dormant death'
fig = figplot(df['fit'], df['m-mean-Dice'], df['dded'], xlab, ylab, fig, 2)

xlab = 'active random dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['ad_r']-df['dd_r'], xlab, ylab, fig, 4)

xlab = 'dormant random dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['dd_r'], xlab, ylab, fig, 5)

xlab = 'active spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['ad_s']-df['dd_s'], xlab, ylab, fig, 7)

xlab = 'dormant spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Dice'], df['dd_s'], xlab, ylab, fig, 8)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Params-Sorensen-2x2-fit.png',
    dpi=200, bbox_inches = "tight")
plt.close()



fig = plt.figure()

xlab = 'active death'
fig = figplot(df['fit'], df['m-mean-Canb'], df['aded']-df['dded'], xlab, ylab, fig, 1)

xlab = 'dormant death'
fig = figplot(df['fit'], df['m-mean-Canb'], df['dded'], xlab, ylab, fig, 2)

xlab = 'active random dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['ad_r']-df['dd_r'], xlab, ylab, fig, 4)

xlab = 'dormant random dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['dd_r'], xlab, ylab, fig, 5)

xlab = 'active spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['ad_s']-df['dd_s'], xlab, ylab, fig, 7)

xlab = 'dormant spatial dispersal'
fig = figplot(df['fit'], df['m-mean-Canb'], df['dd_s'], xlab, ylab, fig, 8)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Params-Canberra-2x2-fit.png',
    dpi=200, bbox_inches = "tight")
plt.close()
