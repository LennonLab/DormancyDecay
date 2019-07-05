from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from os.path import expanduser


def assigncolor(xs):
    xs = np.array(xs)/max(xs)
    clrs = []

    for x in xs:
        c = 0
        if x <= 0.04: c = 'crimson'
        elif x <= 0.05: c = 'r'
        elif x <= 0.06: c = 'orangered'
        elif x <= 0.07: c = 'darkorange'
        elif x <= 0.08: c = 'orange'
        elif x <= 0.09: c = 'gold'

        elif x <= 0.1: c = 'yellow'
        elif x <= 0.2: c = 'greenyellow'
        elif x <= 0.3: c = 'springgreen'
        elif x <= 0.4: c = 'Green'
        elif x <= 0.5: c = 'skyblue'
        elif x <= 0.6: c = 'DodgerBlue'
        elif x <= 0.7: c = 'b'
        elif x <= 0.8: c = 'Plum'
        elif x <= 0.9: c = 'violet'
        else: c = 'purple'

        clrs.append(c)
    return clrs


p, fr, _lw, w, fs, sz2 = 2, 0.75, 0.5, 1, 6, 16
sz, a = 2, 0.6
scl = 'lin'

mydir = expanduser("~/GitHub/DistDecay")
df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
#df = pd.read_csv(mydir+'/model/ModelData/saved/modelresults.txt')
df = df[df['fit'] == 1]

clrs = assigncolor(df['m-mean-Dice-perr'])

fig = plt.figure()
fig.add_subplot(3, 3, 1)

if scl == 'lin':
    x = df['aded']
    y = df['ad_r']
else:
    x = np.log10(df['aded'])
    y = np.log10(df['ad_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, active', fontsize=fs)
plt.ylabel('Random dispersal\nrate, active', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 2)
plt.scatter([1], [1], color='crimson', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.04$')
plt.scatter([1], [1], color='r', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.05$')
plt.scatter([1], [1], color='orangered', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.06$')
plt.scatter([1], [1], color='darkorange', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.07$')
plt.scatter([1], [1], color='orange', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.08$')
plt.scatter([1], [1], color='gold', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.09$')
plt.scatter([1], [1], color='yellow', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.1$')
plt.scatter([1], [1], color='greenyellow', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.2$')
plt.scatter([1], [1], color='springgreen', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.3$')
plt.scatter([1], [1], color='Green', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.4$')
plt.scatter([1], [1], color='skyblue', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.5$')
plt.scatter([1], [1], color='DodgerBlue', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.6$')
plt.scatter([1], [1], color='b', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.7$')
plt.scatter([1], [1], color='Plum', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.8$')
plt.scatter([1], [1], color='violet', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.9$')
plt.scatter([1], [1], color='purple', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 1.0$')

plt.ylim(2,3)
plt.xlim(2,3)
plt.xticks([])
plt.yticks([])
plt.box(on=None)
plt.legend(loc=2, fontsize=fs, frameon=False)



fig.add_subplot(3, 3, 4)

if scl == 'lin':
    x = df['dded']
    y = df['dd_r']
else:
    x = np.log10(df['dded'])
    y = np.log10(df['dd_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, dormant', fontsize=fs)
plt.ylabel('Random dispersal\nrate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 7)

if scl == 'lin':
    x = df['ad_r']
    y = df['dd_r']
else:
    x = np.log10(df['dd_r'])
    y = np.log10(df['ad_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Random dispersal\nrate, active', fontsize=fs)
plt.ylabel('Random dispersal\nrate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 8)

if scl == 'lin':
    x = df['aded']
    y = df['dded']
else:
    x = np.log10(df['aded'])
    y = np.log10(df['dded'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, active', fontsize=fs)
plt.ylabel('Death rate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Heat_Dice.png',
    dpi=200, bbox_inches = "tight")
plt.close()







clrs = assigncolor(df['m-mean-Bray-perr'])

fig = plt.figure()
fig.add_subplot(3, 3, 1)

if scl == 'lin':
    x = df['aded']
    y = df['ad_r']
else:
    x = np.log10(df['aded'])
    y = np.log10(df['ad_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, active', fontsize=fs)
plt.ylabel('Random dispersal\nrate, active', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 2)
plt.scatter([1], [1], color='crimson', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.04$')
plt.scatter([1], [1], color='r', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.05$')
plt.scatter([1], [1], color='orangered', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.06$')
plt.scatter([1], [1], color='darkorange', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.07$')
plt.scatter([1], [1], color='orange', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.08$')
plt.scatter([1], [1], color='gold', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.09$')
plt.scatter([1], [1], color='yellow', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.1$')
plt.scatter([1], [1], color='greenyellow', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.2$')
plt.scatter([1], [1], color='springgreen', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.3$')
plt.scatter([1], [1], color='Green', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.4$')
plt.scatter([1], [1], color='skyblue', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.5$')
plt.scatter([1], [1], color='DodgerBlue', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.6$')
plt.scatter([1], [1], color='b', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.7$')
plt.scatter([1], [1], color='Plum', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.8$')
plt.scatter([1], [1], color='violet', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.9$')
plt.scatter([1], [1], color='purple', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 1.0$')

plt.ylim(2,3)
plt.xlim(2,3)
plt.xticks([])
plt.yticks([])
plt.box(on=None)
plt.legend(loc=2, fontsize=fs, frameon=False)



fig.add_subplot(3, 3, 4)

if scl == 'lin':
    x = df['dded']
    y = df['dd_r']
else:
    x = np.log10(df['dded'])
    y = np.log10(df['dd_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, dormant', fontsize=fs)
plt.ylabel('Random dispersal\nrate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 7)

if scl == 'lin':
    x = df['ad_r']
    y = df['dd_r']
else:
    x = np.log10(df['dd_r'])
    y = np.log10(df['ad_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Random dispersal\nrate, active', fontsize=fs)
plt.ylabel('Random dispersal\nrate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 8)

if scl == 'lin':
    x = df['aded']
    y = df['dded']
else:
    x = np.log10(df['aded'])
    y = np.log10(df['dded'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, active', fontsize=fs)
plt.ylabel('Death rate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Heat_Bray.png',
    dpi=200, bbox_inches = "tight")
plt.close()





clrs = assigncolor(df['m-mean-Canb-perr'])

fig = plt.figure()
fig.add_subplot(3, 3, 1)

if scl == 'lin':
    x = df['aded']
    y = df['ad_r']
else:
    x = np.log10(df['aded'])
    y = np.log10(df['ad_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, active', fontsize=fs)
plt.ylabel('Random dispersal\nrate, active', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 2)
plt.scatter([1], [1], color='crimson', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.04$')
plt.scatter([1], [1], color='r', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.05$')
plt.scatter([1], [1], color='orangered', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.06$')
plt.scatter([1], [1], color='darkorange', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.07$')
plt.scatter([1], [1], color='orange', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.08$')
plt.scatter([1], [1], color='gold', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.09$')
plt.scatter([1], [1], color='yellow', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.1$')
plt.scatter([1], [1], color='greenyellow', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.2$')
plt.scatter([1], [1], color='springgreen', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.3$')
plt.scatter([1], [1], color='Green', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.4$')
plt.scatter([1], [1], color='skyblue', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.5$')
plt.scatter([1], [1], color='DodgerBlue', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.6$')
plt.scatter([1], [1], color='b', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.7$')
plt.scatter([1], [1], color='Plum', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.8$')
plt.scatter([1], [1], color='violet', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 0.9$')
plt.scatter([1], [1], color='purple', s=sz2, linewidths=0.0, edgecolor=None, label=r'$ < 1.0$')

plt.ylim(2,3)
plt.xlim(2,3)
plt.xticks([])
plt.yticks([])
plt.box(on=None)
plt.legend(loc=2, fontsize=fs, frameon=False)



fig.add_subplot(3, 3, 4)

if scl == 'lin':
    x = df['dded']
    y = df['dd_r']
else:
    x = np.log10(df['dded'])
    y = np.log10(df['dd_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, dormant', fontsize=fs)
plt.ylabel('Random dispersal\nrate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 7)

if scl == 'lin':
    x = df['ad_r']
    y = df['dd_r']
else:
    x = np.log10(df['dd_r'])
    y = np.log10(df['ad_r'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Random dispersal\nrate, active', fontsize=fs)
plt.ylabel('Random dispersal\nrate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(3, 3, 8)

if scl == 'lin':
    x = df['aded']
    y = df['dded']
else:
    x = np.log10(df['aded'])
    y = np.log10(df['dded'])

plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None, alpha=a)
plt.xlabel('Death rate, active', fontsize=fs)
plt.ylabel('Death rate, dormant', fontsize=fs)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.tick_params(axis='both', labelsize=fs-4)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/FromSims/temp/Heat_Canb.png',
    dpi=200, bbox_inches = "tight")
plt.close()
