from __future__ import division
import numpy as np
from mpl_toolkits.basemap import Basemap
import  matplotlib.pyplot as plt
from numpy.random import uniform
import os
import sys

mydir = os.path.expanduser('~/GitHub/DistDecay/model/')
sys.path.append(mydir)
#from bide import bide
from bide_iter import bide



def getXY(S):
    lats = [39.12153, 39.16358, 39.15219, 39.14453, 39.14850, 39.13319, 39.12753,
    39.12389, 39.12781, 39.13289, 39.14017, 39.14211, 39.13558, 39.13306, 39.17614,
    39.16792, 39.17525, 39.03828, 39.04647, 39.04306, 39.05306, 39.03511, 38.99197,
    39.01184, 39.02426, 39.02142, 39.03236, 39.04264, 39.04942, 39.00874, 39.02928,
    39.00381, 39.09983, 39.12428, 39.13302, 39.13442, 39.13158, 39.12840, 39.12552,
    39.10983, 39.11367, 39.11690, 39.14217, 39.13778, 39.12981, 39.12757, 39.12328,
    39.11077, 39.31186]
    lons = [-86.19458, -86.21181, -86.19418, -86.19269, -86.19017, -86.19683,
    -86.19928, -86.20672, -86.20983, -86.27564, -86.27589, -86.27064, -86.26500,
    -86.25761, -86.20572, -86.20389, -86.20944, -86.20919, -86.21550, -86.21444,
    -86.21864, -86.32964, -86.38756, -86.40976, -86.31246, -86.31678, -86.30386,
    -86.31614, -86.31747, -86.30509, -86.31967, -86.30583, -86.30682, -86.28244,
    -86.30795, -86.28058, -86.29511, -86.32757, -86.34250, -86.29811, -86.29333,
    -86.28442, -86.28608, -86.29753, -86.28333, -86.28623, -86.28867, -86.28670,
    -86.29028]

    # setup Lambert Conformal basemap
    m = Basemap(width=30000, height=30000, projection='lcc',
        lat_0 = np.mean(lats), lon_0 = np.mean(lons))
    xpt,ypt = m(lons,lats)
    xpt = np.tile(np.array([xpt]).transpose(), (1, S))
    ypt = np.tile(np.array([ypt]).transpose(), (1, S))
    return xpt, ypt



def getEnv(S):

    pca = [-0.481839820346883, -0.628206891741771, 2.83762050151573, -1.26374942837071,
    -0.32516391637875, 1.31232332816171, 1.56089153500539, -0.656228978882647,
    2.23624753387512, 1.43663156211996, 2.97468535247928, 1.63415637773399,
    0.930430720241175, 2.83300066438896, -0.456102014662252, 0.900498693050348,
    6.29218458887682, 0.5485980293433, 1.9302288220143, 1.61045435624923,
    -2.85423016392321, -2.62727153712451, -2.0605095235643, -3.38770055819569,
    -4.30687904691204, -3.51773044842165, -0.853663064487916, -4.84778412594482,
    -2.27368791816158, -4.74875042850953, 0.348926598450966, -0.492855592251113,
    -5.16031776701625, 1.91148035663689, 1.76043973004349, 0.833688153260464,
    2.83468686837334, 0.258410805274002, 0.497759425506841, 2.58404084661871,
    0.479332908820121, -1.34183007293342, -1.40513941453882, -0.470470196199752,
    2.25960128908896, -0.121466334978157, -0.242531823296678, 1.77016932880303,
    -0.0523793090896964]

    pca = np.tile(np.array([pca]).transpose(), (1, S))
    return pca



dd_s = uniform(0, 1)
ad_s = uniform(0, 1)
dd_r = uniform(0, 1)
ad_r = uniform(0, 1)
aded = uniform(0, 1)
dded = uniform(0, 1)
seed = uniform(0, 1)

dmax = 0.0
dampen = uniform(0, dmax)

S = 100
xs, ys, = getXY(S)
pca = getEnv(S)

env_r = 1 #uniform(1, 10)
env = uniform(np.min(pca) * env_r, env_r * np.max(pca), S)
env = np.array([env,] * 49)

Act, Dor, avgMatch, aNs, dNs, allNs = bide(env, pca, xs, ys, S, dd_r, ad_r, dd_s, ad_s, aded, dded, dampen, seed)
All = Dor + Act


fig = plt.figure()
fig.add_subplot(1, 1, 1)

#print aNs
x = range(len(aNs))

plt.plot(x, aNs, color='m')
plt.plot(x, dNs, color='k')
plt.plot(x, allNs, color='c', ls='--')

#plt.yscale('log')
#plt.xscale('log')

plt.tick_params(axis='both', labelsize=6)
plt.show()