from __future__ import division
import sys
import  matplotlib.pyplot as plt
from random import shuffle, sample
import numpy as np
from scipy import stats
from mpl_toolkits.basemap import Basemap
from os.path import expanduser
import scipy as sc
import scipy.spatial.distance


def getXY():
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
    m = Basemap(width=30000, height=30000, projection='lcc', resolution='c',
    lat_1 = None, lat_2 = None, lat_0 = np.mean(lats), lon_0 = np.mean(lons))
    xpt,ypt = m(lons,lats)
    return xpt, ypt


def getEnv():
    pca1 = [0.37338056, -0.54499788,  2.79137336, -0.94759374, 0.28304366,
    1.42451944, 1.30358522, -0.18398612, 2.13160357, 1.46697889, 2.54924319,
    1.09167155, 0.35774751, 2.81465518, -0.30506228, 0.86114584, 5.50983298,
    0.16347067, 2.06112209, 1.21284277, -2.24100634, -3.11508892, -2.27718858,
    -3.48008962, -4.23751359, -3.53990497, -1.19422436, -4.68491624, -2.84324699,
    -4.52427876, -0.21214174, -0.65678703, -5.16903424, 1.65651793, 1.45847875,
    1.04268102, 3.08987537, 0.52105342, 0.77183203, 2.47762923, 0.53534159, -0.99832296,
    -0.52911206, -0.22924320, 2.15798378, 0.29799506, -0.10768611, 1.62756108, -0.01174002]
    return pca1


xs, ys, = getXY()
pca1 = getEnv()

geodif = []
envdif = []

n = len(pca1)
for j in range(n):
    for k in range(j+1, n):

        x1 = xs[j]
        x2 = xs[k]
        y1 = ys[j]
        y2 = ys[k]

        dif = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        geodif.append(dif)

        dif = np.absolute(pca1[j] - pca1[k])
        envdif.append(dif)


envdif = np.array(envdif)
geodif = np.array(geodif)
edif = envdif/max(envdif)
gdif = geodif/max(geodif)

r = []
rp = []
rho = []
rhop = []

sampsize = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
pr = []
pp = []
sr = []
sp = []

for n in sampsize:
    for j in range(1000):
        r_samp = sample(range(len(gdif)), n)
        ed = []
        gd = []

        for i, val in enumerate(r_samp):
            ed.append(edif[val])
            gd.append(gdif[val])

        slope, intercept, r_value, p_value, std_err = stats.linregress(gd, ed)
        r.append(r_value)
        rp.append(p_value)
        ro, p = stats.spearmanr(gd, ed)
        rho.append(ro)
        rhop.append(p)

    pr.append(np.mean(r))
    pp.append(np.mean(rp))
    sr.append(np.mean(rho))
    sp.append(np.mean(rhop))

fig = plt.figure()
ax = fig.add_subplot(3, 3, 1)
plt.scatter(gdif, edif, color = '0.3', alpha= 1 , s = 20, linewidths=0.5, edgecolor='w')
plt.xlabel('Geographic distance', fontsize=8)
plt.ylabel('Environmental distance', fontsize=8)
plt.tick_params(axis='both', labelsize=6)

ax = fig.add_subplot(3, 3, 2)
plt.plot(sampsize, pr, color='0.2', ls='-', lw=2.0)
plt.xlabel('Sample size', fontsize=8)
plt.ylabel('correlation', fontsize=8)
plt.tick_params(axis='both', labelsize=6)

ax = fig.add_subplot(3, 3, 3)
plt.plot(sampsize, pp, color='0.2', ls='-', lw=2.0)
plt.xlabel('Sample size', fontsize=8)
plt.ylabel('p-value', fontsize=8)
plt.tick_params(axis='both', labelsize=6)

mydir = expanduser("~/GitHub/DistDecay/")
path = mydir + "IBM/"

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.45, hspace=0.4)
plt.savefig(mydir + '/figs/FromSims/EnvGeoCorr.png',
    dpi=200, bbox_inches = "tight")
plt.close()
