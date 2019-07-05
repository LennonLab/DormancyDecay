from __future__ import division
import numpy as np
from scipy import stats
from mpl_toolkits.basemap import Basemap
from numpy.random import uniform
from numpy import where
from random import choice
import scipy as sc
import os
import sys

mydir = os.path.expanduser('~/GitHub/DistDecay/model/')
sys.path.append(mydir)
from bide import bide
#from bide_iter import bide


def per_dif(obs, exp):
    return np.abs(obs - exp)
    #obs = np.abs(obs)
    #exp = np.abs(exp)
    #return 100 * np.abs(obs-exp)/np.abs(np.mean([obs, exp]))

def per_err(obs, exp):
    return np.abs(obs - exp)
    #return 100 * np.abs(obs-exp)/np.abs(exp)


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



col_headers = 'Sim,S,Sall,Sact,fit,dd_r,ad_r,dd_s,ad_s,aded,dded,dampen,seed,'
col_headers += 'minAct,avgAct,maxAct,minAll,avgAll,maxAll,'
col_headers += 'm-mean-Bray-perr,b-mean-Bray-perr,t-mean-Bray-perr,'
col_headers += 'm-mean-Bray-pdif,b-mean-Bray-pdif,t-mean-Bray-pdif,'
col_headers += 'm-mean-Dice-perr,b-mean-Dice-perr,t-mean-Dice-perr,'
col_headers += 'm-mean-Dice-pdif,b-mean-Dice-pdif,t-mean-Dice-pdif,'
col_headers += 'm-mean-Canb-perr,b-mean-Canb-perr,t-mean-Canb-perr,'
col_headers += 'm-mean-Canb-pdif,b-mean-Canb-pdif,t-mean-Canb-pdif,'


mydir = os.path.expanduser("~/GitHub/DistDecay/model/ModelData")
OUT = open(mydir+'/modelresults.txt','w+')
print>>OUT, col_headers
OUT.close()


Tfit1 = 0
Tfit2 = 0
ints = True

for i1 in range(0, 100000):

    dd_s = uniform(0, 1)
    ad_s = uniform(0, 1)
    dd_r = uniform(0, 1)
    ad_r = uniform(0, 1)
    aded = uniform(0, 1)
    dded = uniform(0, 1)
    dmax = 6
    dampen = uniform(0, dmax)
    seed = uniform(0.1, 1.0)

    S = 1000
    xs, ys, = getXY(S)
    pca = getEnv(S)

    env = uniform(np.min(pca) * 40, 40 * np.max(pca), S)
    env = np.array([env,] * 49)

    Act, Dor = bide(env, pca, xs, ys, S, dd_r, ad_r, dd_s, ad_s, aded, dded, dampen, seed)
    All = Dor + Act

    r1list = where(~Act.any(axis=0))[0]
    r1list = r1list.tolist()
    r2list = where(~All.any(axis=0))[0]
    r2list = r2list.tolist()
    rlist = list(set(r1list + r2list))

    c1list = where(~Act.any(axis=1))[0]
    c1list = c1list.tolist()
    c2list = where(~All.any(axis=1))[0]
    c2list = c2list.tolist()
    clist = list(set(c1list + c2list))

    Act = np.delete(Act, rlist, 0)
    Act = np.delete(Act, clist, 1)
    All = np.delete(All, rlist, 0)
    All = np.delete(All, clist, 1)

    xs = np.delete(xs, rlist, 0)
    xs = np.delete(xs, clist, 1)
    ys = np.delete(ys, rlist, 0)
    ys = np.delete(ys, clist, 1)

    pca = np.delete(pca, rlist, 0)
    pca = np.delete(pca, clist, 1)

    n = pca.shape[0]
    if n < 10 or pca.shape[1] < 10: continue

    minAct = int(round(np.min(Act)))
    avgAct = int(round(np.mean(Act)))
    maxAct = int(round(np.max(Act)))

    minAll = int(round(np.min(All)))
    avgAll = int(round(np.mean(All)))
    maxAll = int(round(np.max(All)))

    Sall = []
    for r in range(n):
        S = np.count_nonzero(All[r])
        Sall.append(S)
    Sall = int(round(np.mean(Sall)))

    Sact = []
    for r in range(n):
        S = np.count_nonzero(Act[r])
        Sact.append(S)
    Sact = int(round(np.mean(Sact)))


    Act = Act/Act.sum(axis=1)[:, None]
    All = All/All.sum(axis=1)[:, None]

    envdif = []
    geodif = []
    actdif1 = []
    alldif1 = []
    actdif2 = []
    alldif2 = []
    actdif3 = []
    alldif3 = []

    for j in range(n):
        for k in range(j+1, n):

            sim1 = 1 - sc.spatial.distance.braycurtis(Act[j], Act[k])
            sim2 = 1 - sc.spatial.distance.braycurtis(All[j], All[k])
            actdif1.append(sim1)
            alldif1.append(sim2)

            actj = where(Act[j] > 0, 1, 0)
            actk = where(Act[k] > 0, 1, 0)
            allj = where(All[j] > 0, 1, 0)
            allk = where(All[k] > 0, 1, 0)

            sim3 = 1 - sc.spatial.distance.braycurtis(actj, actk)
            sim4 = 1 - sc.spatial.distance.braycurtis(allj, allk)
            actdif2.append(sim3)
            alldif2.append(sim4)

            l = len(Act[j])
            sim5 = 1 - sc.spatial.distance.canberra(Act[j], Act[k])/l
            sim6 = 1 - sc.spatial.distance.canberra(All[j], All[k])/l
            actdif3.append(sim5)
            alldif3.append(sim6)


            x1 = xs[j][0]
            x2 = xs[k][0]
            y1 = ys[j][0]
            y2 = ys[k][0]

            dif = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            geodif.append(dif)

            dif = np.absolute(pca[j][0] - pca[k][0])
            envdif.append(dif)


    envdif = np.array(envdif)
    geodif = np.array(geodif)
    envdif = envdif/np.max(envdif)
    geodif = geodif/np.max(geodif)


    EnvAct_slope1, EnvAct_int1, r_value, p_value, std_err = stats.linregress(envdif, actdif1)
    EnvAll_slope1, EnvAll_int1, r_value, p_value, std_err = stats.linregress(envdif, alldif1)
    GeoAct_slope1, GeoAct_int1, r_value, p_value, std_err = stats.linregress(geodif, actdif1)
    GeoAll_slope1, GeoAll_int1, r_value, p_value, std_err = stats.linregress(geodif, alldif1)

    EnvAct_slope2, EnvAct_int2, r_value, p_value, std_err = stats.linregress(envdif, actdif2)
    EnvAll_slope2, EnvAll_int2, r_value, p_value, std_err = stats.linregress(envdif, alldif2)
    GeoAct_slope2, GeoAct_int2, r_value, p_value, std_err = stats.linregress(geodif, actdif2)
    GeoAll_slope2, GeoAll_int2, r_value, p_value, std_err = stats.linregress(geodif, alldif2)

    EnvAct_slope3, EnvAct_int3, r_value, p_value, std_err = stats.linregress(envdif, actdif3)
    EnvAll_slope3, EnvAll_int3, r_value, p_value, std_err = stats.linregress(envdif, alldif3)
    GeoAct_slope3, GeoAct_int3, r_value, p_value, std_err = stats.linregress(geodif, actdif3)
    GeoAll_slope3, GeoAll_int3, r_value, p_value, std_err = stats.linregress(geodif, alldif3)


    fit = 0
    fits = []
    mls = [EnvAct_slope1, EnvAll_slope1, EnvAct_slope2, EnvAll_slope2, EnvAct_slope3, EnvAll_slope3]
    if max(mls) < 0: fits.append(True)
    else: fits.append(False)

    nls = np.isnan(mls).tolist()
    if nls.count(True) == 0: fits.append(True)
    else: fits.append(False)


    if EnvAct_slope1 < EnvAll_slope1: fits.append(True)
    else: fits.append(False)
    if EnvAct_slope1 < GeoAct_slope1: fits.append(True)
    else: fits.append(False)


    if EnvAct_slope2 < EnvAll_slope2: fits.append(True)
    else: fits.append(False)
    if EnvAct_slope2 < GeoAct_slope2: fits.append(True)
    else: fits.append(False)

    if ints:
        if EnvAct_int2 > EnvAll_int2: fits.append(True)
        else: fits.append(False)
        if EnvAct_int2 > GeoAct_int2: fits.append(True)
        else: fits.append(False)


    if EnvAct_slope3 < EnvAll_slope3: fits.append(True)
    else: fits.append(False)
    if EnvAct_slope3 < GeoAct_slope3: fits.append(True)
    else: fits.append(False)

    if ints:
        if EnvAct_int3 > EnvAll_int3: fits.append(True)
        else: fits.append(False)
        if EnvAct_int3 > GeoAct_int3: fits.append(True)
        else: fits.append(False)


    if GeoAct_slope1 < GeoAll_slope1: fits.append(True)
    else: fits.append(False)
    if GeoAct_slope2 < GeoAll_slope2: fits.append(True)
    else: fits.append(False)
    if GeoAct_slope3 < GeoAll_slope3: fits.append(True)
    else: fits.append(False)

    if fits.count(False) == 0:
        fit = 1
        Tfit1 += 1
        Tfit2 += 1

    outlist = [i1, S, Sall, Sact, fit, dd_r, ad_r, dd_s, ad_s, aded, dded, dmax-dampen,
        seed, minAct, avgAct, maxAct, minAll, avgAll, maxAll]

    e_actslope1 = per_err(EnvAct_slope1, -0.325)

    e_allslope1 = per_err(EnvAll_slope1, -0.248)

    g_actslope1 = per_err(GeoAct_slope1, -0.182)

    g_allslope1 = per_err(GeoAll_slope1, -0.132)

    e_actint1 = per_err(EnvAct_int1, 0.434)

    e_allint1 = per_err(EnvAll_int1, 0.437)

    g_actint1 = per_err(GeoAct_int1, 0.422)

    g_allint1 = per_err(GeoAll_int1, 0.426)


    slopetotmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1]), 5)
    inttotmean1 = round(np.mean([e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    totmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1,
        e_actint1, e_allint1, g_actint1, g_allint1]), 5)

    outlist.extend([slopetotmean1, inttotmean1, totmean1])



    e_actslope1 = per_dif(EnvAct_slope1, -0.325)

    e_allslope1 = per_dif(EnvAll_slope1, -0.248)

    g_actslope1 = per_dif(GeoAct_slope1, -0.182)

    g_allslope1 = per_dif(GeoAll_slope1, -0.132)

    e_actint1 = per_dif(EnvAct_int1, 0.434)

    e_allint1 = per_dif(EnvAll_int1, 0.437)

    g_actint1 = per_dif(GeoAct_int1, 0.422)

    g_allint1 = per_dif(GeoAll_int1, 0.426)


    slopetotmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1]), 5)
    inttotmean1 = round(np.mean([e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    totmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1,
        e_actint1, e_allint1, g_actint1, g_allint1]), 5)

    outlist.extend([slopetotmean1, inttotmean1, totmean1])



    e_actslope2 = per_err(EnvAct_slope2, -0.101)

    e_allslope2 = per_err(EnvAll_slope2, -0.054)

    g_actslope2 = per_err(GeoAct_slope2, -0.052)

    g_allslope2 = per_err(GeoAll_slope2, -0.032)

    e_actint2 = per_err(EnvAct_int2, 0.319)

    e_allint2 = per_err(EnvAll_int2, 0.257)

    g_actint2 = per_err(GeoAct_int2, 0.314)

    g_allint2 = per_err(GeoAll_int2, 0.255)

    slopetotmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2]), 5)
    inttotmean2 = round(np.mean([e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    totmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2,
        e_actint2, e_allint2, g_actint2, g_allint2]), 5)

    outlist.extend([slopetotmean2, inttotmean2, totmean2])



    e_actslope2 = per_dif(EnvAct_slope2, -0.101)

    e_allslope2 = per_dif(EnvAll_slope2, -0.054)

    g_actslope2 = per_dif(GeoAct_slope2, -0.052)

    g_allslope2 = per_dif(GeoAll_slope2, -0.032)

    e_actint2 = per_dif(EnvAct_int2, 0.319)

    e_allint2 = per_dif(EnvAll_int2, 0.257)

    g_actint2 = per_dif(GeoAct_int2, 0.314)

    g_allint2 = per_dif(GeoAll_int2, 0.255)

    slopetotmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2]), 5)
    inttotmean2 = round(np.mean([e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    totmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2,
        e_actint2, e_allint2, g_actint2, g_allint2]), 5)

    outlist.extend([slopetotmean2, inttotmean2, totmean2])



    e_actslope3 = per_err(EnvAct_slope3, -0.054)

    e_allslope3 = per_err(EnvAll_slope3, -0.029)

    g_actslope3 = per_err(GeoAct_slope3, -0.028)

    g_allslope3 = per_err(GeoAll_slope3, -0.016)

    e_actint3 = per_err(EnvAct_int3, 0.106)

    e_allint3 = per_err(EnvAll_int3, 0.081)

    g_actint3 = per_err(GeoAct_int3, 0.103)

    g_allint3 = per_err(GeoAll_int3, 0.080)


    slopetotmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3]), 5)
    inttotmean3 = round(np.mean([e_actint3, e_allint3, g_actint3, g_allint3]), 5)
    totmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3,
        e_actint3, e_allint3, g_actint3, g_allint3]), 5)

    outlist.extend([slopetotmean3, inttotmean3, totmean3])


    e_actslope3 = per_dif(EnvAct_slope3, -0.054)

    e_allslope3 = per_dif(EnvAll_slope3, -0.029)

    g_actslope3 = per_dif(GeoAct_slope3, -0.028)

    g_allslope3 = per_dif(GeoAll_slope3, -0.016)

    e_actint3 = per_dif(EnvAct_int3, 0.106)

    e_allint3 = per_dif(EnvAll_int3, 0.081)

    g_actint3 = per_dif(GeoAct_int3, 0.103)

    g_allint3 = per_dif(GeoAll_int3, 0.080)


    slopetotmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3]), 5)
    inttotmean3 = round(np.mean([e_actint3, e_allint3, g_actint3, g_allint3]), 5)
    totmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3,
        e_actint3, e_allint3, g_actint3, g_allint3]), 5)

    outlist.extend([slopetotmean3, inttotmean3, totmean3])


    outlist = str(outlist).strip('[]')
    outlist = outlist.replace(" ", "")

    mydir = os.path.expanduser("~/GitHub/DistDecay/model/ModelData")
    OUT = open(mydir+'/modelresults.txt','a')
    print>>OUT, outlist
    OUT.close()

    if Tfit2 == 10:
        #figfunction()
        Tfit2 = 0

    print i1, fit, Tfit1, Tfit2, '   :   ', Sact, minAct, avgAct, maxAct, '   :   ', Sall, minAll, avgAll, maxAll, '   :   ', n
