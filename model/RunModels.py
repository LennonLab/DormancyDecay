from __future__ import division
import numpy as np
from numpy.random import uniform
from numpy import where
from scipy import spatial, stats
import os
import sys
from math import radians, cos, sin, asin, sqrt


mydir = os.path.expanduser('~/GitHub/DistDecay/model/')
sys.path.append(mydir)
from bide import bide




def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def difff(obs, exp):
    a_dif = np.abs(obs - exp)

    obs = np.abs(obs)
    exp = np.abs(exp)
    p_dif = 100 * np.abs(obs-exp)/np.abs(np.mean([obs, exp]))
    p_err = 100 * np.abs(obs-exp)/np.abs(exp)

    return p_err, p_dif, a_dif


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
    
    return lons, lats



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



col_headers = 'Sim,S,Sall,Sact,dd_s,ad_s,dded,'
col_headers += 'minAct,avgAct,maxAct,minAll,avgAll,maxAll,'

col_headers += 'm-mean-Bray-perr,b-mean-Bray-perr,t-mean-Bray-perr,'
col_headers += 'm-mean-Bray-pdif,b-mean-Bray-pdif,t-mean-Bray-pdif,'
col_headers += 'm-mean-Bray-adif,b-mean-Bray-adif,t-mean-Bray-adif,'
col_headers += 'Bray-ActEnv-m,Bray-AllEnv-m,Bray-ActGeo-m,Bray-AllGeo-m,'

col_headers += 'm-mean-Dice-perr,b-mean-Dice-perr,t-mean-Dice-perr,'
col_headers += 'm-mean-Dice-pdif,b-mean-Dice-pdif,t-mean-Dice-pdif,'
col_headers += 'm-mean-Dice-adif,b-mean-Dice-adif,t-mean-Dice-adif,'
col_headers += 'Dice-ActEnv-m,Dice-AllEnv-m,Dice-ActGeo-m,Dice-AllGeo-m,'

col_headers += 'm-mean-Canb-perr,b-mean-Canb-perr,t-mean-Canb-perr,'
col_headers += 'm-mean-Canb-pdif,b-mean-Canb-pdif,t-mean-Canb-pdif,'
col_headers += 'm-mean-Canb-adif,b-mean-Canb-adif,t-mean-Canb-adif,'
col_headers += 'Canb-ActEnv-m,Canb-AllEnv-m,Canb-ActGeo-m,Canb-AllGeo-m,'

col_headers += 'env_r,avgMatch,fit'


mydir = os.path.expanduser("~/GitHub/DistDecay/model/ModelData")
OUT = open(mydir+'/modelresults.txt','w+')
print>>OUT, col_headers
OUT.close()


Tfit1 = 0
ints = True
slopes = 'notall'
env_max = 8
for i1 in range(0, 100000):

    dd_s = uniform(0, 1)
    ad_s = uniform(0, 1)
    dded = uniform(0, 1)

    S = 20000
    xs, ys = getXY(S)
    pca = getEnv(S)

    env_r = uniform(0, env_max)
    
    env = uniform(np.amin(pca) * env_r, env_r * np.amax(pca), S)
    env = np.array([env,] * 49)

    Act, Dor, avgMatch = bide(env, pca, xs, ys, S, dd_s, ad_s, dded)
    All = Dor + Act
    
    xs = np.tile(np.array([xs]).transpose(), (1, S))
    ys = np.tile(np.array([ys]).transpose(), (1, S))
    
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

    minAct = int(round(np.amin(Act)))
    avgAct = int(round(np.mean(Act)))
    maxAct = int(round(np.amax(Act)))

    minAll = int(round(np.amin(All)))
    avgAll = int(round(np.mean(All)))
    maxAll = int(round(np.amax(All)))

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
            
            sadr = np.asarray([Act[j], Act[k]])
            #sadr = np.delete(sadr, np.where(~sadr.any(axis=0))[0], axis=1)
            sad1 = sadr[0]
            sad2 = sadr[1]
            l = len(sad1)
            
            sadr = np.asarray([All[j], All[k]])
            #sadr = np.delete(sadr, np.where(~sadr.any(axis=0))[0], axis=1)
            sad3 = sadr[0]
            sad4 = sadr[1]
            
            pair = [sad1, sad2]
            sim1 = 1 - spatial.distance.pdist(pair, metric='braycurtis')[0]
            sim3 = 1 - spatial.distance.pdist(pair, metric='dice')[0]
            sim5 = 1 - spatial.distance.pdist(pair, metric='canberra')[0]/l
            
            pair = [sad3, sad4]
            sim2 = 1 - spatial.distance.pdist(pair, metric='braycurtis')[0]
            sim4 = 1 - spatial.distance.pdist(pair, metric='dice')[0]
            sim6 = 1 - spatial.distance.pdist(pair, metric='canberra')[0]/l
            
            
            actdif1.append(sim1)
            alldif1.append(sim2)
            actdif2.append(sim3)
            alldif2.append(sim4)
            actdif3.append(sim5)
            alldif3.append(sim6)

            
            x1 = xs[j][0]
            x2 = xs[k][0]
            y1 = ys[j][0]
            y2 = ys[k][0]

            dif = haversine(x1, y1, x2, y2)
            geodif.append(dif)

            dif = np.absolute(pca[j][0] - pca[k][0])
            envdif.append(dif)


    envdif = np.array(envdif)
    geodif = np.array(geodif)
    envdif = envdif/np.amax(envdif)
    geodif = geodif/np.amax(geodif)


    EnvAct_slope1, EnvAct_int1, r_value, p_val1, std_err = stats.linregress(envdif, actdif1)
    EnvAll_slope1, EnvAll_int1, r_value, p_val, std_err = stats.linregress(envdif, alldif1)
    GeoAct_slope1, GeoAct_int1, r_value, p_val, std_err = stats.linregress(geodif, actdif1)
    GeoAll_slope1, GeoAll_int1, r_value, p_val, std_err = stats.linregress(geodif, alldif1)

    EnvAct_slope2, EnvAct_int2, r_value, p_val2, std_err = stats.linregress(envdif, actdif2)
    EnvAll_slope2, EnvAll_int2, r_value, p_val, std_err = stats.linregress(envdif, alldif2)
    GeoAct_slope2, GeoAct_int2, r_value, p_val, std_err = stats.linregress(geodif, actdif2)
    GeoAll_slope2, GeoAll_int2, r_value, p_val, std_err = stats.linregress(geodif, alldif2)

    EnvAct_slope3, EnvAct_int3, r_value, p_val3, std_err = stats.linregress(envdif, actdif3)
    EnvAll_slope3, EnvAll_int3, r_value, p_val, std_err = stats.linregress(envdif, alldif3)
    GeoAct_slope3, GeoAct_int3, r_value, p_val, std_err = stats.linregress(geodif, actdif3)
    GeoAll_slope3, GeoAll_int3, r_value, p_val, std_err = stats.linregress(geodif, alldif3)


    fit = 0
    fits = []
    pvs = [p_val1, p_val2, p_val3]
    if max(pvs) > 0.05: fits.append(False)

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

    if EnvAct_int2 > EnvAll_int2: fits.append(True)
    else: fits.append(False)
    if EnvAct_int2 > GeoAct_int2: fits.append(True)
    else: fits.append(False)


    if EnvAct_slope3 < EnvAll_slope3: fits.append(True)
    else: fits.append(False)
    if EnvAct_slope3 < GeoAct_slope3: fits.append(True)
    else: fits.append(False)

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

    e_actslope1,e_actslope2,e_actslope3, e_allslope1,e_allslope2,e_allslope3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    g_actslope1,g_actslope2,g_actslope3, g_allslope1,g_allslope2,g_allslope3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    e_actint1,e_actint2,e_actint3, e_allint1,e_allint2,e_allint3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    g_actint1,g_actint2,g_actint3, g_allint1,g_allint2,g_allint3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')

    if EnvAct_slope1 < 0:
        e_actslope1,e_actslope2,e_actslope3 = difff(EnvAct_slope1, -0.325)
    else: fits.append(False)

    if EnvAll_slope1 < 0:
        e_allslope1,e_allslope2,e_allslope3 = difff(EnvAll_slope1, -0.248)
    else: fits.append(False)

    if GeoAct_slope1 < 0:
        g_actslope1,g_actslope2,g_actslope3 = difff(GeoAct_slope1, -0.182)
    else: fits.append(False)

    if GeoAll_slope1 < 0:
        g_allslope1,g_allslope2,g_allslope3 = difff(GeoAll_slope1, -0.132)
    else: fits.append(False)

    if EnvAct_int1 > 0:
        e_actint1,e_actint2,e_actint3 = difff(EnvAct_int1, 0.434)
    else: fits.append(False)

    if EnvAll_int1 > 0:
        e_allint1,e_allint2,e_allint3 = difff(EnvAll_int1, 0.437)
    else: fits.append(False)

    if GeoAct_int1 > 0:
        g_actint1,g_actint2,g_actint3 = difff(GeoAct_int1, 0.422)
    else: fits.append(False)

    if GeoAll_int1 > 0:
        g_allint1,g_allint2,g_allint3 = difff(GeoAll_int1, 0.426)
    else: fits.append(False)


    inttotmean1 = round(np.mean([e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    totmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1, e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    inttotmean2 = round(np.mean([e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    totmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2, e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    inttotmean3 = round(np.mean([e_actint3, e_allint3, g_actint3, g_allint3]), 5)
    totmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3, e_actint3, e_allint3, g_actint3, g_allint3]), 5)

    if slopes == 'all':
        slopetotmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1]), 5)
        slopetotmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2]), 5)
        slopetotmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3]), 5)
    else:
        slopetotmean1 = round(np.mean([e_actslope1, e_allslope1]), 5)
        slopetotmean2 = round(np.mean([e_actslope2, e_allslope2]), 5)
        slopetotmean3 = round(np.mean([e_actslope3, e_allslope3]), 5)

    outlist = [i1, S, Sall, Sact, dd_s, ad_s, dded, minAct, avgAct, maxAct, minAll, avgAll, maxAll]
    outlist.extend([slopetotmean1, inttotmean1, totmean1])
    outlist.extend([slopetotmean2, inttotmean2, totmean2])
    outlist.extend([slopetotmean3, inttotmean3, totmean3])
    outlist.extend([EnvAct_slope1,EnvAll_slope1,GeoAct_slope1,GeoAll_slope1])


    e_actslope1,e_actslope2,e_actslope3, e_allslope1,e_allslope2,e_allslope3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    g_actslope1,g_actslope2,g_actslope3, g_allslope1,g_allslope2,g_allslope3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    e_actint1,e_actint2,e_actint3, e_allint1,e_allint2,e_allint3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    g_actint1,g_actint2,g_actint3, g_allint1,g_allint2,g_allint3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')

    if EnvAct_slope2 < 0:
        e_actslope1,e_actslope2,e_actslope3 = difff(EnvAct_slope2, -0.101)
    else: fits.append(False)

    if EnvAll_slope2 < 0:
        e_allslope1,e_allslope2,e_allslope3 = difff(EnvAll_slope2, -0.054)
    else: fits.append(False)

    if GeoAct_slope2 < 0:
        g_actslope1,g_actslope2,g_actslope3 = difff(GeoAct_slope2, -0.052)
    else: fits.append(False)

    if GeoAll_slope2 < 0:
        g_allslope1,g_allslope2,g_allslope3 = difff(GeoAll_slope2, -0.032)
    else: fits.append(False)

    if EnvAct_int2 > 0:
        e_actint1,e_actint2,e_actint3 = difff(EnvAct_int2, 0.319)
    else: fits.append(False)

    if EnvAll_int2 > 0:
        e_allint1,e_allint2,e_allint3 = difff(EnvAll_int2, 0.257)
    else: fits.append(False)

    if GeoAct_int2 > 0:
        g_actint1,g_actint2,g_actint3 = difff(GeoAct_int2, 0.314)
    else: fits.append(False)

    if GeoAll_int2 > 0:
        g_allint1,g_allint2,g_allint3 = difff(GeoAll_int2, 0.255)
    else: fits.append(False)

    inttotmean1 = round(np.mean([e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    totmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1, e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    inttotmean2 = round(np.mean([e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    totmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2, e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    inttotmean3 = round(np.mean([e_actint3, e_allint3, g_actint3, g_allint3]), 5)
    totmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3, e_actint3, e_allint3, g_actint3, g_allint3]), 5)

    if slopes == 'all':
        slopetotmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1]), 5)
        slopetotmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2]), 5)
        slopetotmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3]), 5)
    else:
        slopetotmean1 = round(np.mean([e_actslope1, e_allslope1]), 5)
        slopetotmean2 = round(np.mean([e_actslope2, e_allslope2]), 5)
        slopetotmean3 = round(np.mean([e_actslope3, e_allslope3]), 5)


    outlist.extend([slopetotmean1, inttotmean1, totmean1])
    outlist.extend([slopetotmean2, inttotmean2, totmean2])
    outlist.extend([slopetotmean3, inttotmean3, totmean3])
    outlist.extend([EnvAct_slope2,EnvAll_slope2,GeoAct_slope2,GeoAll_slope2])

    e_actslope1,e_actslope2,e_actslope3, e_allslope1,e_allslope2,e_allslope3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    g_actslope1,g_actslope2,g_actslope3, g_allslope1,g_allslope2,g_allslope3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    e_actint1,e_actint2,e_actint3, e_allint1,e_allint2,e_allint3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')
    g_actint1,g_actint2,g_actint3, g_allint1,g_allint2,g_allint3 = float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')

    if EnvAct_slope3 < 0:
        e_actslope1,e_actslope2,e_actslope3 = difff(EnvAct_slope3, -0.054)
    else: fits.append(False)

    if EnvAll_slope3 < 0:
        e_allslope1,e_allslope2,e_allslope3 = difff(EnvAll_slope3, -0.029)
    else: fits.append(False)

    if GeoAct_slope3 < 0:
        g_actslope1,g_actslope2,g_actslope3 = difff(GeoAct_slope3, -0.028)
    else: fits.append(False)

    if GeoAll_slope3 < 0:
        g_allslope1,g_allslope2,g_allslope3 = difff(GeoAll_slope3, -0.016)
    else: fits.append(False)

    if EnvAct_int3 > 0:
        e_actint1,e_actint2,e_actint3 = difff(EnvAct_int3, 0.106)
    else: fits.append(False)

    if EnvAll_int3 > 0:
        e_allint1,e_allint2,e_allint3 = difff(EnvAll_int3, 0.081)
    else: fits.append(False)

    if GeoAct_int3 > 0:
        g_actint1,g_actint2,g_actint3 = difff(GeoAct_int3, 0.103)
    else: fits.append(False)

    if GeoAll_int3 > 0:
        g_allint1,g_allint2,g_allint3 = difff(GeoAll_int3, 0.080)
    else: fits.append(False)

    inttotmean1 = round(np.mean([e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    totmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1, e_actint1, e_allint1, g_actint1, g_allint1]), 5)
    inttotmean2 = round(np.mean([e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    totmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2, e_actint2, e_allint2, g_actint2, g_allint2]), 5)
    inttotmean3 = round(np.mean([e_actint3, e_allint3, g_actint3, g_allint3]), 5)
    totmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3, e_actint3, e_allint3, g_actint3, g_allint3]), 5)

    if slopes == 'all':
        slopetotmean1 = round(np.mean([e_actslope1, e_allslope1, g_actslope1, g_allslope1]), 5)
        slopetotmean2 = round(np.mean([e_actslope2, e_allslope2, g_actslope2, g_allslope2]), 5)
        slopetotmean3 = round(np.mean([e_actslope3, e_allslope3, g_actslope3, g_allslope3]), 5)
    else:
        slopetotmean1 = round(np.mean([e_actslope1, e_allslope1]), 5)
        slopetotmean2 = round(np.mean([e_actslope2, e_allslope2]), 5)
        slopetotmean3 = round(np.mean([e_actslope3, e_allslope3]), 5)


    if fits.count(False) == 0:
        fit = 1
        Tfit1 += 1
    else:
        fit = 0


    outlist.extend([slopetotmean1, inttotmean1, totmean1])
    outlist.extend([slopetotmean2, inttotmean2, totmean2])
    outlist.extend([slopetotmean3, inttotmean3, totmean3])
    outlist.extend([EnvAct_slope3,EnvAll_slope3,GeoAct_slope3,GeoAll_slope3])
    outlist.extend([env_r, avgMatch, fit])
    outlist = str(outlist).strip('[]')
    outlist = outlist.replace(" ", "")

    mydir = os.path.expanduser("~/GitHub/DistDecay/model/ModelData")
    OUT = open(mydir+'/modelresults.txt','a')
    print>>OUT, outlist
    OUT.close()


    print i1, fit, Tfit1, '   :   ', Sact, minAct, avgAct, maxAct, '   :   ', Sall, minAll, avgAll, maxAll, '   :   ', n
