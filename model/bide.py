from __future__ import division
import numpy as np
from numpy.random import binomial, uniform
from random import choice
import sys



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    
    return km


def cdist(d):
    return 0.001/(0.001+d)

def dispersal(Dor, Act, xs, ys, S, dd_s, ad_s):
    
    r = choice(range(1,49))
    cDor = np.roll(Dor, r, axis=0)
    x2 = np.roll(xs, r, axis=0)
    y2 = np.roll(ys, r, axis=0)

    dist = haversine(xs, ys, x2, y2)
    dist = cdist(dist)
    dist = np.tile(np.array([dist]).transpose(), (1, S))
    
    Dor = Dor + cDor * dist * dd_s
    
    r = choice(range(1,49))
    cAct = np.roll(Act, r, axis=0)
    x2 = np.roll(xs, r, axis=0)
    y2 = np.roll(ys, r, axis=0)

    dist = haversine(xs, ys, x2, y2)
    dist = cdist(dist)
    dist = np.tile(np.array([dist]).transpose(), (1, S))
    
    Act = Act + cAct * dist * ad_s
    
    return Dor, Act




def bide(env, pca, xs, ys, S, dd_s, ad_s, dded):

    match = 1/(1+np.abs(env - pca))
    mismatch = 1 - match
    
    Act = binomial(1, match, (49,S)).astype(float) * 50000
    Dor = binomial(1, 0.1, (49,S)).astype(float) * 10**uniform(0, 2, (49,S))

    x = Act * mismatch
    Act -= x
    Dor += x
    
    " Dispersal "
    Dor, Act = dispersal(Dor, Act, xs, ys, S, dd_s, ad_s)

    " death "
    Dor -= Dor * dded
    Act = Act * match

    return np.round(Act), np.round(Dor), np.mean(match)








