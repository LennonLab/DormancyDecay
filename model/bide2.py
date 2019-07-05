from __future__ import division
import numpy as np
from numpy.random import binomial, logseries
from numpy import sin, cos, arcsin
from random import choice
from math import pi



def bide(env, pca, xs, ys, S, dd_r, ad_r, dd_s, ad_s, dded, env_r):

    m = 1/(1+np.abs(env - pca))
    m2 = (m + env_r)/(env_r + 1)

    Act = logseries(m2, (49,S))
    Dor = logseries(0.999, (49,S)) * binomial(1, 0.1, (49,S))
    
    " Dispersal "
    '''
    for i in range(1):
        r = choice(range(1, 49))
        cDor = np.roll(Dor, r, axis=0)
        cAct = np.roll(Act, r, axis=0)

        x1 = (xs * pi)/180.0
        y1 = (ys * pi)/180.0
        x2 = np.roll(x1, r, axis=0)
        y2 = np.roll(y1, r, axis=0)
            
        dlon = x2 - x1
        dlat = y2 - y1
        a = sin(dlat/2.0)**2.0 + cos(y1) * cos(y2) * sin(dlon/2)**2
        dist = 6371.0 * 2.0 * arcsin(a**0.5)
        
        Dor = Dor + (cDor/dist) * dd_s
        Act = Act + (cAct/dist) * ad_s
    '''
        
    " death "
    Dor = Dor * (1.0 - dded)
    Act = Act * m
    
    return np.round(Act), np.round(Dor), np.mean(m)
