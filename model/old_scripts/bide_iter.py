from __future__ import division
from random import shuffle, choice
import numpy as np
from numpy.random import binomial
from numpy import where
import sys



def bide(env, pca, xs, ys, S, dd_r, ad_r, dd_s, ad_s, dded, env_r):

    match = 1/(1+np.abs(env - pca))
    mismatch = 1 - match

    aNs, dNs, allNs = [], [], []
    Act = binomial(1, 1.0, (49,S))
    Dor = binomial(1, 0.0, (49,S))

    step = 1000
    match = 1/(1+np.abs(env - pca))
    mismatch = 1 - match
    avgMatch = np.mean(match)

    procs = range(6)
    for t in range(step):
        #print t

        shuffle(procs)
        for i3 in procs:

            if i3 == 0:
                " Growth "
                p = match #/(1 + np.exp(match*(Act-match*1000)))
                Act += binomial(1, p, (49, S)) * where(Act > 0, 1, 0)

            elif i3 == 1:
                " Transition to dormancy "
                x = binomial(1, mismatch, (49, S)) * where(Act > 0, 1, 0)
                Dor += x
                Act -= x

            elif i3 == 2:
                " Transition to activity "
                x = binomial(1, match, (49, S)) * where(Dor > 0, 1, 0)
                Act += x
                Dor -= x

            elif i3 == 3:
                " Immigration "
                '''
                r = choice(range(1, 49))
                cDor = np.copy(Dor)
                cAct = np.copy(Act)
                x2 = np.copy(xs)
                y2 = np.copy(ys)

                cDor = np.roll(cDor, r, axis=0)
                cAct = np.roll(cAct, r, axis=0)
                x2 = np.roll(x2, r, axis=0)
                y2 = np.roll(y2, r, axis=0)

                dist = ((xs - x2)**2 + (ys - y2)**2)**0.5
                p = 100/dist

                pd = p*dd_s
                pa = p*ad_s

                Dor += binomial(1, pd, (49, S)) * where(cDor > 0, 1, 0)
                Act += binomial(1, pa, (49, S)) * where(cAct > 0, 1, 0)

                Dor += binomial(1, dd_r, (49, S)) * where(cDor > 0, 1, 0)
                Act += binomial(1, ad_r, (49, S)) * where(cAct > 0, 1, 0)
                '''


            elif i3 == 5:
                " death "
                #Act -= binomial(1, mismatch, (49, S)) * where(Act > 0, 1, 0)
                #Dor -= binomial(1, dded, (49, S)) * where(Dor > 0, 1, 0)


    return Act, Dor, avgMatch
