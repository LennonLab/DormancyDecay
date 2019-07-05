from __future__ import division
import numpy as np
from numpy.random import binomial, uniform
from random import choice


def bide(env, pca, xs, ys, S, dd_r, ad_r, dd_s, ad_s, dded, env_r):

    match = 1/(1+np.abs(env - pca)) # Less difference between the environment and species optima equals a greater match.
    #match = (match + env_r)/(env_r + 1)
    mismatch = 1 - match

    #Act = logseries(match, (49,S)) # A species' probability of occurrence is proportional to its match to the environment.
    Act = binomial(1, match, (49, S)) * 50000 # A species' probability of occurrence is proportional to its match to the environment.
    # Initial active abundances are 50K for quantitative similarity to empirical data.
    Dor = binomial(1, 0.1, (49, S)) * 10**uniform(0, 2, (49, S))

    " Transition to dormancy "
    x = Act * mismatch # Species active/dormant abundances are proportional to local match/mismatch
    Dor = Dor + x
    Act = Act - x

    " Dispersal "
    '''
    r = choice(range(1,49))
    cDor = np.roll(Dor, r, axis=0)
    cAct = np.roll(Act, r, axis=0)
    x2 = np.roll(xs, r, axis=0)
    y2 = np.roll(ys, r, axis=0)

    dist = (((xs - x2)**2 + (ys - y2)**2))**0.5
    Dor = np.array(Dor) + ((cDor/dist) * binomial(1, dd_s, (49,S)))
    Act = np.array(Act) + ((cAct/dist) * binomial(1, ad_s, (49,S)))
    '''


    " death "
    Dor -= Dor * (1 - dded) # dded = the proportion of dormant organisms that die
    Act -= Act * mismatch # assume active organisms die in proportion to their environmental mismatch

    return np.round(Act), np.round(Dor), np.mean(match)
