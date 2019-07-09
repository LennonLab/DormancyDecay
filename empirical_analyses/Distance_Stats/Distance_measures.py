from __future__ import division
import numpy as np
from math import radians, cos, sin, asin, sqrt
from os.path import expanduser
import sys



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



######################### RUN SIMULATIONS ###############################


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


Ds = []    
Ds_nn = []    
Ds_fn = []  

for i, lat1 in enumerate(lats):
    
    lon1 = lons[i]        
    d_nn = 10**6
    d_fn = 0
    
    for ii, lat2 in enumerate(lats):
        
        if ii <= i:
            continue
        
        lon2 = lons[ii]
                
        d = haversine(lon1, lat1, lon2, lat2)
                    
        Ds.append(d)
                
        if d < d_nn:
            d_nn = float(d)
                    
        if d > d_fn:
            d_fn = float(d)
    
    if d_nn != 10**6 and d_fn != 0:
        Ds_nn.append(d_nn)
        Ds_fn.append(d_fn)
            
                
min_geo_dist = min(Ds)
max_geo_dist = max(Ds)
mean_geo_dist = np.mean(Ds)
mean_geo_dist_std = np.std(Ds)
        
min_geo_dist_nn = min(Ds_nn)
max_geo_dist_nn = max(Ds_nn)
mean_geo_dist_nn = np.mean(Ds_nn)
mean_geo_dist_nn_std = np.std(Ds_nn)

min_geo_dist_fn = min(Ds_fn)
max_geo_dist_fn = max(Ds_fn)
mean_geo_dist_fn = np.mean(Ds_fn)
mean_geo_dist_fn_std = np.std(Ds_fn)
        

print 'min dist:', min_geo_dist
print 'max dist:', max_geo_dist
print 'mean dist:', mean_geo_dist_std

print 'nn:', mean_geo_dist_nn_std
print 'fn:', mean_geo_dist_fn_std
        
            

