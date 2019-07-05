from __future__ import division
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt



mydir = expanduser("~/GitHub/DistDecay")
# set up the figure
fig = plt.figure()


fig.add_subplot(2,2,1)

xs = np.linspace(0, 1, 100)
ys1 = (1 - xs)
ys2 = 0.75*(1 - 0.65*xs)

plt.plot(xs, ys1, ls='-', linewidth=2, color='k')
plt.plot(xs, ys2, ls='--', linewidth=2, color='0.4')

plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)

plt.ylabel('Community similarity', fontsize=9)
plt.xlabel('Distance, geographical or\nenvironmental', fontsize=9)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir+'/figs/fig1/Fig1.png', dpi=400, bbox_inches = "tight")
plt.close()

