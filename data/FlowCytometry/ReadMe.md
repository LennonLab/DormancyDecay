# flow_cytometry

This directory contains data files produced by the flow analysis of each IN Pond sample

## Contents
### Directories and files

**INPonds_flowdat.csv**: summary output data from each IN pond sample.

Column descriptions

 * *Sample_ID*: IN pond ID
 * *sample*: flow sample ID
 * *sample.type*: unstained or stained controls
 * *ratio.min*: minimum calculated value for the sample RNA/DNA ratio
 * *ratio.max*:  minimum calculated value for the sample RNA/DNA ratio
 * *ratio.mean*: mean RNA/DNA ratio
 * *ratio.median* median RNA/DNA ratio
 * *ratio.variance*: sample RNA/DNA ratio variance
 * *dead.dens*: adjusted density (cells/mL) of dead bacteria in community
 * *live.dens*: adjusted density (cells/mL) of live bacteria in community
 * *act.dens*: adjusted density (cells/mL) of the active population
 * *dorm.dens*: adjusted density (cells/mL) of the dormant population
 * *act.perc* percentage of active population within tive
 * *dorm.perc*: percentage of dormant population with total live


**event_counts/**: This directory contains the RNA/DNA ratio as well as the RNA and DNA fluorescence per event within each pond. Files are named according to sample name as described above.

**figures/**: This directory contains .png images of the RNA/DNA histograms


