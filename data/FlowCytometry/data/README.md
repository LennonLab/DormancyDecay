##Sample collection

Samples for this analysis were collected as part of a larger project evaluating microbial communities in freshwater ponds throughout south central Indiana. Briefly, one liter water samples were collected and brought back to the lab. Each sample was passed through a 5 uM syringe filter to remove large particulate matter that could potentially get stuck in the flow cytometer tubing lines. Two one milliliter samples were stained with 1 uL eFluor660 at room temperature for 30 minutes prior to fixation with 13.5 uL 37% Formalin. eFluor660 is a fixed viablity dye that stains cells with non-intact walls indicating that they are dead. All samples were stored at -80C until thawing for analysis.

Samples were processed in batches of 40-60 samples. We thawed each sample in the dark on ice to preserve the eFluor660 staining. Once thawed, we transferred each sample to a 15x75 mm clamp cap tube with two drops of Hoeschst 33342, 5uL 1:1000 Pyronin-y stain, and 1 uL bead standard for cell counting (final bead concentration in sample ~ 1x10^6 beads/mL).   

We collected 50,000 bead events on the Aria II flow cytometer in the Indiana University Flow Cytometry Core Facility (director: C. Hassel) and analyzed data using R. 

## List of columns in `sample_data.csv`

1. **sample_date** Date of analysis
2. **flow-sample** Name of corresponding flow sample in `./FCS/' directory. Sample names with a 'u' indicate unstained controls. 
3. **pond** Name of sample pond
