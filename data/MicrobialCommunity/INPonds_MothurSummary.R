################################################################################
#                                                                              #
#  IN Ponds Mothur Output Check                                                #
#                                                                              #
################################################################################
#                                                                              #
#  Written by: Mario Muscarella                                                #
#                                                                              #
#	Last update: 2015/04/14                                                      #
#                                                                              #
################################################################################
#                                                                              #
# Notes:                                                                       #
#                                                                              #
# Dependencies: 1) Vegan v2.2-0                                                #
#               2) MothurTools.R (2015/04/04)                                  #
#                                                                              #
# Issues:                                                                      #
#         1.                                                                   #
#                                                                              #
# Recent Changes:                                                              #
#         1.                                                                   #
#                                                                              #
# Future Changes (To-Do List):                                                 #
#         1.                                                                   #
#                                                                              #
################################################################################

# Setup Work Environment
rm(list=ls())
setwd("~/GitHub/Dimensions/Aim3/Mothur")
source("../bin/MothurTools.R")
require("vegan")
se <- function(x, ...){sd(x, ...)/sqrt(length(na.omit(x)))}

# Define Inputs
# Design = general design file for experiment
# shared = OTU table from mothur with sequence similarity clustering
design <- ""
shared <- "./INPonds.bac.final.shared"

# Import Design
# design <- read.delim(design, header=T, row.names=1)

# Import Shared Files
Pond97 <- read.otu(shared = shared, cutoff = "0.03")         # 97% Similarity
Pond95 <- read.otu(shared = shared, cutoff = "0.05")         # 95% Similarity

# Remove OTUs with less than two occurances across all sites
Pond97 <- Pond97[, which(colSums(Pond97) >= 2)]

# Coverage Stats
coverage <- rowSums(Pond97)
summary(coverage)

# Make Presence Absence Matrices
PondsPA <- (Pond97 > 0) * 1

rich <- rowSums(PondsPA)
summary(rich)

# Make Relative Abundence Matrices
PondsREL <- Pond97
for(i in 1:44){
  PondsREL[i,]<-Pond97[i,]/sum(Pond97[i,])
}
