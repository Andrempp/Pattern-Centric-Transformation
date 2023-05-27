#!/usr/bin/env Rscript

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

pacman::p_load(BiocManager,"edgeR")

input_message = "Input in format: Rscript file.R {input_file} {output_file} {pvalue_cutoff}"
args = commandArgs(trailingOnly=TRUE) # get arguments from command line

if (length(args)!=3) {
  stop(input_message, call.=FALSE)
} else{
  input_file <- args[1]
  output_file <- args[2]
  pvalue_cutoff <- args[3]
}

dge <- readRDS(input_file)

corrected_genes <- decideTestsDGE(dge, adjust.method = "BH", p.value = pvalue_cutoff) #returns significance of genes (and if up or down expressed) according to corrected p-value 

#genes = topTags(et, n=10, adjust.method = "BH", sort.by = "PValue", p.value = 1) #returns top n genes by sort.by, adjusted with adjust.method and only returns if p-value lower than p.value

write.csv(corrected_genes, output_file, row.names = TRUE)
