#!/usr/bin/env Rscript

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

pacman::p_load("BiocManager", "edgeR", "rjson")


input_message <- "Input in format: Rscript file.R {input_file} {output_file} {target}"
args <- commandArgs(trailingOnly = TRUE) # get arguments from command line

if (length(args) != 3) {
  stop(input_message, call. = FALSE)
} else{
  input_file <- args[1]
  output_file <- args[2]
  target <- args[3]
}

# input_file <- "/home/andrepatricio/Desktop/phd_thesis/packages/pydge/temp_get_filtered_genes.csv"
# output_file <- "/home/andrepatricio/Desktop/phd_thesis/data/TCGA-LGG/mirna/vital_status/filtered.json"
# target <- "vital_status"

x <- as.matrix(read.csv(input_file, row.names = 1, check.names=FALSE)) #read file as a matrix, index column is the first one, don't alter the name of columns from - to .
groups <- x[, target]                            #get the groupings (classes) of samples
x <- x[, colnames(x) != target]                  #remove column with groupings
x2 <- t(x)                                       #transverse of matrix to genes in rows and samples i columns
class(x2) <- "numeric"                           #if target was not numeric, need to revert the matrix back to numeric

y <- DGEList(counts = x2, group = groups)          #create object DGEList, contains count matrix and data.frame samples with:
                                                 # groupings, lib sizes, and normalization factors per sample

keep <- filterByExpr(y)                          #selects genes to filter out due to very low counts
myfile = toJSON(keep)
write(myfile, output_file)

# saveRDS(keep, output_file)
