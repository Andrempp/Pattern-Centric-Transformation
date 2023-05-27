#!/usr/bin/env Rscript

install.packages("pacman")

pacman::p_load(BiocManager, preprocessCore, DT, TCGAbiolinks, 
               SummarizedExperiment, sesameData, sesame)

BiocManager::install("TCGAbiolinks", force = TRUE)


input_message = "Input in format: Rscript file.R {project} {path to save file}"
args = commandArgs(trailingOnly=TRUE) # get arguments from command line

if (length(args)!=2) {
  stop(input_message, call.=FALSE)
} else{
  project <- args[1]
  file_path <- args[2]
}

#dir_path <- paste(dir_path, project, "/", sep="")

clinical <- TCGAbiolinks::GDCquery_clinic(project = project, type = "clinical")

clinical <- subset(clinical, select=-c(project))

write.csv(clinical, file_path, row.names = FALSE)

#p_unload(all)
#detach("package:", unload = TRUE)
#cat("\014")

