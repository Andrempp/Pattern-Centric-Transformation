#!/usr/bin/env Rscript

install.packages('pacman')
install.packages("BiocManager")
BiocManager::install("TCGAbiolinks", force=TRUE)
BiocManager::install("sesame", force=TRUE)
BiocManager::install("sesameData", force=TRUE)

pacman::p_load(BiocManager, preprocessCore, DT, TCGAbiolinks, 
               SummarizedExperiment, sesameData, sesame)


input_message = "Input in format: Rscript file.R {project} {path to save file}"
args = commandArgs(trailingOnly=TRUE) # get arguments from command line

if (length(args)!=2) {
  stop(input_message, call.=FALSE)
} else{
  project <- args[1]
  file_path <- args[2]
}

# project <- 'TCGA-LGG'
# file_path <- '~/Desktop/temp_ids.RData'

#dir_path <- paste(dir_path, project, "/", sep="")

#dir.create(file.path(dir_path), showWarnings = FALSE)

clinical <- TCGAbiolinks::GDCquery_clinic(project = project, type = "clinical")

#write.csv(clinical$submitter_id, paste(dir, "ids.csv", sep=""), row.names = FALSE)

saveRDS(clinical$submitter_id, file=file_path)

pacman::p_unload(all)

#detach("package:", unload = TRUE)

cat("\014")

