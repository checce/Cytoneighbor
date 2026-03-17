# Network/run_singler.R

# --- 1. Load necessary libraries ---
# (这部分保持不变)
packages <- c("SingleR", "SingleCellExperiment", "zellkonverter", "Matrix")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Package '", pkg, "' is not installed. Please install it by running:\n",
               "if (!require('BiocManager', quietly = TRUE)) install.packages('BiocManager')\n",
               "BiocManager::install('", pkg, "')", sep=""), call. = FALSE)
  }
}

suppressPackageStartupMessages({
  library(SingleR)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

# --- 2. 【重大修改】: 使用 R 内置的 commandArgs 来解析参数 ---
# 我们不再使用 argparse 包
args_vector <- commandArgs(trailingOnly = TRUE)

# 检查是否收到了正确数量的参数
if (length(args_vector) != 4) {
  stop("脚本需要 4 个命令行参数: <ref_path> <query_path> <outfile_path> <label_key>", call. = FALSE)
}

# 按顺序分配参数
ref_path <- args_vector[1]
query_path <- args_vector[2]
outfile_path <- args_vector[3]
label_key <- args_vector[4]


# --- 3. Main execution logic ---
cat("--- Starting SingleR Annotation Script (Robust Version) ---\n")
# 打印接收到的参数以供调试
cat("1. Reference file:", ref_path, "\n")
cat("2. Query file:", query_path, "\n")
cat("3. Output file:", outfile_path, "\n")
cat("4. Label key:", label_key, "\n\n")

tryCatch({
  # --- Read data ---
  cat("Step 1: Reading .h5ad files into SingleCellExperiment objects...\n")
  ref_sce <- readH5AD(ref_path, reader = "R")
  query_sce <- readH5AD(query_path, reader = "R")
  
  assayNames(ref_sce)[1] <- "logcounts"
  assayNames(query_sce)[1] <- "logcounts"
  
  # --- Train SingleR model ---
  cat("Step 2: Training SingleR model on reference data...\n")
  trained_ref <- trainSingleR(ref_sce, labels = colData(ref_sce)[[label_key]])
  
  # --- Classify query data ---
  cat("Step 3: Classifying query data...\n")
  predictions <- classifySingleR(query_sce, trained_ref)
  
  # --- Save predictions ---
  cat("Step 4: Saving predictions to CSV...\n")
  output_df <- data.frame(predicted_label = predictions$pruned.labels)
  write.csv(output_df, file = outfile_path, row.names = FALSE)
  
  cat("\n--- SingleR Annotation successful! ---\n")
  
}, error = function(e) {
  cat("Error during SingleR execution:\n", file = stderr())
  cat(e$message, "\n", file = stderr())
  quit(status = 1)
})