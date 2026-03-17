# run_scibet_prediction.R (V4 - Final Correct Version)

# 1. 加载必要的库
# suppressPackageStartupMessages 会阻止加载包时显示版本等信息
suppressPackageStartupMessages(library(scibetR))
suppressPackageStartupMessages(library(data.table))

# 2. 使用R的内置方法获取命令行参数
# commandArgs(trailingOnly = TRUE) 会返回一个包含所有脚本后参数的字符向量
args <- commandArgs(trailingOnly = TRUE)

# 检查是否提供了正确数量的参数 (3个)
if (length(args) != 3) {
  # 如果参数数量不对，打印错误信息并以失败状态退出
  stop("错误: 需要提供3个文件路径参数: <reference_csv> <query_csv> <output_csv>", call. = FALSE)
}

# 3. 按位置为参数分配有意义的名称，方便阅读
ref_path <- args[1]
query_path <- args[2]
out_path <- args[3]

# 4. 主逻辑，包含在tryCatch错误处理块中
# 这能确保任何意外错误都会被捕获，并以清晰的方式报告给Python
tryCatch({
  # --- 数据加载 ---
  cat("R_SCRIPT: Loading reference data from:", ref_path, "\n")
  # 使用data.table::fread()，这是R中读取大型CSV最快的方式
  ref_df <- fread(ref_path, data.table = FALSE)
  # fread读取后，第一列是pandas写入的索引列，需要将其设为行名
  rownames(ref_df) <- ref_df[,1]
  ref_df <- ref_df[,-1] # 移除作为第一列的索引
  cat("R_SCRIPT: Reference data dimensions:", dim(ref_df)[1], "rows,", dim(ref_df)[2], "cols\n")
  
  cat("R_SCRIPT: Loading query data from:", query_path, "\n")
  query_df <- fread(query_path, data.table = FALSE)
  rownames(query_df) <- query_df[,1]
  query_df <- query_df[,-1]
  cat("R_SCRIPT: Query data dimensions:", dim(query_df)[1], "rows,", dim(query_df)[2], "cols\n")
  
  # --- 核心预测 ---
  cat("R_SCRIPT: Running SciBet prediction...\n")
  
  # *** 最终修正点 ***
  # 根据 ls("package:scibetR") 的输出，调用正确的函数 `SciBet_R`
  predictions <- SciBet_R(ref_df, query_df)
  
  cat("R_SCRIPT: SciBet prediction finished.\n")
  
  # --- 结果保存 ---
  cat("R_SCRIPT: Saving predictions to:", out_path, "\n")
  # 将预测结果（一个向量）转换为数据框并写入CSV文件
  write.csv(data.frame(predictions=predictions), file=out_path, row.names=FALSE)
  
  cat("R_SCRIPT: Prediction complete and file saved successfully.\n")
  
}, error = function(e) {
  # --- 错误处理 ---
  # 如果tryCatch块中的任何地方发生错误，将执行此部分
  cat("\n\n--- R SCRIPT ERROR ---\n")
  cat("An error occurred during the R script execution.\n")
  # 打印R提供的详细错误信息
  print(e)
  cat("--- END R SCRIPT ERROR ---\n\n")
  # 以非零状态码退出，这会通知Python脚本执行失败
  quit(status = 1)
})