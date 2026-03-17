# Network/run_seurat.R (最终稳健版 - 单线程)

# --- 1. 加载包 ---
suppressPackageStartupMessages({
  library(Seurat)
  library(argparse)
  library(Matrix)
  # library(future) # 不再需要显式加载
})

# ==================== 最终策略调整: 回归单线程以保证稳定性 ====================
# 由于数据集过大，并行计算导致子进程内存溢出而崩溃。
# 我们将禁用并行计算，使用单线程模式运行。
# 这会牺牲速度，但能确保在内存有限的机器上成功完成任务。
# plan("sequential") # 这是默认行为，无需显式设置
cat("警告: 由于内存限制，脚本将以单线程模式运行，可能会非常耗时。\n")
# 仍然保留内存限制设置，因为它对主进程有益
options(future.globals.maxSize = 8000 * 1024^2) 
# ==============================================================================

# --- 2. 参数定义 (不变) ---
parser <- ArgumentParser(description = "使用Seurat TransferData（通过CSV输入）进行细胞类型注释")
parser$add_argument("--data_dir", type = "character", required = TRUE, help = "包含所有输入CSV文件的目录路径")
parser$add_argument("--celltype_key", type = "character", required = TRUE, help = "参考数据中包含细胞类型标签的.obs列名")
parser$add_argument("--out", type = "character", required = TRUE, help = "输出预测结果CSV文件的路径")
args <- parser$parse_args()

# --- 3. 主逻辑 (与之前几乎一样) ---
tryCatch({
  # ... (数据加载与清洗部分完全不变) ...
  cat("正在从CSV加载数据...\n")
  ref_counts <- readMM(file.path(args$data_dir, "ref_counts.mtx"))
  ref_meta <- read.csv(file.path(args$data_dir, "ref_meta.csv"), row.names = 1, check.names = FALSE)
  ref_genes <- read.csv(file.path(args$data_dir, "features.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
  ref_cells <- read.csv(file.path(args$data_dir, "ref_barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
  colnames(ref_counts) <- ref_cells
  rownames(ref_counts) <- ref_genes
  
  query_counts <- readMM(file.path(args$data_dir, "query_counts.mtx"))
  query_meta <- read.csv(file.path(args$data_dir, "query_meta.csv"), row.names = 1, check.names = FALSE)
  query_cells <- read.csv(file.path(args$data_dir, "query_barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
  colnames(query_counts) <- query_cells
  rownames(query_counts) <- ref_genes
  
  cat("正在清洗参考元数据...\n")
  ref_meta <- ref_meta[!is.na(ref_meta[[args$celltype_key]]), ]
  valid_ref_cells <- rownames(ref_meta)
  ref_counts <- ref_counts[, valid_ref_cells]
  
  # 创建Seurat对象
  cat("正在创建Seurat对象...\n")
  ref_seurat <- CreateSeuratObject(counts = ref_counts, meta.data = ref_meta)
  query_seurat <- CreateSeuratObject(counts = query_counts, meta.data = query_meta)
  
  # --- Seurat分析流程 (回归到使用高变基因，因为内存压力太大) ---
  cat("正在对参考数据进行标准化和特征选择...\n")
  ref_seurat <- NormalizeData(ref_seurat, verbose = FALSE)
  
  # ==================== 策略调整：减少内存消耗 ====================
  # 由于FindVariableFeatures在超大数据集上消耗巨大内存，
  # 我们先只对参考集做这一步，且使用默认的2000个高变基因。
  # 之前为了解决254个基因的问题，我们使用了所有基因，但现在基因数正常了。
  ref_seurat <- FindVariableFeatures(ref_seurat, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  # ==============================================================
  
  # 后续步骤使用高变基因，而不是所有基因，以降低内存和计算需求
  ref_seurat <- ScaleData(ref_seurat, features = VariableFeatures(object = ref_seurat), verbose = FALSE)
  ref_seurat <- RunPCA(ref_seurat, features = VariableFeatures(object = ref_seurat), verbose = FALSE)
  
  cat("正在对查询数据进行标准化...\n")
  query_seurat <- NormalizeData(query_seurat, verbose = FALSE)
  
  cat("正在寻找转移锚点 (FindTransferAnchors)...\n")
  # 锚点寻找也只使用高变基因，这能极大降低计算复杂度
  anchors <- FindTransferAnchors(
    reference = ref_seurat,
    query = query_seurat,
    dims = 1:30,
    reduction = "rpca",
    features = VariableFeatures(object = ref_seurat), # <-- 使用高变基因
    verbose = TRUE
  )
  
  # ... (后续代码不变) ...
  if (is.null(anchors) || length(anchors@anchors) == 0) {
    stop("关键错误: FindTransferAnchors未能找到任何锚点！无法进行预测。")
  }
  
  cat("正在转移数据并进行预测 (TransferData)...\n")
  predictions <- TransferData(
    anchorset = anchors,
    refdata = ref_seurat[[args$celltype_key, drop = TRUE]],
    dims = 1:30,
    # confidence.threshold = 0.2,
    verbose = FALSE
  )
  
  cat("预测完成，正在保存结果...\n")
  output_df <- predictions[, "predicted.id", drop = FALSE]
  output_df$cell_barcode <- rownames(output_df)
  write.csv(output_df, file = args$out, row.names = FALSE)
  
  cat("R脚本执行成功！\n")
  
}, error = function(e) {
  cat("R脚本执行时发生严重错误:\n")
  cat(e$message, "\n")
  quit(status = 1)
})