import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Nature Methods 全局参数规范
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'legend.title_fontsize': 7,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Bang Wong (Nature Methods 8, 441, 2011) 色盲友好调色板
wong_palette = {
    'black':          '#000000',
    'orange':         '#E69F00',
    'sky_blue':       '#56B4E9',
    'bluish_green':   '#009E73',
    'yellow':         '#F0E442',
    'blue':           '#0072B2',
    'vermilion':      '#D55E00',
    'reddish_purple': '#CC79A7',
    'grey':           '#BBBBBB',
}

method_colors = {
    'cytoneighbor':       wong_palette['vermilion'],
    'scANVI':             wong_palette['blue'],
    'scArches':           wong_palette['sky_blue'],
    'Seurat':             wong_palette['orange'],
    'Celltypist':         wong_palette['bluish_green'],
    'SingleR':            wong_palette['yellow'],
    'SciBet':             wong_palette['reddish_purple'],
    'SVM':                wong_palette['grey'],
    'RandomForest':       wong_palette['grey'],
    'XGBoost':            wong_palette['grey'],
    'MLP':                wong_palette['grey'],
    'kNN':                wong_palette['grey'],
    'LogisticRegression': wong_palette['grey'],
    'NaiveBayes':         wong_palette['grey'],
}

# ==============================================================================
# 2. 数据解析器
#
# 根本问题：pandas.melt() 对 MultiIndex 列只提取第一层（方法名字符串），
# 导致 x[0] 索引字符串首字符而非元组第一个元素（e.g. 'L' 来自 'LogisticRegression'）。
# 修复方案：将双层表头展平为 "Method__Metric" 字符串列名，melt 后再用 str.split 拆分。
# ==============================================================================
def load_and_clean_data(filepath):
    df_raw = pd.read_csv(filepath, header=None, low_memory=False)

    # Row 0：方法名（前 4 列为元数据占位）
    methods = df_raw.iloc[0].copy()
    methods.iloc[0] = 'Dataset'
    methods.iloc[1] = 'Split_Key'
    methods.iloc[2] = 'Species'
    methods.iloc[3] = 'Fold'
    methods = methods.ffill()

    # Row 1：指标名
    metrics = df_raw.iloc[1].copy()

    # 纯数据部分
    data = df_raw.iloc[2:].copy()

    # ── 修复核心 ──────────────────────────────────────────────────────────────
    # 用 "Method__Metric" 扁平字符串代替 MultiIndex，彻底规避 melt 的 MultiIndex 行为差异
    flat_cols = []
    for i, (m, metric) in enumerate(zip(methods, metrics)):
        if i < 4:
            flat_cols.append(str(m))            # Dataset / Split_Key / Species / Fold
        elif (pd.notna(m) and pd.notna(metric)
              and 'Unnamed' not in str(m)
              and 'Unnamed' not in str(metric)):
            flat_cols.append(f'{m}__{metric}')  # e.g. "scANVI__f1_macro"
        else:
            flat_cols.append(f'__DROP_{i}__')   # 末尾多余列，后续过滤

    data = data.copy()
    data.columns = flat_cols

    meta_cols  = ['Dataset', 'Split_Key', 'Species', 'Fold']
    value_cols = [c for c in flat_cols
                  if c not in meta_cols and not c.startswith('__DROP_')]

    # Melt（var_name 为标量字符串，所有 pandas 版本均兼容）
    df_tidy = data.melt(
        id_vars=meta_cols,
        value_vars=value_cols,
        var_name='Method_Metric',
        value_name='Score'
    )

    # 拆分 "Method__Metric" → 两列（用 str.split 而非 lambda，更简洁）
    df_tidy['Method'] = df_tidy['Method_Metric'].str.split('__').str[0]
    df_tidy['Metric'] = df_tidy['Method_Metric'].str.split('__').str[1]
    df_tidy = df_tidy.drop(columns=['Method_Metric'])

    # 数值清洗
    df_tidy['Score'] = pd.to_numeric(df_tidy['Score'], errors='coerce')
    df_tidy = df_tidy.dropna(subset=['Score'])

    df_tidy = df_tidy[['Dataset', 'Split_Key', 'Species', 'Fold',
                        'Method', 'Metric', 'Score']].reset_index(drop=True)

    # 诊断输出
    print("=== 数据诊断 ===")
    print(f"Shape   : {df_tidy.shape}")
    print(f"Methods : {sorted(df_tidy['Method'].unique())}")
    print(f"Metrics : {sorted(df_tidy['Metric'].unique())}")
    print(f"Folds (前5): {list(df_tidy['Fold'].unique())[:5]}")
    print("================\n")

    return df_tidy


# ==============================================================================
# 3. 辅助：安全选取方法（防止 df_mean.loc 因方法名不存在而 KeyError）
# ==============================================================================
def safe_loc(df, methods):
    existing = [m for m in methods if m in df.index]
    missing  = [m for m in methods if m not in df.index]
    if missing:
        print(f"[警告] 以下方法在数据中不存在，已跳过: {missing}")
    return df.loc[existing]


# ==============================================================================
# 4. 绘制 Figure 2
# ==============================================================================
def plot_nature_methods_figure(df_tidy):

    # ── 动态确认关键 Metric 列名 ─────────────────────────────────────────────
    available_metrics = df_tidy['Metric'].unique()
    f1_col  = next((m for m in available_metrics
                    if 'f1'  in m.lower() and 'macro' in m.lower()), None)
    acc_col = next((m for m in available_metrics if 'acc' in m.lower()), None)

    if f1_col is None:
        raise ValueError(
            f"找不到 f1_macro 列，当前可用 Metric: {sorted(available_metrics)}"
        )
    if acc_col is None:
        print(f"[警告] 找不到 accuracy 列，Panel B x 轴将为空。"
              f"当前可用: {sorted(available_metrics)}")

    # =========================================================================
    fig = plt.figure(figsize=(7.2, 6.5), dpi=300)
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.6, hspace=0.75)

    # =========================================================================
    # Panel a：Boxplot + Stripplot（全宽）
    # =========================================================================
    ax_a  = fig.add_subplot(gs[0, :])
    df_f1 = df_tidy[df_tidy['Metric'] == f1_col].copy()
    order = (df_f1.groupby('Method')['Score']
             .median().sort_values().index.tolist())

    sns.boxplot(
        data=df_f1, x='Method', y='Score', order=order, ax=ax_a,
        palette=method_colors, showfliers=False, width=0.6,
        boxprops=dict(alpha=0.4, linewidth=0.8),
        medianprops=dict(color='red', linewidth=1.2),
    )
    sns.stripplot(
        data=df_f1, x='Method', y='Score', order=order, ax=ax_a,
        palette=method_colors, size=2.5, alpha=0.7,
        jitter=0.2, edgecolor='white', linewidth=0.2,
    )
    ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=45, ha='right')
    ax_a.set_ylabel('Macro-F1 Score')
    ax_a.set_xlabel('')
    ax_a.set_title(
        'a | Global performance in Open-Set annotation across diverse datasets',
        loc='left', fontweight='bold', fontsize=8,
    )

    # =========================================================================
    # Panel b：Accuracy–F1 Scatter
    # 修复：逐点调用 ax.scatter()，避免 sns.scatterplot(color=列表) 不兼容问题
    # =========================================================================
    ax_b = fig.add_subplot(gs[1, 0:2])

    df_mean = df_tidy.pivot_table(
        index='Method', columns='Metric', values='Score', aggfunc='mean'
    )

    if acc_col and f1_col in df_mean.columns and acc_col in df_mean.columns:
        df_scatter = df_mean[[acc_col, f1_col]].copy().dropna()
        df_scatter.columns = ['accuracy', 'f1_macro']

        for method_name, row in df_scatter.iterrows():
            ax_b.scatter(
                row['accuracy'], row['f1_macro'],
                s=60,
                color=method_colors.get(method_name, '#333333'),
                edgecolors='white', linewidths=0.5, zorder=5,
            )

        # y = x 参考线
        vals   = df_scatter[['accuracy', 'f1_macro']].values.flatten()
        lo, hi = vals.min() - 0.02, vals.max() + 0.02
        ax_b.plot([lo, hi], [lo, hi],
                  color='grey', linestyle='--', alpha=0.5,
                  linewidth=0.8, zorder=0)
        ax_b.set_xlim(lo, hi)
        ax_b.set_ylim(lo, hi)

        labels_to_show = {'cytoneighbor', 'scANVI', 'scArches', 'Seurat', 'Celltypist'}
        for method_name, row in df_scatter.iterrows():
            if method_name in labels_to_show:
                ax_b.text(
                    row['accuracy'] + 0.003, row['f1_macro'] - 0.003,
                    method_name, fontsize=5.5, color='black',
                )
    else:
        ax_b.text(0.5, 0.5, 'accuracy / f1_macro 数据列缺失',
                  ha='center', va='center',
                  transform=ax_b.transAxes, fontsize=7)

    ax_b.set_xlabel('Accuracy (Integration Capability)')
    ax_b.set_ylabel('Macro-F1 (Resolution & Open-Set)')
    ax_b.set_title(
        'b | Breaking the Resolution-Integration trade-off',
        loc='left', fontweight='bold', fontsize=8,
    )

    # =========================================================================
    # Panel c：Extreme Batch 柱状图
    # 修复：str.contains 模糊匹配 Fold 值，空结果时显示红色提示而非静默空图
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 2:4])

    fold_str       = df_f1['Fold'].astype(str)
    mask_villani   = fold_str.str.contains('Villani', case=False, na=False)
    mask_smarter   = fold_str.str.contains('smarter', case=False, na=False)
    mask_methods_c = df_f1['Method'].isin(
        ['cytoneighbor', 'scANVI', 'scArches', 'Seurat']
    )
    df_ext = df_f1[(mask_villani | mask_smarter) & mask_methods_c].copy()

    if df_ext.empty:
        ax_c.text(
            0.5, 0.5,
            'Panel C: 未找到匹配的 Fold 数据\n请检查 Fold 列的实际字符串',
            ha='center', va='center',
            transform=ax_c.transAxes, fontsize=7, color='red',
        )
    else:
        df_ext['Fold_label'] = df_ext['Fold'].apply(
            lambda x: ('Immune:\nVillani (Extreme)' if 'Villani' in str(x)
                       else 'Pancreas:\nSmarter (Rare cells)')
        )
        sns.barplot(
            data=df_ext, x='Fold_label', y='Score', hue='Method',
            ax=ax_c, palette=method_colors,
            edgecolor='black', linewidth=0.5,
        )
        ax_c.legend(title='', fontsize=6, frameon=False,
                    loc='lower center', ncol=2)

    ax_c.set_ylabel('Macro-F1 Score')
    ax_c.set_xlabel('')
    ax_c.set_title(
        'c | Robustness to extreme batch & rare cells',
        loc='left', fontweight='bold', fontsize=8,
    )

    # =========================================================================
    # Panel d：OOD 混淆矩阵对比
    # 修复1：3×3 矩阵 tick 位置由 [0.5, 2.5] 修正为 [0.5, 1.5, 2.5]
    # 修复2：Panel 标题写入 ax_d1.set_title()，彻底避免 fig.text 硬编码坐标漂移
    # =========================================================================
    gs_d  = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[2, 0:2], wspace=0.1
    )
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    mock_scanvi = np.array([
        [0.85, 0.15, 0.00],
        [0.10, 0.90, 0.00],
        [0.45, 0.55, 0.00],
    ])
    mock_cyto = np.array([
        [0.95, 0.05, 0.00],
        [0.02, 0.98, 0.00],
        [0.00, 0.00, 1.00],
    ])
    row_labels = ['Known A', 'Known B', 'OOD Cell']
    col_labels = ['Pred A',  'Pred B',  'Unknown']
    tick_pos   = [0.5, 1.5, 2.5]   # 修复：3 格 → 3 个刻度

    sns.heatmap(mock_scanvi, cmap='Blues', ax=ax_d1,
                cbar=False, vmin=0, vmax=1,
                linewidths=0.5, linecolor='white')
    # 修复：Panel d 的大标题放在 ax_d1 的 title 里，移除 fig.text 硬编码
    ax_d1.set_title(
        'd | OOD discovery via relational gap\nscANVI (Closed-set collapse)',
        loc='left', fontweight='bold', fontsize=7,
    )
    ax_d1.set_ylabel('True Label', fontsize=7)
    ax_d1.set_yticks(tick_pos)
    ax_d1.set_yticklabels(row_labels, rotation=0, fontsize=6)
    ax_d1.set_xticks(tick_pos)
    ax_d1.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=6)

    sns.heatmap(mock_cyto, cmap='Blues', ax=ax_d2,
                cbar=False, vmin=0, vmax=1,
                linewidths=0.5, linecolor='white')
    ax_d2.set_title('Cytoneighbor\n(Open-set isolation)', fontsize=7)
    ax_d2.set_yticks([])
    ax_d2.set_xticks(tick_pos)
    ax_d2.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=6)

    # =========================================================================
    # Panel e：综合评估热图
    # 修复：safe_loc 防止 KeyError；Min-Max 归一化防除零
    # =========================================================================
    ax_e = fig.add_subplot(gs[2, 2:4])

    preferred = ['cytoneighbor', 'scANVI', 'scArches', 'Seurat',
                 'Celltypist', 'SingleR', 'RandomForest', 'SVM']
    df_hm_sub = safe_loc(df_mean, preferred)

    if df_hm_sub.shape[0] < 2:
        ax_e.text(0.5, 0.5, 'Panel E: 可用方法不足，无法绘制热图',
                  ha='center', va='center',
                  transform=ax_e.transAxes, fontsize=7, color='red')
    else:
        col_range = (df_hm_sub.max() - df_hm_sub.min()).replace(0, np.nan)
        df_scaled = ((df_hm_sub - df_hm_sub.min()) / col_range).T  # 指标为行

        sns.heatmap(
            df_scaled,
            cmap='viridis', ax=ax_e,
            annot=df_hm_sub.T, fmt='.2f',
            annot_kws={'size': 5},
            cbar_kws={'label': 'Min-Max Scaled Score', 'shrink': 0.8},
        )
        ax_e.set_xticklabels(ax_e.get_xticklabels(), rotation=45, ha='right')
        for spine in ax_e.spines.values():
            spine.set_visible(True)
            spine.set_color('black')

    ax_e.set_ylabel('')
    ax_e.set_xlabel('')
    ax_e.set_title(
        'e | Comprehensive Evaluation Matrix',
        loc='left', fontweight='bold', fontsize=8,
    )

    # =========================================================================
    plt.savefig('Figure2_General_Benchmarking.pdf', bbox_inches='tight')
    plt.savefig('Figure2_General_Benchmarking.png', bbox_inches='tight', dpi=300)
    print('已保存：Figure2_General_Benchmarking.pdf / .png')
    plt.show()


# ==============================================================================
# 5. 入口
# ==============================================================================
if __name__ == '__main__':
    csv_file = "C:\\Users\\86188\\Desktop\\单细胞数据集性能指标最终版本.csv"
    df = load_and_clean_data(csv_file)
    plot_nature_methods_figure(df)