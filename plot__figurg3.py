"""
Assembles the composite integration figure for the Cytoneighbor paper.

Layout (Nature Methods double-column, ~7.2 inch wide):
┌────────────────────┬──────────────────────────┐
│                    │  B  Human Immune          │
│   A  UMAP Grid     ├──────────────────────────┤
│   (5 methods ×     │  C  Pancreas              │
│    Batch/CellType) ├──────────────────────────┤
│                    │  D  Diabetic Kidney Dis.  │
├────────────────────┴──────────────────────────┤
│           Performance Table (CSV)              │
└────────────────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Global style (Nature Methods) ────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'font.size':        7,
    'axes.titlesize':   8,
    'axes.labelsize':   7,
    'xtick.labelsize':  6,
    'ytick.labelsize':  6,
    'legend.fontsize':  6,
    'axes.linewidth':   0.6,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

# ── Colour palette ────────────────────────────────────────────────────────────
wong = {
    'vermilion':      '#D55E00',
    'blue':           '#0072B2',
    'sky_blue':       '#56B4E9',
    'orange':         '#E69F00',
    'bluish_green':   '#009E73',
    'yellow':         '#F0E442',
    'reddish_purple': '#CC79A7',
    'grey':           '#999999',
    'light_grey':     '#DDDDDD',
}

METHOD_COLOR = {
    'cytoneighbor':       wong['vermilion'],
    'scANVI':             wong['blue'],
    'scArches':           wong['sky_blue'],
    'Seurat':             wong['orange'],
    'Celltypist':         wong['bluish_green'],
    'SingleR':            wong['yellow'],
    'RandomForest':       wong['grey'],
    'SVM':                wong['grey'],
}

# ── Image paths ───────────────────────────────────────────────────────────────
IMG_UMAP  = 'E:/Project/Cytoneighbor_Project/results/CV_Rebatch/Human immune/FigA_UMAP_Grid.png'
IMG_B     = 'E:/Project/Cytoneighbor_Project/results/CV_Rebatch/Human immune/FigB_Pareto_Fron_real.png'       # Human Immune
IMG_C     = 'E:/Project/Cytoneighbor_Project/results/CV_Rebatch/FigB_Pareto_Front.png'   # Pancreas
IMG_D     = 'E:/Project/Cytoneighbor_Project/results/CV_Rebatch/FigB_Pareto_Front.png'   # DKD

# ── Load & clean CSV ──────────────────────────────────────────────────────────
def load_data(path):
    df_raw = pd.read_csv(path, header=None, low_memory=False)
    methods = df_raw.iloc[0].copy()
    for i, v in enumerate(['Dataset', 'Split_Key', 'Species', 'Fold']):
        methods.iloc[i] = v
    methods = methods.ffill()
    metrics = df_raw.iloc[1].copy()
    data = df_raw.iloc[2:].copy()
    flat = []
    for i, (m, metric) in enumerate(zip(methods, metrics)):
        if i < 4:
            flat.append(str(m))
        elif (pd.notna(m) and pd.notna(metric)
              and 'Unnamed' not in str(m) and 'Unnamed' not in str(metric)):
            flat.append(f'{m}__{metric}')
        else:
            flat.append(f'__DROP_{i}__')
    data.columns = flat
    meta = ['Dataset', 'Split_Key', 'Species', 'Fold']
    vcols = [c for c in flat if c not in meta and not c.startswith('__DROP_')]
    df = data.melt(id_vars=meta, value_vars=vcols,
                   var_name='MM', value_name='Score')
    df['Method'] = df['MM'].str.split('__').str[0]
    df['Metric'] = df['MM'].str.split('__').str[1]
    df['Score']  = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score']).drop(columns=['MM'])
    return df[['Dataset', 'Split_Key', 'Species', 'Fold', 'Method', 'Metric', 'Score']]

# ── Build table data ──────────────────────────────────────────────────────────
def build_table(df):
    METHODS  = ['cytoneighbor', 'scANVI', 'scArches', 'Seurat',
                'Celltypist', 'SingleR', 'RandomForest', 'SVM']
    DATASETS = ['Human immune', 'Pancreas', 'Diabetic Kidney Disease']
    METRICS  = ['accuracy', 'f1_macro']
    MLABEL   = {'accuracy': 'Acc.', 'f1_macro': 'F1'}

    sub = df[df['Metric'].isin(METRICS) &
             df['Method'].isin(METHODS) &
             df['Dataset'].isin(DATASETS)]
    agg = (sub.groupby(['Dataset', 'Method', 'Metric'])['Score']
              .mean().reset_index())

    # Pivot: rows = Method, columns = (Dataset, Metric)
    pivot = agg.pivot_table(index='Method', columns=['Dataset', 'Metric'],
                            values='Score')
    pivot = pivot.reindex(METHODS)                  # enforce row order
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product(
        [DATASETS, METRICS]))                        # enforce col order

    # Flatten column names for display
    col_labels = [f'{MLABEL[m]}' for (_, m) in pivot.columns]
    return pivot, DATASETS, METRICS, MLABEL, col_labels

# ── Draw table panel ──────────────────────────────────────────────────────────
def draw_table(ax, pivot, datasets, metrics, mlabel):
    ax.set_axis_off()

    METHODS  = list(pivot.index)
    DATASETS = datasets
    n_methods  = len(METHODS)
    n_datasets = len(DATASETS)
    n_metrics  = len(metrics)
    n_cols = n_datasets * n_metrics  # total data columns

    # layout params
    left_w   = 0.18   # method name column width (fraction of axes)
    data_w   = (1.0 - left_w) / n_cols
    row_h    = 1.0 / (n_methods + 2.5)   # +2.5 for two header rows + padding
    header_h = row_h * 1.2

    colors_bg = ['#F7F7F7', '#FFFFFF']   # alternating row colours

    # ── Draw dataset super-headers ────────────────────────────────────────────
    for di, ds in enumerate(DATASETS):
        x = left_w + di * n_metrics * data_w
        w = n_metrics * data_w
        y = 1.0 - header_h * 0.5
        ax.add_patch(FancyBboxPatch(
            (x + 0.002, y - header_h * 0.45), w - 0.004, header_h * 0.88,
            boxstyle='round,pad=0.005', linewidth=0.5,
            edgecolor='#AAAAAA', facecolor='#E8F0F8',
            transform=ax.transAxes, clip_on=False))
        ax.text(x + w / 2, y, ds,
                ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='#1a3a5c',
                transform=ax.transAxes)

    # ── Draw metric sub-headers ───────────────────────────────────────────────
    for di in range(n_datasets):
        for mi, metric in enumerate(metrics):
            col = di * n_metrics + mi
            x   = left_w + col * data_w + data_w / 2
            y   = 1.0 - header_h * 1.25
            ax.text(x, y, mlabel[metric],
                    ha='center', va='center', fontsize=6,
                    color='#444444', fontstyle='italic',
                    transform=ax.transAxes)

    # ── "Method" column header ─────────────────────────────────────────────────
    ax.text(left_w / 2, 1.0 - header_h * 0.5, 'Method',
            ha='center', va='center', fontsize=6.5,
            fontweight='bold', color='#1a3a5c',
            transform=ax.transAxes)

    # ── Find per-column max for bolding ───────────────────────────────────────
    col_max = {}
    for di, ds in enumerate(DATASETS):
        for mi, metric in enumerate(metrics):
            vals = pivot[(ds, metric)]
            col_max[(di, mi)] = vals.max()

    # ── Draw data rows ────────────────────────────────────────────────────────
    for ri, method in enumerate(METHODS):
        y_top = 1.0 - header_h * 2.0 - ri * row_h
        bg    = colors_bg[ri % 2]

        # row background
        ax.add_patch(FancyBboxPatch(
            (0.002, y_top - row_h + 0.003), 0.996, row_h - 0.005,
            boxstyle='round,pad=0.003', linewidth=0,
            facecolor=bg, transform=ax.transAxes, clip_on=False))

        # Method name (coloured dot + label)
        dot_x = 0.018
        dot_y = y_top - row_h / 2
        c     = METHOD_COLOR.get(method, '#888888')

        # highlight cytoneighbor row
        if method == 'cytoneighbor':
            ax.add_patch(FancyBboxPatch(
                (0.002, y_top - row_h + 0.003), 0.996, row_h - 0.005,
                boxstyle='round,pad=0.003', linewidth=0.8,
                edgecolor=wong['vermilion'], facecolor='#FFF0EC',
                transform=ax.transAxes, clip_on=False))

        ax.plot(dot_x, dot_y, 'o', color=c, markersize=4,
                transform=ax.transAxes, clip_on=False)
        ax.text(dot_x + 0.022, dot_y, method,
                ha='left', va='center', fontsize=6,
                fontweight='bold' if method == 'cytoneighbor' else 'normal',
                color='#222222', transform=ax.transAxes)

        # Data cells
        for di, ds in enumerate(DATASETS):
            for mi, metric in enumerate(metrics):
                col  = di * n_metrics + mi
                x    = left_w + col * data_w + data_w / 2
                val  = pivot.loc[method, (ds, metric)]
                is_best = (val == col_max[(di, mi)])
                txt  = f'{val:.3f}' if pd.notna(val) else '—'
                fw   = 'bold' if is_best else 'normal'
                fc   = wong['vermilion'] if (is_best and method == 'cytoneighbor') else (
                       '#0a4e8a' if is_best else '#333333')
                ax.text(x, dot_y, txt,
                        ha='center', va='center', fontsize=5.8,
                        fontweight=fw, color=fc,
                        transform=ax.transAxes)

    # ── Column separator lines ─────────────────────────────────────────────────
    top_y    = 1.0 - header_h * 1.65
    bottom_y = 1.0 - header_h * 2.0 - n_methods * row_h
    for di in range(1, n_datasets):
        x = left_w + di * n_metrics * data_w
        ax.plot([x, x], [bottom_y, top_y],
                color='#CCCCCC', linewidth=0.5,
                transform=ax.transAxes, clip_on=False)

    # ── Bottom border line ────────────────────────────────────────────────────
    ax.plot([0.0, 1.0], [bottom_y + 0.01, bottom_y + 0.01],
            color='#AAAAAA', linewidth=0.6,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.0, 1.0], [top_y + row_h * 0.3, top_y + row_h * 0.3],
            color='#AAAAAA', linewidth=0.6,
            transform=ax.transAxes, clip_on=False)

    # ── Caption below table ───────────────────────────────────────────────────
    ax.text(0.5, bottom_y - 0.06,
            'Bold values indicate the best score per column. '
            'Values are means across all folds within each dataset.',
            ha='center', va='top', fontsize=5, color='#555555',
            transform=ax.transAxes, style='italic')

# ── Main assembly ─────────────────────────────────────────────────────────────
def main():
    df = load_data("C:/Users/86188/Desktop/批次整合任务.csv")
    pivot, datasets, metrics, mlabel, _ = build_table(df)

    # Figure: 7.2 × 9.5 inches  (double-column Nature Methods)
    fig = plt.figure(figsize=(7.2, 9.5), dpi=300)

    # Outer grid: 2 rows  (top = images, bottom = table)
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3.6, 1.0],
        hspace=0.08,
    )

    # ── Top section: Panel A (left) + Panels B/C/D (right) ───────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1, 1.85],
        wspace=0.04,
    )

    # Panel A — UMAP grid
    ax_a = fig.add_subplot(top_gs[0])
    img_a = mpimg.imread(IMG_UMAP)
    ax_a.imshow(img_a)
    ax_a.set_axis_off()
    ax_a.text(0.02, 0.995, 'A', transform=ax_a.transAxes,
              fontsize=10, fontweight='bold', va='top', color='black')

    # Panels B / C / D — Pareto fronts (stacked vertically)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=top_gs[1],
        hspace=0.03,
    )
    pareto_imgs  = [IMG_B,     IMG_C,        IMG_D]
    pareto_labels = ['B', 'C', 'D']
    pareto_titles = [
        'Human Immune dataset',
        'Pancreas dataset',
        'Diabetic Kidney Disease dataset',
    ]

    for i, (img_path, label, title) in enumerate(
            zip(pareto_imgs, pareto_labels, pareto_titles)):
        ax = fig.add_subplot(right_gs[i])
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_axis_off()
        # Panel letter
        ax.text(0.01, 0.97, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', color='black')

    # ── Bottom section: Table ─────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(outer[1])
    draw_table(ax_tbl, pivot, datasets, metrics, mlabel)

    # Table panel letter
    ax_tbl.text(0.005, 0.98, 'E',
                transform=ax_tbl.transAxes,
                fontsize=10, fontweight='bold', va='top', color='black')
    ax_tbl.text(0.025, 0.98,
                'Cell-type annotation performance across datasets '
                '(mean Accuracy and Macro-F1)',
                transform=ax_tbl.transAxes,
                fontsize=7, fontweight='bold', va='top', color='#1a3a5c')

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = 'E:\\Project\\Cytoneighbor_Project\\Paper_Figures\\Figure_Integration_Composite.pdf'
    out_png = 'E:\\Project\\Cytoneighbor_Project\\Paper_Figures\\Figure_Integration_Composite.png'
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f'Saved:/n  {out_pdf}/n  {out_png}')

if __name__ == '__main__':
    main()