import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. SciFont block — adjust here for fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
try:
    from scifont import set_font
    set_font(font_name="Helvetica")
    print("[Success] SciFont loaded. Publication-ready fonts enabled.")
except ImportError:
    print("[Warning] 'scifont' not found. Falling back to rcParams.")
    print("Consider: pip install scifont")
    plt.rcParams.update({
        'font.family':    'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype':   42,   # TrueType, editable in Illustrator
        'ps.fonttype':    42,
        'svg.fonttype':   'none',
    })

# Global font & line-weight settings (Nature Methods spec)
plt.rcParams.update({
    'font.size':             7,
    'axes.titlesize':        8,
    'axes.labelsize':        7,
    'xtick.labelsize':       6,
    'ytick.labelsize':       6,
    'legend.fontsize':       6,
    'legend.title_fontsize': 7,
    'lines.linewidth':       1.0,
    'axes.linewidth':        0.6,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'pdf.fonttype':          42,
    'ps.fonttype':           42,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. Bang Wong colour palette (Nature Methods 8:441, 2011)
# ══════════════════════════════════════════════════════════════════════════════
WONG = {
    'black':          '#000000',
    'orange':         '#E69F00',
    'sky_blue':       '#56B4E9',
    'bluish_green':   '#009E73',
    'yellow':         '#F0E442',
    'blue':           '#0072B2',
    'vermilion':      '#D55E00',
    'reddish_purple': '#CC79A7',
    'grey':           '#999999',
}

METHOD_COLOR = {
    'Cytoneighbor': WONG['vermilion'],
    'scVI':         WONG['blue'],
    'Seurat':       WONG['orange'],
    'Harmony':      WONG['grey'],
}

METHOD_ORDER = ['Cytoneighbor', 'scVI', 'Seurat', 'Harmony']

# ══════════════════════════════════════════════════════════════════════════════
# 2. 修正后数据 —— 100%匹配 summary_report.txt 原文
# ══════════════════════════════════════════════════════════════════════════════

# Panel E — NKT misclassification rates (%)
NKT_MISC = {
    'Cytoneighbor': {'NKT→NK':  0.7,  'NKT→CD8': 4.2},
    'scVI':         {'NKT→NK':  6.2,  'NKT→CD8': 0.5},
    'Seurat':       {'NKT→NK':  4.2,  'NKT→CD8': 0.9},
    'Harmony':      {'NKT→NK':  4.7,  'NKT→CD8': 0.5},
}

# Panel D — per-class F1 scores (from report)
PER_CLASS_F1 = {
    'Cytoneighbor': [0.9195, 0.5904, 0.8338, 0.8129],
    'scVI':         [0.9309, 0.6880, 0.9008, 0.8468],
    'Seurat':       [0.9516, 0.8256, 0.9302, 0.8743],
    'Harmony':      [0.9221, 0.6537, 0.8507, 0.7760],
}
CELL_TYPE_LABELS = ['CD4+\nT cells', 'CD8+\nT cells', 'NK\ncells', 'NKT\ncells']

# Panel F — robustness: performance drop Easy→Hard (%)
PERF_DROP = {
    'Cytoneighbor': 11.6,
    'scVI':          9.0,
    'Seurat':        8.1,
    'Harmony':      15.8,
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. Image paths  ← PDF 格式，直接指向原始文件
# ══════════════════════════════════════════════════════════════════════════════
IMG_UMAP_GLOBAL = r'E:\Project\备份\1\benchmark_resolution7new\umap_comparison_global_pin版.pdf'
IMG_UMAP_ZOOM   = r'E:\Project\备份\1\benchmark_resolution7new\umap_zoom_nk_nkt_pin版.pdf'
IMG_CONFUSION   = r'E:\Project\备份\1\benchmark_resolution7new\confusion_matrix_comparison_pin版.pdf'
IMG_F1_GRAD     = r'E:\Project\备份\1\benchmark_resolution7new\f1_score_gradient_pin版.pdf'

# ══════════════════════════════════════════════════════════════════════════════
# 3b. PDF → numpy array  (尝试三个后端，确保跨平台可用)
# ══════════════════════════════════════════════════════════════════════════════
def load_pdf_as_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """
    将单页 PDF 渲染为 RGB numpy 数组。
    依次尝试四个后端，第一个成功即返回：
      1. pdftoppm   — 系统工具 (Windows: 安装 Poppler 并加入 PATH)
      2. pdf2image  — pip install pdf2image  (封装 poppler)
      3. pymupdf    — pip install pymupdf    ★ 无需系统依赖，推荐
      4. PIL        — 部分 PDF 可直接打开

    Windows 用户推荐：pip install pymupdf
    """
    import subprocess, tempfile, glob
    from PIL import Image as PILImage

    # ── 后端 1: pdftoppm (系统 Poppler) ──────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_prefix = os.path.join(tmp, 'page')
            result = subprocess.run(
                ['pdftoppm', '-r', str(dpi), '-png', pdf_path, out_prefix],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                pages = sorted(glob.glob(f'{out_prefix}*.png'))
                if pages:
                    img = np.array(PILImage.open(pages[0]).convert('RGB'))
                    print(f"[pdftoppm]   {os.path.basename(pdf_path)} → {img.shape}")
                    return img
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        print(f"[pdftoppm] failed: {e}")

    # ── 后端 2: pdf2image (pip install pdf2image) ─────────────────────────────
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        img = np.array(pages[0].convert('RGB'))
        print(f"[pdf2image]  {os.path.basename(pdf_path)} → {img.shape}")
        return img
    except ImportError:
        pass
    except Exception as e:
        print(f"[pdf2image] failed: {e}")

    # ── 后端 3: pymupdf ───────────────────────────────────────────────────────
    # 新版 PyMuPDF (≥1.24) 的模块名从 `fitz` 改为 `pymupdf`，需两者都尝试
    _fitz = None
    for _mod_name in ('pymupdf', 'fitz'):      # 新版优先，旧版兜底
        try:
            import importlib
            _fitz = importlib.import_module(_mod_name)
            break
        except ImportError:
            continue

    if _fitz is not None:
        try:
            doc  = _fitz.open(pdf_path)
            page = doc[0]
            mat  = _fitz.Matrix(dpi / 72, dpi / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                       pix.height, pix.width, 3)
            doc.close()
            print(f"[pymupdf]    {os.path.basename(pdf_path)} → {img.shape}")
            return img
        except Exception as e:
            print(f"[pymupdf] failed: {e}")
    else:
        print("[pymupdf] not installed — skipping")

    # ── 后端 4: PIL 直接打开（部分 PDF 支持）─────────────────────────────────
    try:
        img = np.array(PILImage.open(pdf_path).convert('RGB'))
        print(f"[PIL]        {os.path.basename(pdf_path)} → {img.shape}")
        return img
    except Exception as e:
        raise RuntimeError(
            f"无法渲染 PDF: {pdf_path}\n"
            f"请安装任一后端：\n"
            f"  pip install pymupdf          (推荐，无系统依赖)\n"
            f"  pip install pdf2image        (需系统安装 Poppler)\n"
            f"原始错误: {e}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# 4. Helper: add panel letter
# ══════════════════════════════════════════════════════════════════════════════
def panel_label(ax, letter, x=-0.03, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', color='black',
            clip_on=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Panel E — NKT misclassification grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def draw_nkt_misc(ax):
    methods  = METHOD_ORDER
    nk_vals  = [NKT_MISC[m]['NKT→NK']  for m in methods]
    cd8_vals = [NKT_MISC[m]['NKT→CD8'] for m in methods]

    x    = np.arange(len(methods))
    w    = 0.35
    clrs = [METHOD_COLOR[m] for m in methods]

    # NK bars (solid) & CD8 bars (hatched)
    bars_nk  = ax.bar(x - w/2, nk_vals,  width=w, color=clrs,
                      alpha=0.85, linewidth=0.5, edgecolor='white',
                      label='NKT→NK')
    bars_cd8 = ax.bar(x + w/2, cd8_vals, width=w, color=clrs,
                      alpha=0.45, linewidth=0.5, edgecolor='white',
                      hatch='//', label='NKT→CD8')

    # Value annotations
    for bar in list(bars_nk) + list(bars_cd8):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=4.8, color='#333333')

    # 适配报告结论：Seurat为NKT最优，调整标注
    best_idx = methods.index('Seurat')
    ax.annotate('', xy=(bars_nk[best_idx].get_x() + bars_nk[best_idx].get_width()/2,
                         max(nk_vals[best_idx], cd8_vals[best_idx]) + 2.5),
                xytext=(bars_nk[best_idx].get_x() + bars_nk[best_idx].get_width()/2,
                        max(nk_vals[best_idx], cd8_vals[best_idx]) + 0.8),
                arrowprops=dict(arrowstyle='->', color=WONG['orange'],
                                lw=0.8))
    ax.text(bars_nk[best_idx].get_x() + bars_nk[best_idx].get_width()/2,
            max(nk_vals[best_idx], cd8_vals[best_idx]) + 3.0, 'Best',
            ha='center', va='bottom', fontsize=5,
            color=WONG['orange'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=5.5)
    ax.set_ylabel('Misclassification Rate (%)', fontsize=6)
    ax.set_title('NKT Cell\nMisclassification', fontsize=7,
                 fontweight='bold', pad=3)
    ax.set_ylim(0, 8)  # 适配修正后最大数值7，避免留白过多
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

    # Legend
    patch_nk  = mpatches.Patch(facecolor='#888888', alpha=0.85, label='NKT→NK')
    patch_cd8 = mpatches.Patch(facecolor='#888888', alpha=0.45,
                                hatch='//', label='NKT→CD8')
    ax.legend(handles=[patch_nk, patch_cd8], fontsize=5,
              frameon=False, loc='upper right')

    panel_label(ax, 'E')

# ══════════════════════════════════════════════════════════════════════════════
# 6. Panel F — performance drop (robustness) horizontal bar
# ══════════════════════════════════════════════════════════════════════════════
def draw_perf_drop(ax):
    methods = METHOD_ORDER
    drops   = [PERF_DROP[m] for m in methods]
    colors  = [METHOD_COLOR[m] for m in methods]
    y       = np.arange(len(methods))

    bars = ax.barh(y, drops, height=0.55, color=colors,
                   alpha=0.82, linewidth=0.5, edgecolor='white')

    # Value labels
    for bar, v in zip(bars, drops):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}%', va='center', ha='left',
                fontsize=5, color='#333333')

    # 适配报告结论：Seurat为最稳健（下降最小），调整标注
    best_idx = methods.index('Seurat')
    ax.annotate('Smallest drop\n= Most robust',
                xy=(drops[best_idx] + 0.2, y[best_idx]),
                xytext=(drops[best_idx] + 4, y[best_idx] + 0.6),
                arrowprops=dict(arrowstyle='->', color=WONG['orange'], lw=0.8),
                fontsize=4.8, color=WONG['orange'], fontweight='bold',
                ha='center')

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=5.5)
    ax.set_xlabel('F1 Drop Easy→Hard (%)', fontsize=6)
    ax.set_title('Robustness\n(F1 Performance Drop)', fontsize=7,
                 fontweight='bold', pad=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 18)  # 适配修正后最大数值15.8，避免留白过多
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))

    panel_label(ax, 'F', x=-0.12)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Main assembly
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Nature Methods double-column: 180 mm wide, ≤ 240 mm tall
    W_IN = 180 / 25.4
    H_IN = 234 / 25.4

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)

    # ── Outer grid: 4 rows ──────────────────────────────────────────────────
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.15],
        hspace=0.15,
    )

    # ── Panel A: Global UMAP ─────────────────────────────────────────────────
    ax_a = fig.add_subplot(outer[0])
    img_a = load_pdf_as_image(IMG_UMAP_GLOBAL)
    ax_a.imshow(img_a)
    ax_a.set_aspect(img_a.shape[1] / img_a.shape[0])  # 防拉伸
    ax_a.set_axis_off()
    panel_label(ax_a, 'A')

    # ── Panel B: NK/NKT Zoom UMAP ────────────────────────────────────────────
    ax_b = fig.add_subplot(outer[1])
    img_b = load_pdf_as_image(IMG_UMAP_ZOOM)
    ax_b.imshow(img_b)
    ax_b.set_aspect(img_b.shape[1] / img_b.shape[0])  # 防拉伸
    ax_b.set_axis_off()
    panel_label(ax_b, 'B')

    # ── Panel C: Confusion Matrices ──────────────────────────────────────────
    ax_c = fig.add_subplot(outer[2])
    img_c = load_pdf_as_image(IMG_CONFUSION)
    ax_c.imshow(img_c)
    ax_c.set_aspect(img_c.shape[1] / img_c.shape[0])  # 防拉伸
    ax_c.set_axis_off()
    panel_label(ax_c, 'C')

    # ── Bottom row: D (F1 gradient image) + E (NKT misc) + F (perf drop) ────
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[3],
        width_ratios=[1.55, 1.0, 1.0],
        wspace=0.45,
    )

    # Panel D: F1 gradient (existing figure)
    ax_d = fig.add_subplot(bottom_gs[0])
    img_d = load_pdf_as_image(IMG_F1_GRAD)
    ax_d.imshow(img_d)
    ax_d.set_aspect(img_d.shape[1] / img_d.shape[0])  # 防拉伸
    ax_d.set_axis_off()
    panel_label(ax_d, 'D')

    # Panel E: NKT misclassification bar chart (generated from data)
    ax_e = fig.add_subplot(bottom_gs[1])
    draw_nkt_misc(ax_e)

    # Panel F: Robustness / performance drop bar chart (generated from data)
    ax_f = fig.add_subplot(bottom_gs[2])
    draw_perf_drop(ax_f)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_pdf = 'E:\\Project\备份\\1\\benchmark_resolution7new\\figure_experiment1B.pdf'
    out_png = 'E:\\Project\备份\\1\\benchmark_resolution7new\\figure_experiment1B.png'
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=300)

    from PIL import Image as PILImage
    im   = PILImage.open(out_png)
    w, h = im.size
    print(f"\nSaved : {out_pdf}")
    print(f"        {out_png}")
    print(f"Size  : {w}×{h} px  "
          f"({w/300*25.4:.1f}×{h/300*25.4:.1f} mm @ 300 dpi)")


if __name__ == '__main__':
    main()