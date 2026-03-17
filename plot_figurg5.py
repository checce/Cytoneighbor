"""
figure_mechanism.py
━━━━━━━━━━━━━━━━━━━
Composite figure: Zero-shot manifold repair & rare cell recovery.
修改后：保持图片原始比例，适配A4画布拼接
"""

import os
import subprocess
import tempfile
import glob
import importlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0.  SciFont block
# ══════════════════════════════════════════════════════════════════════════════
try:
    from scifont import set_font
    set_font(font_name="Helvetica")
    print("[Success] SciFont loaded. Publication-ready fonts enabled.")
except ImportError:
    print("[Warning] 'scifont' library not found. Falling back to default matplotlib fonts.")
    print("Consider running: pip install scifont")
    plt.rcParams.update({
        'font.family':    'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype':   42,
        'ps.fonttype':    42,
        'svg.fonttype':   'none',
    })

# --- 全局字体与线宽调整 ---
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
# 1.  Colours (Bang Wong palette, Nature Methods 8:441, 2011)
# ══════════════════════════════════════════════════════════════════════════════
WONG = {
    'blue':       '#0072B2',
    'orange':     '#E69F00',
    'vermilion':  '#D55E00',
    'green':      '#009E73',
    'grey':       '#999999',
}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Input PDF paths  ← modify if needed
# ══════════════════════════════════════════════════════════════════════════════
DIR = r'E:\\Project\\Cytoneighbor_Project\\results\\mechanism_analysis'   # ← change to your folder

IMG_MECHANISM = os.path.join(DIR, 'Figure2_Mechanism_Comparison.pdf')
IMG_TREND     = os.path.join(DIR, 'Figure2_Quantitative_Full_Trend.pdf')

# ══════════════════════════════════════════════════════════════════════════════
# 3.  PDF → numpy array  (pdftoppm → pdf2image → pymupdf → PIL fallback)
# ══════════════════════════════════════════════════════════════════════════════
def load_pdf_as_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """Render first page of a PDF to an RGB numpy array. Tries four backends."""
    from PIL import Image as PILImage

    # ── Backend 1: pdftoppm ───────────────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'pg')
            r = subprocess.run(
                ['pdftoppm', '-r', str(dpi), '-png', pdf_path, out],
                capture_output=True, timeout=60)
            if r.returncode == 0:
                pages = sorted(glob.glob(f'{out}*.png'))
                if pages:
                    img = np.array(PILImage.open(pages[0]).convert('RGB'))
                    print(f"[pdftoppm]  {os.path.basename(pdf_path)} → {img.shape}")
                    return img
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        print(f"[pdftoppm] failed: {e}")

    # ── Backend 2: pdf2image ──────────────────────────────────────────────────
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        img = np.array(pages[0].convert('RGB'))
        print(f"[pdf2image] {os.path.basename(pdf_path)} → {img.shape}")
        return img
    except ImportError:
        pass
    except Exception as e:
        print(f"[pdf2image] failed: {e}")

    # ── Backend 3: pymupdf (supports both 'pymupdf' and legacy 'fitz') ────────
    _fitz = None
    for _name in ('pymupdf', 'fitz'):
        try:
            _fitz = importlib.import_module(_name)
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
            print(f"[pymupdf]   {os.path.basename(pdf_path)} → {img.shape}")
            return img
        except Exception as e:
            print(f"[pymupdf] failed: {e}")
    else:
        print("[pymupdf] not installed — skipping")

    # ── Backend 4: PIL ────────────────────────────────────────────────────────
    try:
        img = np.array(PILImage.open(pdf_path).convert('RGB'))
        print(f"[PIL]       {os.path.basename(pdf_path)} → {img.shape}")
        return img
    except Exception as e:
        raise RuntimeError(
            f"Cannot render PDF: {pdf_path}\n"
            f"Install a backend:  pip install pymupdf\n"
            f"Original error: {e}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# 新增：保持图片比例的辅助函数
# ══════════════════════════════════════════════════════════════════════════════
def imshow_keep_aspect(ax, img):
    """在坐标轴中展示图片，保持原始宽高比且填充坐标轴"""
    # 获取图片宽高
    h, w = img.shape[:2]
    # 计算坐标轴的宽高比
    ax_aspect = ax.get_position().width / ax.get_position().height
    # 计算图片的宽高比
    img_aspect = w / h
    
    # 调整图片展示范围，保持比例
    if img_aspect > ax_aspect:
        # 图片更宽，按宽度填充，高度居中
        scale = 1.0
        y_offset = (1 - ax_aspect / img_aspect) / 2
        ax.imshow(img, extent=[0, 1, y_offset, 1 - y_offset])
    else:
        # 图片更高，按高度填充，宽度居中
        scale = 1.0
        x_offset = (1 - img_aspect / ax_aspect) / 2
        ax.imshow(img, extent=[x_offset, 1 - x_offset, 0, 1])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Data (from experiment_results.json & summary_report.txt)
# ══════════════════════════════════════════════════════════════════════════════
K_VALS   = [5, 10, 20, 30, 50, 100]
CASE_A   = [0.0829, 0.0981, 0.1204, 0.1371, 0.1661, 0.2155]
CASE_B   = [0.1330, 0.1459, 0.1614, 0.1734, 0.1950, 0.2374]
REL_IMP  = [60.5, 48.8, 34.1, 26.4, 17.4, 10.2]
ABS_IMP  = [5.01, 4.78, 4.10, 3.63, 2.89, 2.19]

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Panel label helper
# ══════════════════════════════════════════════════════════════════════════════
def panel_label(ax, letter, x=-0.06, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', color='black',
            clip_on=False)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Panel D — Relative improvement bar chart (generated from JSON)
# ══════════════════════════════════════════════════════════════════════════════
def draw_panel_D(ax):
    x    = np.arange(len(K_VALS))
    bars = ax.bar(x, REL_IMP, width=0.6,
                  color=WONG['orange'], alpha=0.85,
                  linewidth=0.5, edgecolor='white', zorder=3)

    # Value labels above bars
    for bar, v in zip(bars, REL_IMP):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.8,
                f'+{v:.1f}%',
                ha='center', va='bottom',
                fontsize=5.0, color='#333333', fontweight='bold')

    # Highlight K=5 (maximum gain)
    bars[0].set_edgecolor(WONG['vermilion'])
    bars[0].set_linewidth(1.0)

    # Trend arrow annotating diminishing returns
    ax.annotate(
        'Diminishing\nbenefit →',
        xy=(x[-1], REL_IMP[-1] + 2),
        xytext=(x[-2], REL_IMP[-2] + 14),
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7),
        fontsize=4.8, color='#555555', ha='center',
    )

    # Reference dashed line at 0
    ax.axhline(0, color='#AAAAAA', linewidth=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in K_VALS], fontsize=5.5)
    ax.set_ylabel('Relative Improvement (%)', fontsize=6.5)
    ax.set_title('Gene-CutMix Gain vs Neighbourhood\nSize (Relative, %)',
                 fontsize=7, fontweight='bold', pad=3)
    ax.set_ylim(0, 75)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis='y', which='minor', length=2, width=0.4)
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    panel_label(ax, 'D')

# # ══════════════════════════════════════════════════════════════════════════════
# # 7.  Main assembly (适配A4尺寸，保持图片比例)
# # ══════════════════════════════════════════════════════════════════════════════
# def main():
#     # ── Load images ───────────────────────────────────────────────────────────
#     img_mech  = load_pdf_as_image(IMG_MECHANISM)
#     img_trend = load_pdf_as_image(IMG_TREND)

#     # Split mechanism image into A (left) and B (right) halves
#     mid = img_mech.shape[1] // 2
#     img_A = img_mech[:, :mid, :]
#     img_B = img_mech[:, mid:, :]

#     # ── Figure canvas: A4尺寸 (210×297mm)，300dpi ─────────────────────────────
#     A4_WIDTH_MM = 210
#     A4_HEIGHT_MM = 297
#     W_IN = A4_WIDTH_MM / 25.4  # 英寸转换
#     H_IN = A4_HEIGHT_MM / 25.4
#     fig  = plt.figure(figsize=(W_IN, H_IN), dpi=300)

#     # Outer grid: 2 rows — 适配A4高度，保持子图比例
#     outer = gridspec.GridSpec(
#         2, 1, figure=fig,
#         height_ratios=[1.2, 1.0],  # 上调机制图行高，适配A4
#         hspace=0.15,  # 增加行间距，避免挤压
#     )

#     # ── Top row: A | B ────────────────────────────────────────────────────────
#     top_gs = gridspec.GridSpecFromSubplotSpec(
#         1, 2, subplot_spec=outer[0],
#         wspace=0.05,  # 微调列间距
#     )

#     # 绘制Panel A，保持比例
#     ax_a = fig.add_subplot(top_gs[0])
#     imshow_keep_aspect(ax_a, img_A)
#     panel_label(ax_a, 'A', x=-0.04)  # 微调标签位置

#     # 绘制Panel B，保持比例
#     ax_b = fig.add_subplot(top_gs[1])
#     imshow_keep_aspect(ax_b, img_B)
#     panel_label(ax_b, 'B', x=-0.04)

#     # ── Bottom row: C (PDF image) | D (generated bar chart) ──────────────────
#     bot_gs = gridspec.GridSpecFromSubplotSpec(
#         1, 2, subplot_spec=outer[1],
#         wspace=0.4,
#         width_ratios=[1.1, 1.0],
#     )

#     # 绘制Panel C，保持比例
#     ax_c = fig.add_subplot(bot_gs[0])
#     imshow_keep_aspect(ax_c, img_trend)
#     panel_label(ax_c, 'C', x=-0.04)

#     # 绘制Panel D
#     ax_d = fig.add_subplot(bot_gs[1])
#     draw_panel_D(ax_d)

#     # ── Save ──────────────────────────────────────────────────────────────────
#     out_dir = DIR
#     os.makedirs(out_dir, exist_ok=True)

#     out_pdf = os.path.join(out_dir, 'figure5_mechanism_A4.pdf')
#     out_png = os.path.join(out_dir, 'figure5_mechanism_A4.png')

#     # 保存时不裁剪，保证A4完整尺寸
#     fig.savefig(out_pdf, bbox_inches='tight', dpi=300, pad_inches=0.2)  # 增加边距
#     fig.savefig(out_png, bbox_inches='tight', dpi=300, pad_inches=0.2)

#     from PIL import Image
#     im   = Image.open(out_png)
#     w, h = im.size
#     print(f"\nSaved (A4, 300dpi):")
#     print(f"  PDF: {out_pdf}")
#     print(f"  PNG: {out_png}")
#     print(f"  Size: {w}×{h} px  ({w/300*25.4:.1f}×{h/300*25.4:.1f} mm)")
# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main assembly (横版 A4：297×210 mm，保持原图比例，不拉伸)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load images ───────────────────────────────────────────────────────────
    img_mech  = load_pdf_as_image(IMG_MECHANISM)
    img_trend = load_pdf_as_image(IMG_TREND)

    # Split mechanism image into A (left) and B (right) halves
    mid = img_mech.shape[1] // 2
    img_A = img_mech[:, :mid, :]
    img_B = img_mech[:, mid:, :]

    # ── 横版 A4 canvas：297 mm × 210 mm @ 300 dpi ────────────────────────────
    W_IN = 297 / 25.4   # 横向宽
    H_IN = 210 / 25.4   # 纵向高
    fig  = plt.figure(figsize=(W_IN, H_IN), dpi=300)

    # 外部分栏：2行，上下比例更舒服
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 1],    # 上下两行等高
        hspace=0.15,
    )

    # ── Top row: A | B ────────────────────────────────────────────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        wspace=0.04,
    )

    ax_a = fig.add_subplot(top_gs[0])
    # 关键：保持原图比例，不拉伸
    ax_a.imshow(img_A, aspect='equal')
    ax_a.set_axis_off()
    panel_label(ax_a, 'A', x=-0.03)

    ax_b = fig.add_subplot(top_gs[1])
    ax_b.imshow(img_B, aspect='equal')
    ax_b.set_axis_off()
    panel_label(ax_b, 'B', x=-0.03)

    # ── Bottom row: C (PDF image) | D (generated bar chart) ──────────────────
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.35,
        width_ratios=[1, 1],
    )

    ax_c = fig.add_subplot(bot_gs[0])
    ax_c.imshow(img_trend, aspect='equal')
    ax_c.set_axis_off()
    panel_label(ax_c, 'C', x=-0.03)

    ax_d = fig.add_subplot(bot_gs[1])
    draw_panel_D(ax_d)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = DIR
    os.makedirs(out_dir, exist_ok=True)

    out_pdf = os.path.join(out_dir, 'figure5_mechanism_A4_landscape.pdf')
    out_png = os.path.join(out_dir, 'figure5_mechanism_A4_landscape.png')

    fig.savefig(out_pdf, bbox_inches='tight', dpi=300, pad_inches=0.1)
    fig.savefig(out_png, bbox_inches='tight', dpi=300, pad_inches=0.1)

    from PIL import Image
    im   = Image.open(out_png)
    w, h = im.size
    print(f"\nSaved A4 横版：")
    print(f"PDF: {out_pdf}")
    print(f"PNG: {out_png}")
    print(f"尺寸: {w}×{h} px ({w/300*25.4:.1f}×{h/300*25.4:.1f} mm)")


if __name__ == '__main__':
    main()