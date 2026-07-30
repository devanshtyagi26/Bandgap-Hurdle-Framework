
# ── Visualisation ────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os

# ── Correct cache dir location (matplotlib >= 3.5) ────────────────────
cache_dir = matplotlib.get_cachedir()
print(f"Font cache location: {cache_dir}")

# ── Clear cache files ─────────────────────────────────────────────────
cleared = 0
for f in os.listdir(cache_dir):
    if f.endswith((".json", ".cache")):
        path = os.path.join(cache_dir, f)
        os.remove(path)
        print(f"🗑️  Removed: {f}")
        cleared += 1

if cleared == 0:
    print("ℹ️  No cache files found — may have already been cleared")

# ── Rebuild font list ─────────────────────────────────────────────────
fm.fontManager.__init__()                      # ✅ works across all versions
print("✅ Font cache rebuilt")

# ── Quick version check for debugging ────────────────────────────────
print(f"\nmatplotlib version : {matplotlib.__version__}")
print(f"Cache directory    : {cache_dir}")

# ── 1. Font Availability Check ────────────────────────────────────────

def check_font(name: str) -> bool:
    """Return True if font family is available in matplotlib."""
    available = {f.name for f in fm.fontManager.ttflist}
    found = name in available
    print(f"  {'✅' if found else '⚠️ '} '{name}' : {'found' if found else 'not found — will use fallback'}")
    return found

print("Font availability:")
check_font("Times New Roman")
check_font("DejaVu Serif")

# ── 2. Global Plot Style ──────────────────────────────────────────────
# Set ONCE here — before any plot cell runs
plt.rcParams.update({
    # Typography
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman'],
    'font.size'        : 11,
    'axes.labelsize'   : 12,
    'axes.titlesize'   : 13,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'legend.fontsize'  : 10,

    # DPI — lower for display, high for saved figures
    'figure.dpi'       : 120,    # screen display (was 300 — too large in Jupyter)
    'savefig.dpi'      : 300,    # publication quality on save
    'savefig.bbox'     : 'tight',
    'savefig.format'   : 'pdf',  # vector format — best for dissertation submission

    # Axes & Grid
    'axes.linewidth'   : 1.2,
    'axes.spines.top'  : False,  # cleaner look — remove top/right spines
    'axes.spines.right': False,
    'axes.grid'        : False,
    # 'grid.linewidth'   : 0.5,
    # 'grid.alpha'       : 0.4,

    # Lines
    'lines.linewidth'  : 2.0,
})

print("\n✅ Global plot style configured")

# ── 3. Colour Palette ─────────────────────────────────────────────────
# Organised as a dict — self-documenting, easy to pass to plot functions
COLORS = {
    # General purpose (print-friendly, colourblind-safe)
    "primary"     : "#2E5090",   # Deep blue
    "secondary"   : "#D2691E",   # Burnt orange
    "tertiary"    : "#008B8B",   # Dark teal

    # Domain-specific — materials classification
    "metal"       : "#C41E3A",   # Crimson red
    "nonmetal"    : "#2E5090",   # Deep blue
    "stable"      : "#228B22",   # Forest green
    "metastable"  : "#FF8C00",   # Dark orange

    # Utility
    "highlight"   : "#FFD700",   # Gold — for annotations
    "neutral"     : "#808080",   # Grey — for reference lines
}

# Convenience list for sequential multi-category plots
PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["tertiary"],
    COLORS["stable"],
    COLORS["metastable"],
]

# Apply to seaborn as well
sns.set_palette(PALETTE)

print(f"🎨 Colour palette registered — {len(COLORS)} named colours")
print(f"   Seaborn palette set to {len(PALETTE)}-colour sequence")

def apply_font(
    ax         : plt.Axes,
    fontfamily : str  = "Times New Roman",
    fontsize   : dict = None,
) -> None:
    """
    Apply a consistent font to ALL text elements of a matplotlib Axes:
    tick labels, title, axis labels, annotations, legend, and colourbar.

    Args:
        ax         : Target Axes object.
        fontfamily : Font family name (default 'Times New Roman').
        fontsize   : Optional dict overriding sizes, e.g.
                     {'title': 13, 'ticks': 10, 'legend': 9, 'annot': 9}

    Example:
        fig, ax = plt.subplots()
        sns.heatmap(..., ax=ax)
        apply_font(ax)
    """
    fs = {
        "title"  : 13,
        "labels" : 12,
        "ticks"  : 10,
        "legend" : 10,
        "annot"  : 9,
        "cbar"   : 11,
    }
    if fontsize:
        fs.update(fontsize)

    # ── Title ─────────────────────────────────────────────────────────
    ax.title.set_fontfamily(fontfamily)
    ax.title.set_fontsize(fs["title"])

    # ── Axis labels ───────────────────────────────────────────────────
    ax.xaxis.label.set_fontfamily(fontfamily)
    ax.xaxis.label.set_fontsize(fs["labels"])
    ax.yaxis.label.set_fontfamily(fontfamily)
    ax.yaxis.label.set_fontsize(fs["labels"])

    # ── Tick labels — x ───────────────────────────────────────────────
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(fontfamily)
        tick.set_fontsize(fs["ticks"])

    # ── Tick labels — y ───────────────────────────────────────────────
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(fontfamily)
        tick.set_fontsize(fs["ticks"])

    # ── Annotations (heatmap cells, ax.text, etc.) ────────────────────
    for text in ax.texts:
        text.set_fontfamily(fontfamily)
        text.set_fontsize(fs["annot"])

    # ── Legend ────────────────────────────────────────────────────────
    legend = ax.get_legend()
    if legend:
        title = legend.get_title()
        if title:
            title.set_fontfamily(fontfamily)
            title.set_fontsize(fs["legend"])
        for text in legend.get_texts():
            text.set_fontfamily(fontfamily)
            text.set_fontsize(fs["legend"])

   # ── Colourbar — replace the cbar block in apply_font() ───────────────
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            #  Handle both orientations
            is_vertical   = cbar.orientation == "vertical"
            current_label = (
                cbar.ax.get_ylabel() if is_vertical
                else cbar.ax.get_xlabel()
            )
            if is_vertical:
                cbar.ax.set_ylabel(
                    current_label,
                    fontfamily = fontfamily,
                    fontsize   = fs["cbar"]
                )
                for tick in cbar.ax.get_yticklabels():
                    tick.set_fontfamily(fontfamily)
                    tick.set_fontsize(fs["ticks"])
            else:
                cbar.ax.set_xlabel(
                    current_label,
                    fontfamily = fontfamily,
                    fontsize   = fs["cbar"]
                )
                for tick in cbar.ax.get_xticklabels():
                    tick.set_fontfamily(fontfamily)
                    tick.set_fontsize(fs["ticks"])


def save_figure(
    fig       : plt.Figure = None,
    filename  : str        = None,
    out_dir   : str        = "Images",
    dpi       : int        = 300,
    formats   : list[str]  = None,
    close     : bool       = True,
    verbose   : bool       = True,
) -> dict[str, str]:
    """
    Save a matplotlib figure to PDF and PNG (or custom formats) at
    publication quality. Uses CONFIG["images_dir"] as the base directory.

    Args:
        fig      : Matplotlib Figure object. Defaults to plt.gcf().
        filename : Output filename WITHOUT extension (e.g. 'eda_bg_dist').
        subdir   : Optional subfolder inside images_dir (e.g. 'eda', 'results').
                   Created automatically if it doesn't exist.
        dpi      : Resolution for raster formats (default 300).
        formats  : List of format strings (default ['pdf', 'png']).
                   Supported: 'pdf', 'png', 'svg', 'eps'.
        close    : Whether to close the figure after saving (default True).
        verbose  : Whether to print save paths (default True).

    Returns:
        Dict mapping format → full save path (e.g. {'pdf': '...', 'png': '...'})

    Raises:
        ValueError : If filename is None.
        OSError    : If the output directory cannot be created.

    Example:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        save_figure(fig, "my_plot", subdir="eda")
    """
    if filename is None:
        raise ValueError("filename must be provided — e.g. 'eda_bg_distribution'")

    # ── Defaults ──────────────────────────────────────────────────────
    if formats is None:
        formats = ["pdf", "png"]

    if fig is None:
        fig = plt.gcf()


    # ── Save each format ──────────────────────────────────────────────
    saved_paths = {}
    for fmt in formats:
        if fmt not in {"pdf", "png", "svg", "eps"}:
            print(f"  ⚠️  Unsupported format '{fmt}' — skipping")
            continue

        path = os.path.join(out_dir, f"{filename}.{fmt}")

        fig.savefig(
            path,
            dpi         = dpi if fmt != "pdf" else None,  # PDF is vector — dpi irrelevant
            bbox_inches = "tight",
            facecolor   = fig.get_facecolor(),
            edgecolor   = "none",
        )
        saved_paths[fmt] = path

        if verbose:
            size_kb = os.path.getsize(path) / 1024
            print(f"  💾 {fmt.upper():<4} → {path}  ({size_kb:.1f} KB)")

    # ── Close figure ──────────────────────────────────────────────────
    if close:
        plt.close(fig)

    return saved_paths

# Load CSVs
cpu_df = pd.read_csv("../Final CPU Execution/Output CSVs/MP Dataset_memory.csv")
gpu_df = pd.read_csv("../Final GPU Execution/Output CSVs/MP Dataset_memory.csv")

# X-axis
x = range(len(cpu_df))
labels = cpu_df["stage"]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    x,
    cpu_df["df_memory_mb"],
    marker="o",
    linewidth=2,
    color=COLORS["primary"],
    label="CPU"
)

ax.plot(
    x,
    gpu_df["df_memory_mb"],
    marker="s",
    linewidth=2,
    color=COLORS["secondary"],
    label="GPU"
)

cpu_avg = cpu_df["df_memory_mb"].mean()
gpu_avg = gpu_df["df_memory_mb"].mean()

ax.axhline(
    cpu_avg,
    color=COLORS["primary"],
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label=f"CPU Avg ({cpu_avg:.1f} MB)"
)

ax.axhline(
    gpu_avg,
    color=COLORS["secondary"],
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label=f"GPU Avg ({gpu_avg:.1f} MB)"
)

ax.set_xlabel("Pipeline Operational Stages")
ax.set_ylabel("DataFrame Memory (MB)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.legend(frameon=False)

apply_font(ax)

plt.tight_layout()

save_figure(fig, "CPU_vs_GPU_Memory")

# Load CSVs
cpu_df = pd.read_csv("../Final CPU Execution/Output CSVs/MP Dataset_latency.csv")
gpu_df = pd.read_csv("../Final GPU Execution/Output CSVs/MP Dataset_latency.csv")

# X-axis
x = range(len(cpu_df))
labels = cpu_df["stage"]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    x,
    cpu_df["elapsed_sec"],
    marker="o",
    linewidth=2,
    color=COLORS["primary"],
    label="CPU"
)

ax.plot(
    x,
    gpu_df["elapsed_sec"],
    marker="s",
    linewidth=2,
    color=COLORS["secondary"],
    label="GPU"
)

cpu_avg = cpu_df["elapsed_sec"].mean()
gpu_avg = gpu_df["elapsed_sec"].mean()

ax.axhline(cpu_avg, color=COLORS["primary"], linestyle="--",
           linewidth=1, alpha=0.7,
           label=f"CPU Avg ({cpu_avg:.1f} s)")

ax.axhline(gpu_avg, color=COLORS["secondary"], linestyle="--",
           linewidth=1, alpha=0.7,
           label=f"GPU Avg ({gpu_avg:.1f} s)")

ax.set_xlabel("Pipeline Operational Stages")
ax.set_ylabel("Latency (s)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.legend(frameon=False)

apply_font(ax)

plt.tight_layout()

save_figure(fig, "CPU_vs_GPU_Latency")

# Load CPU CSV
summary_df = pd.read_csv("../Final CPU Execution/Output CSVs/MP Dataset_data_corpus_structure.csv")

# X-axis
x = range(len(summary_df))
labels = summary_df["stage"]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    x,
    summary_df["columns"],
    marker="o",
    linewidth=2,
    color=COLORS["primary"],
    label="Columns"
)

avg = summary_df["columns"].mean()

ax.axhline(
    avg,
    color=COLORS["neutral"],
    linestyle="--",
    linewidth=1,
    label=f"Average ({avg:.1f})"
)

ax.set_xlabel("Pipeline Operational Stages")
ax.set_ylabel("Number of Columns")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.legend(frameon=False)

apply_font(ax)

plt.tight_layout()

save_figure(fig, "MP_Dataset_Columns")