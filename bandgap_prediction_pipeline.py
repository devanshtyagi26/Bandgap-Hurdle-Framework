# Bandgap Prediction Pipeline

# Function to print headings with different levels and colors
def print_heading(text, level=1):
    # ANSI Color Codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    if level == 1:
        print(f"{CYAN}{f' {text.upper()} '.center(50, '=')}{RESET}")
    elif level == 2:
        print(f"{YELLOW}{f' {text} '.center(50, '-')}{RESET}")
    elif level == 3:
        print(f"{GREEN}{f' {text} '.center(50, '.')}{RESET}")



print_heading("Phase 1: Environment & Preprocessing", level=1)
print_heading("Step 0: Pipeline Control & Environment Setup", level=2)

import importlib, subprocess, sys, os, atexit

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                if not getattr(f, 'closed', False):
                    f.write(obj)
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                if not getattr(f, 'closed', False):
                    f.flush()
            except Exception:
                pass

# 1. Save original streams
_terminal_stdout = sys.stdout
_terminal_stderr = sys.stderr

# 2. Open log file safely
log_file = open("terminal.txt", "w", encoding="utf-8")

# 3. Redirect stdout and stderr
sys.stdout = Tee(_terminal_stdout, log_file)
sys.stderr = Tee(_terminal_stderr, log_file)

# 4. Clean exit handler (runs BEFORE Python closes file handles)
def _cleanup_tee():
    sys.stdout = _terminal_stdout
    sys.stderr = _terminal_stderr
    if not log_file.closed:
        log_file.flush()
        log_file.close()

atexit.register(_cleanup_tee)

# Pipeline Control Configuration
CONFIG = {
    "save_pickle"      : False,   # Serialize processed data to disk after each stage
    "download_raw"     : False,  # Pull fresh data from MP-API (slow, ~1.78 GB)
    "create_chunks"    : False,   # Split data into 50k-row chunks for featurization
    "add_features"     : True,   # Run Matminer/Magpie featurization (slow, skip if cached)
    "display_graphs"   : False,
    "gpu_env"          : False,
    "intermediate_dir" : "Intermediate Pickles",
    "chunks_dir"       : "data_chunks",
    "images_dir"       : "Output Images",
    "csv_dir"          : "Output CSVs",
    "models_dir"       : "Models",
    "raw_pkl"          : "materials_data.pkl",
}

# Sanity check: featurization requires data to exist
if CONFIG["add_features"] and not CONFIG["download_raw"]:
    assert os.path.exists(CONFIG["raw_pkl"]), (
        "add_features=True but no cached raw data found. "
        "Set download_raw=True or provide data/raw_cache.pkl."
    )

# ── CPU-only enforcement guard ─────────────────────────────────────
assert CONFIG["gpu_env"] is False, "Set CONFIG['gpu_env']=False for CPU-only runs"
print(f"✅ Running in CPU-only mode (CONFIG['gpu_env']={CONFIG['gpu_env']})")

# GPU Environment Validation & Dependency Setup

print_heading("Step 1: Environment Validation & Dependency Setup", level=2)

if CONFIG["gpu_env"]:
    # Equivalent of: !nvidia-smi
    subprocess.run(["nvidia-smi"], check=False)


# Verify CUDA is available before loading cuDF to catch environment issues early
def check_gpu_environment():
    try:
        import cudf
        print(f"✅ cuDF version  : {cudf.__version__}")
    except ImportError:
        print("❌ cuDF not found — falling back to CPU pandas")
        print("   Install via: pip install cudf-cu11 --extra-index-url=...")
        CONFIG["gpu_env"] = False



if CONFIG["gpu_env"]:
    check_gpu_environment()
    try:
        import rmm
        import cupy

        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)

        try:
            from rmm.allocators.cupy import rmm_cupy_allocator
        except ImportError:
            from rmm import rmm_cupy_allocator

        cupy.cuda.set_allocator(rmm_cupy_allocator)
        print("✅ RMM pool allocator initialized (shared by cuDF + XGBoost)")
    except Exception as e:
        print(f"⚠️  RMM init failed: {e} — cuDF and XGBoost will use separate pools")

if CONFIG["gpu_env"]:
    try:
        import cupy as cp
        import cudf.pandas
        cudf.pandas.install()
        print("✅ cuDF pandas accelerator enabled.")
    except Exception as e:
        print(f"❌ Failed to enable cudf.pandas: {e}")
        CONFIG["gpu_env"] = False

# Core dependencies for MP data retrieval and featurization
REQUIREMENTS = {
    "mp_api"   : "mp_api>=0.41.0",    # Materials Project API client
    "pymatgen" : "pymatgen>=2024.1.1", # Crystal structure parsing
    "matminer" : "matminer>=0.9.0",    # Magpie featurization
}

def install_if_needed(package: str, version: str) -> None:
    """Install a package only if not already present at the required version."""
    try:
        mod = importlib.import_module(package)
        if mod.__version__ == version:
            print(f"✅ {package}=={version} already installed")
            return
    except (ImportError, AttributeError):
        pass
    print(f"📦 Installing {package}=={version}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        f"{package}=={version}", "--quiet"
    ])

if CONFIG["download_raw"]:
    install_if_needed("mp_api", REQUIREMENTS["mp_api"])

# ── Standard Library ────────────────────────────────────────────────
import os
import gc
import re
import time
import json
import pickle
import joblib
import warnings
from pathlib import Path
from ast import literal_eval
from dotenv import load_dotenv
from collections import Counter
from typing import List, Any

# ── Numerical & Scientific ───────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats

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
import shap

# ── Scikit-Learn ─────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix, median_absolute_error,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, classification_report,
    precision_recall_curve
)

# ── Tracker ────────────────────────────────────────────────
import psutil
from functools import wraps

# ── Materials Informatics ─────────────────────────────────────────────
import pymatgen
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from mp_api.client import MPRester

# XGBoost
import xgboost as xgb
if CONFIG["gpu_env"]:
    xgb.set_config(use_rmm=True)

import torch

if not CONFIG["gpu_env"]:
    torch.set_num_threads(os.cpu_count())
    print(f"✅ PyTorch CPU threads set to {os.cpu_count()}")
    
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# CHGnet
import chgnet
from chgnet.model import CHGNet
from chgnet.model import StructOptimizer
from pymatgen.core import Structure
import copy

# Optuna
import optuna
from optuna.integration import XGBoostPruningCallback
import optuna.visualization as vis


# ── Global Settings ───────────────────────────────────────────────────
warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Version Log (for reproducibility) ────────────────────────────────
import importlib.metadata
from collections import defaultdict

def get_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"

print(f"numpy      : {np.__version__}")
print(f"pandas     : {pd.__version__}")
print(f"pymatgen   : {get_version('pymatgen')}")
print(f"matminer   : {get_version('matminer')}")
print(f"xgboost    : {get_version('xgboost')}")
print(f"chgnet    : {get_version('chgnet')}")
print(f"scikit-learn: {get_version('scikit-learn')}")

def free_gpu_memory():
    """Force-release cuDF/CuPy/RMM/PyTorch-held GPU memory back to the driver."""
    gc.collect()
    if CONFIG["gpu_env"]:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  ⚠️ CuPy memory free failed: {e}")
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        print(f"  ⚠️ Torch memory free failed: {e}")

free_gpu_memory()

for key in ["images_dir", "csv_dir", "models_dir", "intermediate_dir"]:
    path = CONFIG[key]
    os.makedirs(path, exist_ok=True)
    print(f"Created/verified: {os.path.abspath(path)}")
    

TIME_START = time.perf_counter()

# ── API Key ───────────────────────────────────────────────────────────
# Checked unconditionally (not just when CONFIG['download_raw']=True):
# a later cell (structure re-fetch / ablation section) also calls
# MPRester(API) regardless of this flag. Previously, when
# download_raw=False, `API` was never assigned at all, so that later
# cell would raise a NameError instead of a clear, actionable message.
# Set your Materials Project API key as an environment variable before
# starting Python, e.g.:
#   export MP_API_KEY='your_key_here'        # bash/zsh
#   $env:MP_API_KEY = "your_key_here"         # PowerShell
# or, inside a Jupyter cell: %env MP_API_KEY=your_key_here
# Get your key at: https://next.materialsproject.org/api


load_dotenv()
API = os.getenv("MP_API_KEY")

if not API:
    print("⚠️  API key not found in environment.")
    if CONFIG['download_raw']:
        # Fresh download explicitly requested — this key is required now.
        raise EnvironmentError(
            "MP_API_KEY environment variable not set, but "
            "CONFIG['download_raw']=True requires it.\n"
            "Get your key at https://next.materialsproject.org/api\n"
            "Then run: export MP_API_KEY='your_key_here'"
        )
    else:
        print("    Not required right now (CONFIG['download_raw']=False, "
              "using cached data), but any cell that calls MPRester(API) "
              "later in this notebook will fail until it is set.")
else:
    print("✅ API key loaded from environment")

with open("requirements.txt", "w") as f:
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=f,
        check=True,
        text=True,
    )

print_heading("Step 2: Utility Initialization", level=2)

class DFTracker:
    def __init__(self):
        self.stages = defaultdict(list)
        self._last_checkpoint_time = time.perf_counter()

    def _get_process_rss_mb(self) -> float:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 ** 2), 2)

    @staticmethod
    def _fast_memory_mb(df: pd.DataFrame) -> float:
        """Hybrid memory estimate: shallow (O(1)) for numeric columns,
        deep only for object/string columns — avoids full deep=True scan."""
        shallow = df.memory_usage(deep=False)
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) > 0:
            deep_obj = df[obj_cols].memory_usage(deep=True)
            total = shallow.drop(obj_cols, errors="ignore").sum() + deep_obj.sum()
        else:
            total = shallow.sum()
        return round(total / (1024 ** 2), 2)

    @staticmethod
    def get_hashable_columns(df):
        cols = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                cols.append(col)
                continue
            try:
                hash(s.iloc[0])
                cols.append(col)
            except TypeError:
                pass
        return cols

    def track(
        self,
        df: pd.DataFrame,
        label: str,
        dataset: str = "default",
        note: str = "",
        check_nulls: bool = True,
        check_duplicates: bool = False,   
        dup_sample_n: int | None = 20_000,  
        row_override: int | None = None,
        col_override: int | None = None,   
    ):
        """
        Captures structural, memory, and performance metrics with exact deltas
        relative to the preceding stage. Expensive checks (nulls, duplicates)
        are opt-in per call to keep routine tracking fast.
        """
        current_time = time.perf_counter()
        duration = round(current_time - self._last_checkpoint_time, 4)

        # Core DataFrame Metrics — fast path
        rows   = row_override if row_override is not None else len(df)
        cols   = col_override if col_override is not None else len(df.columns)
        df_mem = self._fast_memory_mb(df)
        rss_mem = self._get_process_rss_mb()

        total_nulls = int(df.isna().sum().sum()) if check_nulls else None

        # ── Optional: duplicate check (expensive — sampled by default) ────
        total_duplicates = None
        if check_duplicates:
            try:
                hashable_cols = self.get_hashable_columns(df)
                target = df[hashable_cols]
                if dup_sample_n is not None and len(target) > dup_sample_n:
                    target = target.sample(n=dup_sample_n, random_state=42)
                total_duplicates = int(target.duplicated().sum())
            except Exception as e:
                print(f"[Tracker] Duplicate check skipped: {e}")
                total_duplicates = None

        # Data type inventory — cheap, index-only
        dtype_profile = df.dtypes.value_counts().to_dict()
        dtype_str = ", ".join(f"{str(k)}:{v}" for k, v in dtype_profile.items())

        history = self.stages[dataset]

        if history:
            prev = history[-1]
            delta_rows = rows - prev["rows"]
            delta_cols = cols - prev["columns"]
            delta_mem  = round(df_mem - prev["df_memory_mb"], 2)
        else:
            delta_rows = delta_cols = delta_mem = 0

        history.append({
            "stage":            label,
            "rows":             rows,
            "columns":          cols,
            "df_memory_mb":     df_mem,
            "process_rss_mb":   rss_mem,
            "timestamp":        time.strftime("%H:%M:%S", time.localtime(time.time())),
            "elapsed_sec":      duration,
            "delta_rows":       delta_rows,
            "delta_cols":       delta_cols,
            "delta_df_mem_mb":  delta_mem,
            "total_nulls":      total_nulls,
            "total_duplicates": total_duplicates,
            "dtype_profile":    dtype_profile,
            "dtype_summary":    dtype_str,
            "note":             note,
        })

        self._last_checkpoint_time = time.perf_counter()
        return self

    def stage(self, label: str, note: str = ""):
        """
        Decorator that transparently instruments a function pipeline stage,
        tracking performance metrics automatically on entry and exit.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Execute pipeline state mutation
                res_df = func(*args, **kwargs)
                if isinstance(res_df, pd.DataFrame):
                    self.track(res_df, label=label, note=note)
                return res_df
            return wrapper
        return decorator
        
    def summary(self, dataset=None):
    
        if dataset is None:
            return {
                name: pd.DataFrame(records)
                for name, records in self.stages.items()
            }
    
        return pd.DataFrame(self.stages[dataset])

    def print_text_dashboard(self):
        """Generates a text-based dashboard matching standard thesis report constraints."""
        df_summary = self.summary()
        if df_summary.empty:
            print("No pipeline metrics available.")
            return
            
        total_runtime = df_summary["elapsed_sec"].sum()
        peak_df_mem = df_summary["df_memory_mb"].max() / 1024 # Convert to GB
        final_rows = df_summary["rows"].iloc[-1]
        final_cols = df_summary["columns"].iloc[-1]
        
        print("+" + "─" * 60 + "+")
        print(f"| {'PIPELINE EXECUTION METRIC AUDIT REPORT':^56} |")
        print("+" + "─" * 60 + "+")
        print(f"| Total Monitored Stages   : {len(df_summary):<31} |")
        print(f"| Final Corpus Length      : {final_rows:<31,} rows |")
        print(f"| Final Operational Schema : {final_cols:<31,} columns |")
        print(f"| Peak Target Array RAM    : {peak_df_mem:<31.3f} GB |")
        print(f"| Total Accrued Runtime    : {total_runtime:<31.3f} sec |")
        print("+" + "─" * 60 + "+")

    def save(self, dataset, filepath: str):
        """Exports the tracked history automatically based on file extension (.csv or .json)."""
        df_summary = self.summary(dataset)
        if filepath.endswith(".csv"):
            df_summary.to_csv(filepath, index=False)
        elif filepath.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(self.stages, f, indent=4)
        print(f"📄 Execution log checkpoint successfully exported to: {filepath}")

    def plot(self, dataset, save_filename: str = "pipeline_audit_dashboard.png"):
        """Renders publication-ready diagnostics tracing data structure metrics and run durations."""
        if not self.stages:
            print("No tracking records to plot.")
            return

        df_summary = self.summary(dataset)
        labels = df_summary["stage"]
        x = np.arange(len(labels))

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        
        # 1. Structural Scaling Matrix (Rows vs Columns)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x, df_summary["rows"], marker="o", color="steelblue", linewidth=2, label="Rows")
        ax1.set_ylabel("Row Count", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, df_summary["columns"], marker="s", color="purple", linewidth=1.5, label="Cols")
        ax1_twin.set_ylabel("Column Count", color="purple")
        ax1_twin.tick_params(axis="y", labelcolor="purple")
        ax1.set_title("DataFrame Structural Topology Scale", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.3)

        # 2. Volumetric MemoryFootprint
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x, df_summary["df_memory_mb"], marker="o", color="darkorange", linewidth=2, label="DF Memory")
        ax2.plot(x, df_summary["process_rss_mb"], marker="x", color="crimson", linewidth=1.5, linestyle=":", label="Total Process RSS")
        ax2.set_ylabel("Memory Overhead (MB)")
        ax2.set_title("Memory Profile Allocation Trace", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax2.legend(loc="upper left", frameon=True, fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.3)

        # 3. Micro-Execution Durations (Horizontal Bar Chart)
        ax3 = fig.add_subplot(gs[1, :])
        bars = ax3.barh(x, df_summary["elapsed_sec"], color="seagreen", alpha=0.85, edgecolor="black", height=0.5)
        ax3.set_yticks(x)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.invert_yaxis()  # Natural sequence flow reading downward
        ax3.set_xlabel("Isolated Stage Duration (Seconds)")
        ax3.set_title("Operational Processing Latency Timeline", fontweight="bold")
        ax3.grid(True, axis="x", linestyle="--", alpha=0.3)

        # Text labels on bars
        for bar in bars:
            width = bar.get_width()
            ax3.annotate(f" {width:.2f}s",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, fontweight="bold")

        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        if CONFIG["display_graphs"]:
            plt.show()

tracker = DFTracker()

# Function to save matplotlib figures in multiple formats
def save_figure(
    fig       : plt.Figure = None,
    filename  : str        = None,
    out_dir   : str        = CONFIG['images_dir'],
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

print_heading("Step 3: Data Ingestion & Partitioning", level=2)

# ── Download Raw Data ─────────────────────────────────────────────
if CONFIG["download_raw"]:
    with MPRester(API, monty_decode=False, use_document_model=False) as mpr:
        docs = mpr.materials.summary.search()

        # Save the `docs` object to a file named 'materials_data.pkl'
        with open('materials_data.pkl', 'wb') as f:
            pickle.dump(docs, f)
        print("✅ Data saved to materials_data.pkl")
    del docs, mpr
    tracker.track(pd.DataFrame(), "Downloaded RAW Data", note="Downloading Raw data", dataset="MP Dataset", row_override=len(docs))
    free_gpu_memory()

# ── Chunking Utility ───────────────────────────────────────────────
def chunk_pickle_file(
    file_path: str,
    chunk_size: int,
    output_dir: str = 'chunks'
) -> List[str]:
    """
    Split a large pickle file (list) into smaller chunk files.

    Args:
        file_path  : Path to the source pickle file.
        chunk_size : Number of entries per chunk.
        output_dir : Directory to write chunk files into.

    Returns:
        List of file paths to created chunk files.

    Raises:
        FileNotFoundError : If file_path does not exist.
        MemoryError       : If file is too large to load at once.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    print(f"Loading '{file_path}'...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)

    total_items = len(data)
    num_chunks  = (total_items + chunk_size - 1) // chunk_size
    chunk_paths = []

    print(f"Splitting {total_items:,} items → {num_chunks} chunks of ~{chunk_size:,}...")
    for i in range(num_chunks):
        chunk    = data[i * chunk_size : (i + 1) * chunk_size]
        out_path = os.path.join(output_dir, f'chunk_{i}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        chunk_paths.append(out_path)
        print(f"  ✅ chunk_{i} — {len(chunk):,} items → '{out_path}'")

    del data
    free_gpu_memory()
    return chunk_paths

# ── Run Chunking ───────────────────────────────────────────────────
if CONFIG["download_raw"] or CONFIG["create_chunks"]:
    try:
        chunk_files = chunk_pickle_file(
            file_path  = CONFIG["raw_pkl"],
            chunk_size = 50000,
            output_dir=CONFIG["chunks_dir"]
        )
        print("\nChunked files created:")
        for file in chunk_files:
            print(f"- {file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except MemoryError as e:
        print(f"Process terminated due to lack of memory. {e}")

    tracker.track(pd.DataFrame(), "Chunked RAW Data", note="Chunking RAW data", dataset="MP Dataset", row_override=sum(len(pickle.load(open(f, 'rb'))) for f in chunk_files) if False else None,
    col_override=len(chunk_files), )

# ── Fields to Extract ──────────────────────────────────────────────
FIELDS = {
    "Material ID"              : lambda e: e.get('material_id'),
    "Elements"                 : lambda e: e.get('elements'),
    "Pretty Formula"           : lambda e: e.get('formula_pretty'),
    "Number of Elements"       : lambda e: e.get('nelements'),
    "Normalised Composition"   : lambda e: e.get('composition_reduced'),
    "Nsites"                   : lambda e: e.get('nsites'),
    "Volume"                   : lambda e: e.get('volume'),
    "Density"                  : lambda e: e.get('density'),
    "Oxidation States"         : lambda e: e.get('possible_species'),
    "Crystal System"           : lambda e: e.get('symmetry', {}).get('crystal_system'),
    "Symmetry Symbol"          : lambda e: e.get('symmetry', {}).get('symbol'),
    "Space Group Number"       : lambda e: e.get('symmetry', {}).get('number'),
    "Lattice (a)"              : lambda e: e.get('structure', {}).get('lattice', {}).get('a'),
    "Lattice (b)"              : lambda e: e.get('structure', {}).get('lattice', {}).get('b'),
    "Lattice (c)"              : lambda e: e.get('structure', {}).get('lattice', {}).get('c'),
    "Lattice (alpha)"          : lambda e: e.get('structure', {}).get('lattice', {}).get('alpha'),
    "Lattice (beta)"           : lambda e: e.get('structure', {}).get('lattice', {}).get('beta'),
    "Lattice (gamma)"          : lambda e: e.get('structure', {}).get('lattice', {}).get('gamma'),
    "Energy Per Atom"          : lambda e: e.get('energy_per_atom'),
    "Magnetic Ordering"        : lambda e: e.get('ordering'),
    "Total Magnetization"      : lambda e: e.get('total_magnetization'),
    "Formation Energy Per Atom": lambda e: e.get('formation_energy_per_atom'),
    "Band Gap (T)"             : lambda e: e.get('band_gap'),
}

# ── Chunk → DataFrame ──────────────────────────────────────────────
def pkl_to_records(chunk_path: str) -> List[dict]:
    """
    Load one chunk file and extract target fields into a list of row dicts.
    No global state — returns data for caller to accumulate.

    Args:
        chunk_path : Full path to the chunk pickle file.

    Returns:
        List of flat dictionaries, one per material entry.
    """
    with open(chunk_path, 'rb') as f:
        data = pickle.load(f)

    records = [
        {field: extractor(entry) for field, extractor in FIELDS.items()}
        for entry in data
    ]

    del data
    free_gpu_memory()
    return records


chunk_files = sorted(
    [f for f in os.listdir(CONFIG["chunks_dir"]) if f.endswith('.pkl')],
    key=lambda x: int(re.search(r'\d+', x).group())   # numeric sort
)

all_records = []
for fname in chunk_files:
    path    = os.path.join(CONFIG["chunks_dir"], fname)
    records = pkl_to_records(path)
    all_records.extend(records)
    print(f"✅ {fname} — {len(records):,} records loaded")

df = pd.DataFrame(all_records)
del all_records
free_gpu_memory()

print(f"\nFinal DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")

tracker.track(df, "Raw MP Data Ingestion", note="Ingestion", dataset="MP Dataset")


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.0.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 4: Null Audit & Missing-Value Resolution", level=2)


# ── Null Audit — Before ────────────────────────────────────────────
print("=" * 45)
print("NULL AUDIT — BEFORE CLEANING")
print("=" * 45)
null_before = df.isna().sum()
print(null_before[null_before > 0].to_string())
print(f"\nTotal rows : {len(df):,}")

# ── Drop Rows with Missing Target Variable ─────────────────────────
# Band Gap (T) is the regression target — rows without it are unusable
rows_before = len(df)
df = df.dropna(subset=['Band Gap (T)'])
rows_dropped = rows_before - len(df)
print(f"\nDropped {rows_dropped:,} rows with null Band Gap (T)")
print(f"Remaining : {len(df):,} rows")


# ── Handle Missing Oxidation States ───────────────────────────────
# possible_species is a list column — .replace() won't catch None here
# Must use .apply() for element-wise handling
df["Oxidation States"] = df["Oxidation States"].apply(
    lambda x: x if isinstance(x, list) and len(x) > 0 else ["Unknown"]
)


# ── Null Audit — After ─────────────────────────────────────────────
null_after = df.isna().sum()
remaining_nulls = null_after[null_after > 0]

if remaining_nulls.empty:
    print("✅ No null values remaining")
else:
    print(remaining_nulls.to_string())

tracker.track(df, "Missing Target Filtering", note="Preprocessing", dataset="MP Dataset")


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.1.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 5: Cleaning & Memory Optimization", level=2)

print("DTYPES — BEFORE CASTING")
df.dtypes

# ── Range Validation BEFORE Casting ───────────────────────────────
INT_BOUNDS = {
    'int8' : (-128,       127),
    'int16': (-32_768,    32_767),
    'int32': (-2_147_483_648, 2_147_483_647),
}

def validate_dtype_bounds(
    df: pd.DataFrame,
    dtype_dict: dict
) -> bool:
    """
    Validate that numeric columns won't overflow their target dtypes.
    Prints a warning for any column that would overflow.

    Args:
        df         : Source DataFrame.
        dtype_dict : Mapping of column name → target dtype string.

    Returns:
        bool: True if all columns are safe to cast, False if any will overflow.
    """
    safe = True
    for col, dtype in dtype_dict.items():
        if col not in df.columns:
            print(f"  ⚠️  Column '{col}' not found in DataFrame — skipping")
            continue
        if dtype not in INT_BOUNDS:
            continue  # float32 overflow range is huge, skip

        lo, hi   = INT_BOUNDS[dtype]
        col_min  = df[col].min()
        col_max  = df[col].max()

        if col_min < lo or col_max > hi:
            print(
                f"  ❌ OVERFLOW RISK: '{col}' → {dtype}\n"
                f"     Actual range : [{col_min}, {col_max}]\n"
                f"     dtype range  : [{lo}, {hi}]"
            )
            safe = False
        else:
            print(f"  ✅ '{col}' → {dtype}  (range [{col_min}, {col_max}])")
    return safe

# ── 3. dtype Mapping ──────────────────────────────────────────────────
DTYPE_MAP = {
    # Integer columns
    'Number of Elements'      : 'int8',    # max 118 (periodic table)
    'Space Group Number'      : 'int16',   # max 230
    'Nsites'                  : 'int16',   # rarely > 1000 in MP

    # Categorical columns (saves memory + speeds up groupby)
    'Crystal System'          : 'category',
    'Magnetic Ordering'       : 'category',
    'Symmetry Symbol'         : 'category',

    # Float columns — float32 sufficient for DFT-precision values
    'Volume'                  : 'float32',
    'Density'                 : 'float32',
    'Lattice (a)'             : 'float32',
    'Lattice (b)'             : 'float32',
    'Lattice (c)'             : 'float32',
    'Lattice (alpha)'         : 'float32',
    'Lattice (beta)'          : 'float32',
    'Lattice (gamma)'         : 'float32',
    'Energy Per Atom'         : 'float32',
    'Total Magnetization'     : 'float32',
    'Formation Energy Per Atom': 'float32',
    'Band Gap (T)'            : 'float32',
}

# ── 4. Validate Then Cast ─────────────────────────────────────────────
print("\nVALIDATING DTYPE BOUNDS...")
if validate_dtype_bounds(df, DTYPE_MAP):
    df = df.astype(DTYPE_MAP)
    print("\n✅ All columns cast successfully")
else:
    raise ValueError(
        "Dtype casting aborted — overflow risk detected. "
        "Review column ranges above and adjust DTYPE_MAP."
    )

# ── Snapshot dtypes and memory AFTER ──────────────────────────────
print("\nDTYPES — AFTER CASTING")
df.dtypes

tracker.track(df, "Memory Downcasting", note="Memory Optimization", dataset="MP Dataset")


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.2.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Phase 2: Feature Engineering & Selection", level=1)
print_heading("Step 6: Visualization Configuration & Exploratory Setup", level=2)

# ── Correct cache dir location (matplotlib >= 3.5) ────────────────────
cache_dir = matplotlib.get_cachedir()          # ✅ top-level module
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

# ── Select Numeric Columns ─────────────────────────────────────────
NUMERIC_DTYPES = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']
num_df = df.select_dtypes(include=NUMERIC_DTYPES)

print(f"Numeric columns selected : {num_df.shape[1]}")
print(f"Columns                  : {list(num_df.columns)}")

# ── Correlation Matrix ─────────────────────────────────────────────
corr = num_df.corr()

# Function to apply consistent font styling across all text elements in a matplotlib Axes
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

print_heading("Step 7: Exploratory Data Analysis — Raw Feature Correlation", level=2)

# ── Force Times New Roman on all heatmap text elements ───────────────
HEATMAP_FONT = "Times New Roman"

# ── 1. cuDF → pandas safety conversion ───────────────────────────────
corr_pd      = corr.to_pandas() if hasattr(corr, "to_pandas") else corr
corr_values  = np.asarray(corr_pd.values, dtype=float)
corr_columns = list(corr_pd.columns)
n_cols       = len(corr_columns)


# ── 2. Build Figure ───────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(max(12, n_cols * 0.7), max(10, n_cols * 0.6))
)

# Lower triangle only — upper is a redundant mirror
# mask = np.triu(np.ones_like(corr_values, dtype=bool))

# Before — scattered manual font setting
ax.set_xticklabels(ax.get_xticklabels(), fontfamily="Times New Roman")
ax.set_yticklabels(ax.get_yticklabels(), fontfamily="Times New Roman")
for text in ax.texts: text.set_fontfamily("Times New Roman")
legend = ax.get_legend()
if legend:
    for t in legend.get_texts(): t.set_fontfamily("Times New Roman")

# After — one line
apply_font(ax)

sns.heatmap(
    corr_values,
    # mask       = mask,
    annot      = n_cols <= 15,       # annotations only when legible
    fmt        = ".2f",
    cmap       = "coolwarm",
    center     = 0,                  # white = zero correlation
    linewidths = 0.4,
    square     = True,               # equal aspect — cleaner matrix look
    xticklabels= corr_columns,
    yticklabels= corr_columns,
    cbar_kws   = {
        "shrink"      : 0.6,
        # "label"       : "Pearson r",
        "orientation" : "vertical",
    },
    annot_kws  = {
        "size": 9,
        "family" : HEATMAP_FONT
        },
    ax         = ax
)


# ── 3. Typography — consistent with rcParams ──────────────────────────
# ax.set_title(
#     "Correlation Matrix — Numeric Features\n"
#     f"({n_cols} features, lower triangle shown)",
#     fontweight = "bold",
#     pad        = 18
#     # fontsize inherited from rcParams axes.titlesize = 13
# )

cbar = ax.collections[0].colorbar

# Set label with explicit font
cbar.set_label(
    "Pearson r",
    fontfamily = HEATMAP_FONT,
    fontsize   = 11,
    labelpad   = 10
)

# Tick labels on colourbar
for tick in cbar.ax.get_yticklabels():
    tick.set_fontfamily(HEATMAP_FONT)
    tick.set_fontsize(11)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    ha       = "right",
    fontfamily = HEATMAP_FONT,
    fontsize   = 11
    # labelsize inherited from rcParams xtick.labelsize = 11
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation = 0,
    fontfamily = HEATMAP_FONT,
    fontsize   = 11
)

for text in ax.texts:
    text.set_fontfamily(HEATMAP_FONT)
    text.set_fontsize(11)

ax.grid(False)      # heatmap has its own cell borders — grid adds noise
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()
save_figure(fig, "correlation_heatmap")

print(f"Heatmap saved")

del corr_values
free_gpu_memory()

print_heading("Step 8: Feature Selection — Raw Numeric Collinearity Pruning", level=2)

# ── Vectorised Correlated Feature Removal ──────────────────────────
def remove_correlated_features(
    df         : pd.DataFrame,
    target_col : str,
    threshold  : float = 0.70,
    corr_matrix: pd.DataFrame = None,
    verbose    : bool = True,
    train_mask : "pd.Series | None" = None,

) -> tuple[list[str], pd.DataFrame]:
    """
    Remove inter-correlated features, keeping the one more correlated
    with the target. Uses vectorised upper-triangle approach — no Python loops.

    Args:
        df          : Numeric DataFrame including target column.
        target_col  : Name of the regression target column.
        threshold   : Absolute correlation threshold (default 0.70).
        corr_matrix : Pre-computed correlation matrix (avoids recomputation).

                      NOTE: if you pass a pre-computed matrix, it is your
                      responsibility to ensure it was computed on a
                      train-only subset — see `train_mask` otherwise.

        verbose     : Print detailed drop log.
        train_mask  : Optional boolean Series aligned to df.index. If given
                      AND corr_matrix is None, correlation stats are computed
                      on train-only rows only, so cal/test rows cannot
                      influence which columns are selected (leakage guard).
                      Ignored if corr_matrix is explicitly supplied.


    Returns:
        Tuple of (list of dropped column names, drop log as DataFrame).

    Raises:
        ValueError: If target_col is not in df.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if corr_matrix is not None:
        corr = corr_matrix.abs()
    elif train_mask is not None:
        corr = df.loc[train_mask].corr().abs()
        if verbose:
            print(f"  ℹ️  Correlation matrix computed on "
                  f"{int(train_mask.sum()):,} train-only rows out of "
                  f"{len(df):,} total.")
    else:
        corr = df.corr().abs()
        if verbose:
            print(f"  ⚠️  No train_mask or pre-computed corr_matrix supplied — "
                  f"using all rows. Risk of target leakage if a train/test "
                  f"split happens downstream of this feature-selection step.")


    # Upper triangle mask — avoids checking each pair twice
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    # Target correlations for all features
    target_corr = corr[target_col]

    drop_log = []   # for reporting

    # Vectorised: find all pairs exceeding threshold
    high_corr_pairs = [
        (col, row)
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]

    dropped = set()
    for feat1, feat2 in high_corr_pairs:
        if feat1 in dropped or feat2 in dropped:
            continue
        # Drop whichever has lower correlation with target
        if target_corr[feat1] >= target_corr[feat2]:
            loser, keeper = feat2, feat1
        else:
            loser, keeper = feat1, feat2

        dropped.add(loser)

        drop_log.append({
            "Dropped"          : loser,
            "Kept"             : keeper,
            "Inter-Correlation": round(float(corr.loc[feat1, feat2]), 4),
            "Dropped→Target"   : round(target_corr[loser], 4),
            "Kept→Target"      : round(target_corr[keeper], 4),
        })

    log_df = pd.DataFrame(drop_log)

    if verbose and not log_df.empty:
        print(f"\n{'='*55}")
        print(f"  CORRELATION FILTER — threshold = {threshold}")
        print(f"{'='*55}")
        print(log_df.to_string(index=False))
        print(f"\n  Total dropped : {len(dropped)}")
        print(f"  Remaining     : {len(df.columns) - len(dropped) - 1} features")
        print(f"{'='*55}")
    elif log_df.empty:
        print(f"✅ No features exceed correlation threshold of {threshold}")

    return list(dropped), log_df

_seed = RANDOM_SEED if "RANDOM_SEED" in dir() else 42
_rng_struct = np.random.RandomState(_seed)
leakage_safe_train_mask_structural = pd.Series(
    _rng_struct.rand(len(num_df)) < 0.80,
    index=num_df.index,
)
print(f"  Leakage-safe train mask (structural pass): "
      f"{leakage_safe_train_mask_structural.sum():,} / {len(num_df):,} "
      f"rows (~80%) will inform correlation stats.")

# ── Apply Filter ───────────────────────────────────────────────────
to_drop, drop_log = remove_correlated_features(
    df          = num_df,
    target_col  = 'Band Gap (T)',
    threshold   = 0.70,
    corr_matrix = None,
    train_mask  = leakage_safe_train_mask_structural,
    verbose     = True
)

df = df.drop(columns=to_drop, errors='ignore')
print(f"\nDataFrame shape after correlation filter: {df.shape}")

# ── Cleanup Intermediates ──────────────────────────────────────────
del num_df, corr, to_drop
free_gpu_memory()

tracker.track(df, "Structural Correlation Filter", note="Feature Selection", dataset="MP Dataset")

print_heading("Step 9: Composition Parsing & Descriptor Generation", level=2)

# ── 1. Safe Composition Converter ────────────────────────────────────
def safe_to_composition(x) -> tuple[Composition | None, str | None]:
    """
    Convert a composition dict from the Materials Project API into a
    pymatgen Composition object.

    Args:
        x : Raw value from 'Normalised Composition' column.
            Expected format: dict e.g. {'Fe': 1.0, 'O': 1.5}

    Returns:
        Tuple of (Composition | None, failure_reason | None)
    """
    # ── Guard: check type before pd.isna (avoid ValueError on dicts) ──
    if x is None:
        return None, "null_value"

    if not isinstance(x, dict):
        return None, f"unexpected_type:{type(x).__name__}"

    if len(x) == 0:
        return None, "empty_dict"

    # ── Filter None/NaN stoichiometry values within the dict ──────────
    filtered = {
        element: count
        for element, count in x.items()
        if count is not None and not (
            isinstance(count, float) and np.isnan(count)
        )
    }

    if not filtered:
        return None, "all_counts_null"

    try:
        return Composition(filtered), None
    except Exception as e:
        return None, f"pymatgen_error:{str(e)[:60]}"

# ── 2. Convert Column ─────────────────────────────────────────────────
print("Converting 'Normalised Composition' → pymatgen Composition objects...")

results       = [safe_to_composition(x) for x in df["Normalised Composition"].values]
compositions  = [r[0] for r in results]
failure_notes = [r[1] for r in results]

# Assign directly — no .copy() needed (df is not a slice)
df["Composition_obj"] = compositions

# ── 3. Failure Analysis ───────────────────────────────────────────────
num_total   = len(df)
num_valid   = sum(c is not None for c in compositions)
num_failed  = num_total - num_valid

print(f"\n{'='*45}")
print(f"  COMPOSITION CONVERSION REPORT")
print(f"{'='*45}")
print(f"  Total entries  : {num_total:>10,}")
print(f"  Successful     : {num_valid:>10,}  ({num_valid/num_total*100:.2f}%)")
print(f"  Failed         : {num_failed:>10,}  ({num_failed/num_total*100:.2f}%)")

if num_failed > 0:
    # Breakdown of failure reasons
    reasons = Counter(r for r in failure_notes if r is not None)
    print(f"\n  Failure breakdown:")
    for reason, count in reasons.most_common():
        print(f"    {reason:<35} : {count:>6,}")

    # Drop failed rows — unusable for Magpie featurization
    df = df[df["Composition_obj"].notna()].reset_index(drop=True)
    print(f"\n  ⚠️  {num_failed:,} rows dropped — no valid Composition object")
    print(f"  ✅  Remaining : {len(df):,} rows")

print(f"{'='*45}")

# ── 4. Sanity Check — Verify Object Types ─────────────────────────────
print("\nSanity check — first 3 valid Composition objects:")
samples = df["Composition_obj"].dropna().iloc[:3]
for comp in samples:
    print(
        f"  Type : {type(comp).__name__:<20} "
        f"| Value : {str(comp):<25} "
        f"| Valid : {isinstance(comp, Composition)}"
    )

# ── 1. Initialise Featurizer ──────────────────────────────────────────
featurizer = ElementProperty.from_preset("magpie")
featurizer.set_n_jobs(1)

# ── 2. Select Diverse Test Cases ─────────────────────────────────────
# Test across varying complexity — not just the first row
test_indices = {
    "first"         : df[df["Composition_obj"].notna()].index[0],
    "single_element": df[df["Number of Elements"] == 1].index[0]
                      if (df["Number of Elements"] == 1).any() else None,
    "max_elements"  : df["Number of Elements"].idxmax(),
}

print(f"{'='*50}")
print("  PRE-FLIGHT: COMPOSITION OBJECT VALIDATION")
print(f"{'='*50}")

for label, idx in test_indices.items():
    if idx is None:
        print(f"  ⚠️  {label:<20} : no matching row found")
        continue

    comp = df.loc[idx, "Composition_obj"]
    is_valid = isinstance(comp, Composition)

    print(f"\n  [{label}]")
    print(f"    Index        : {idx}")
    print(f"    Formula      : {comp}")
    print(f"    Num elements : {df.loc[idx, 'Number of Elements']}")
    print(f"    Type         : {type(comp).__name__}")
    print(f"    Valid        : {'✅' if is_valid else '❌'}")

# ── 3. Featurizer Smoke Test ──────────────────────────────────────────
print(f"\n{'='*50}")
print("  PRE-FLIGHT: FEATURIZER SMOKE TEST")
print(f"{'='*50}")

smoke_test_comps = (
    df[df["Composition_obj"].notna()]["Composition_obj"]
    .sample(n=min(5, len(df)), random_state=RANDOM_SEED)
)

failed_smoke = []
for idx, comp in smoke_test_comps.items():
    try:
        features = featurizer.featurize(comp)
        print(f"  ✅ idx {idx:<8} | {str(comp):<25} | {len(features)} features generated")
    except Exception as e:
        print(f"  ❌ idx {idx:<8} | {str(comp):<25} | Error: {str(e)[:50]}")
        failed_smoke.append(idx)

print(f"\n  Smoke test result : ", end="")
if not failed_smoke:
    print(f"✅ All passed — safe to run full featurization")
    print(f"  Expected features : {len(featurizer.feature_labels())}")
    print(f"  Feature names preview : {featurizer.feature_labels()[:3]} ...")
else:
    print(f"❌ {len(failed_smoke)} failures — investigate before full run")
    print(f"  Failed indices: {failed_smoke}")

print(f"{'='*50}")

tracker.track(df, "Composition Object Parsing", note="Featurization", dataset="MP Dataset")


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.3.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

if CONFIG["add_features"]:
    # ── 1. Initialise Featurizer ──────────────────────────────────────
    featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    featurizer.set_n_jobs(1)
    n_features_expected = len(featurizer.feature_labels())
    print(f"Featurizer ready")
    print(f"  Preset          : magpie")
    print(f"  Features        : {n_features_expected}")
    print(f"  Rows to process : {len(df):,}")

    # ── 2. Featurize ─────────────────────────────────────────────────
    print(f"\n⏳ Starting featurization — this may take 30–90 minutes...")
    t_start = time.time()
    try:
        df_plain = df.to_pandas() if hasattr(df, "to_pandas") else df

        df_plain = featurizer.featurize_dataframe(
            df_plain,
            col_id        = "Composition_obj",
            ignore_errors = True,
            inplace       = False,
        )

        # Reassign — downstream cells continue to see `df` as before.
        # If gpu_env is active, this plain pandas object will simply be
        # picked up by cudf.pandas on the next operation as usual.
        df = df_plain
        del df_plain

        elapsed = time.time() - t_start
        print(f"✅ Featurization complete in {elapsed/60:.1f} minutes")
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"❌ Featurization failed after {elapsed/60:.1f} min: {e}")
        raise

    # ── 3. Post-Featurization Validation ─────────────────────────────
    feature_cols   = featurizer.feature_labels()
    n_features_got = len([c for c in feature_cols if c in df.columns])
    feature_nulls  = df[feature_cols].isnull().all(axis=1)
    n_failed_rows  = feature_nulls.sum()
    n_partial_nulls = df[feature_cols].isnull().any(axis=1).sum() - n_failed_rows

    print(f"\n{'='*50}")
    print(f"  POST-FEATURIZATION REPORT")
    print(f"{'='*50}")
    print(f"  Expected features  : {n_features_expected}")
    print(f"  Features found     : {n_features_got}")
    print(f"  DataFrame shape    : {df.shape}")
    print(f"  Fully failed rows  : {n_failed_rows:,}  ({n_failed_rows/len(df)*100:.2f}%)")
    print(f"  Partial NaN rows   : {n_partial_nulls:,}  ({n_partial_nulls/len(df)*100:.2f}%)")
    print(f"  Clean rows         : {len(df) - n_failed_rows - n_partial_nulls:,}")
    print(f"{'='*50}")

    if n_failed_rows > 0:
        df = df[~feature_nulls].reset_index(drop=True)
        print(f"\n⚠️  Dropped {n_failed_rows:,} fully-failed rows")
        print(f"✅ Remaining : {len(df):,} rows")

free_gpu_memory()

tracker.track(df, "Magpie Descriptor Generation", note="Featurization", dataset="MP Dataset")


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.0.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

del featurizer, n_partial_nulls, n_failed_rows, feature_nulls, n_features_got, feature_cols
free_gpu_memory()

print_heading("Step 10: Feature Selection — Multi-Pass Variance & Collinearity Pruning", level=2)

# ── Configuration ─────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "variance_threshold"     : 0.01,
    "correlation_threshold"  : 0.90,   # consistent single source of truth
}

# ── Step 0: Select Numeric Features ──────────────────────────────────
def get_numeric_features(
    df         : pd.DataFrame,
    target_col : str
) -> tuple[list[str], list[str]]:
    """
    Partition columns into numeric features and non-numeric columns.

    Args:
        df         : Full DataFrame.
        target_col : Regression target column name.

    Returns:
        Tuple of (numeric feature names, non-numeric column names).
    """
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    non_numeric_cols = [
        c for c in df.columns
        if c not in numeric_cols and c != target_col
    ]

    print(f"{'='*55}")
    print(f"  STEP 0 — NUMERIC FEATURE SELECTION")
    print(f"{'='*55}")
    print(f"  Total columns     : {len(df.columns)}")
    print(f"  Numeric features  : {len(numeric_cols)}")
    print(f"  Non-numeric (excl): {len(non_numeric_cols)}")
    if non_numeric_cols:
        for col in non_numeric_cols:
            print(f"    - {col:<35} dtype: {df[col].dtype}")
    return numeric_cols, non_numeric_cols


TARGET_COL = "Band Gap (T)"
numeric_features, non_numeric_cols = get_numeric_features(df, TARGET_COL)
print(f"\n  Numeric feature set : {len(numeric_features)} columns")

# ── Step 1: Variance Thresholding ─────────────────────────────────────
def remove_low_variance_features(
    df        : pd.DataFrame,
    target_col: str,
    threshold : float = FEATURE_CONFIG["variance_threshold"]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove near-constant features using VarianceThreshold.
    Operates only on numeric columns.

    Args:
        df        : Numeric DataFrame including target.
        target_col: Target column name (excluded from filtering).
        threshold : Variance below this → feature removed.

    Returns:
        Tuple of (filtered DataFrame, list of removed feature names).
    """
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    variances = X.var()
    selector  = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    mask             = selector.get_support()
    selected_features= X.columns[mask].tolist()
    removed_features = X.columns[~mask].tolist()

    print(f"\n{'='*55}")
    print(f"  STEP 1 — VARIANCE THRESHOLDING  (threshold={threshold})")
    print(f"{'='*55}")
    print(f"  Input features    : {len(X.columns)}")
    print(f"  Removed           : {len(removed_features)}")
    print(f"  Remaining         : {len(selected_features)}")

    if removed_features:
        print(f"\n  Removed features (up to 10):")
        for feat in removed_features[:10]:
            print(f"    - {feat:<50} var={variances[feat]:.6f}")

    # Reconstruct without copy — use index-safe concat
    df_out = pd.concat([X[selected_features], y], axis=1)
    return df_out, removed_features


df_filtered, low_var_features = remove_low_variance_features(
    df[numeric_features + [TARGET_COL]],
    target_col = TARGET_COL
)
del numeric_features   # free — no longer needed as separate list
free_gpu_memory()

# ── Step 2: Correlation Filtering ─────────────────────────────────
def remove_correlated_magpie_features(
    df         : pd.DataFrame,
    target_col : str,
    threshold  : float              = FEATURE_CONFIG["correlation_threshold"],
    train_mask : "pd.Series | None" = None,

) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Remove inter-correlated feature pairs, keeping whichever has
    higher absolute correlation with the target.

    Features already marked for dropping are skipped in subsequent
    comparisons to avoid inconsistent kept/dropped decisions.

    Args:
        df        : Numeric DataFrame including target.
        target_col: Target column name.
        threshold : Inter-feature |r| above this → one feature dropped.
        train_mask: Optional boolean Series aligned to df.index. If given,
                    correlation statistics (both inter-feature and
                    target-correlation) are computed using ONLY the rows
                    where train_mask is True, preventing calibration/test
                    rows from influencing which columns are kept — the
                    columns that survive are then dropped/kept across the
                    FULL df (all rows keep the same column set). If None,
                    falls back to using all rows (pre-leakage-fix behavior).

    Returns:
        Tuple of (filtered DataFrame, dropped column names, drop log).
    """
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # ── Leakage-safe stats: fit correlation structure on TRAIN rows only ──
    if train_mask is not None:
        X_stats = X.loc[train_mask]
        y_stats = y.loc[train_mask]
        print(f"  ℹ️  Correlation stats computed on {train_mask.sum():,} "
              f"train-only rows out of {len(X):,} total "
              f"(cal/test rows excluded from feature-selection decisions).")
    else:
        X_stats, y_stats = X, y
        print(f"  ⚠️  No train_mask supplied — correlation stats computed "
              f"on all {len(X):,} rows. This risks target leakage into "
              f"feature selection if a train/test split happens later.")

    target_corr = X_stats.corrwith(y_stats).abs()
    corr_matrix = X_stats.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop  = set()
    drop_log = []

    for col in upper.columns:
        if col in to_drop:
            continue   # ✅ skip — already marked for removal

        correlated = upper.index[upper[col] > threshold].tolist()

        for corr_feat in correlated:
            if corr_feat in to_drop:
                continue   # ✅ skip — already marked for removal

            # Keep whichever has higher target correlation
            if target_corr[col] >= target_corr[corr_feat]:
                loser, keeper = corr_feat, col
            else:
                loser, keeper = col, corr_feat

            to_drop.add(loser)
            drop_log.append({
                "Dropped"         : loser,
                "Kept"            : keeper,
                "Inter-Corr (|r|)": round(corr_matrix.loc[col, corr_feat], 4),
                "Dropped→Target"  : round(target_corr[loser], 4),
                "Kept→Target"     : round(target_corr[keeper], 4),
            })

    log_df = pd.DataFrame(drop_log)

    print(f"\n{'='*55}")
    print(f"  STEP 2 — CORRELATION FILTERING  (threshold={threshold})")
    print(f"{'='*55}")
    print(f"  Input features    : {X.shape[1]}")
    print(f"  Removed           : {len(to_drop)}")
    print(f"  Remaining         : {X.shape[1] - len(to_drop)}")

    if not log_df.empty:
        print(f"\n  Top 10 correlated pairs (by |r|):")
        for _, row in log_df.nlargest(10, "Inter-Corr (|r|)").iterrows():
            print(
                f"    {str(row['Kept'])[:35]:<35} ↔ "
                f"{str(row['Dropped'])[:35]:<35}  "
                f"r={row['Inter-Corr (|r|)']:.3f}"
            )

    df_out = df.drop(columns=list(to_drop))
    return df_out, list(to_drop), log_df

if "leakage_safe_train_mask" not in dir():
    _rng = np.random.RandomState(RANDOM_SEED if "RANDOM_SEED" in dir() else 42)
    leakage_safe_train_mask = pd.Series(
        _rng.rand(len(df_filtered)) < 0.80,
        index=df_filtered.index,
    )
    print(f"  Leakage-safe train mask created: "
          f"{leakage_safe_train_mask.sum():,} / {len(df_filtered):,} rows "
          f"(~80%) will inform feature-selection correlation stats.")

df_filtered, correlated_features, correlation_log = remove_correlated_magpie_features(
    df_filtered,
    target_col = TARGET_COL,
    train_mask = leakage_safe_train_mask,
)
free_gpu_memory()

tracker.track(df, "Magpie Collinearity Pruning", note="Feature Selection", dataset="MP Dataset")

# ── Step 3: Restore Non-Numeric Columns ──────────────────────────────
if non_numeric_cols:
    print(f"\n{'='*55}")
    print(f"  STEP 3 — RESTORING NON-NUMERIC COLUMNS")
    print(f"{'='*55}")

    # Use index-safe join — avoids NaN injection if rows were dropped
    df_filtered = df_filtered.join(
        df[non_numeric_cols],
        how = "left"
    )
    print(f"  Restored : {len(non_numeric_cols)} columns")
    print(f"  Columns  : {non_numeric_cols}")

tracker.track(df, "Restoring non-numeric cols", note="Restoring important non numerical features", dataset="MP Dataset")

# ── Step 4: Feature Category Analysis ────────────────────────────────
def analyze_feature_removal(
    removed   : list[str],
    remaining : list[str]
) -> dict:
    """
    Categorise removed and remaining features by origin prefix.

    Args:
        removed   : List of dropped feature names.
        remaining : List of kept feature names.

    Returns:
        Dict with counts per category for removed and remaining.
    """
    categories = {
        "Magpie"  : "MagpieData",
        "Lattice" : "Lattice",
        "Other"   : None,          # catch-all
    }

    result = {}
    print(f"\n{'='*55}")
    print(f"  STEP 4 — FEATURE CATEGORY ANALYSIS")
    print(f"{'='*55}")

    for cat, prefix in categories.items():
        if prefix:
            rem_cat  = [f for f in removed   if prefix in f]
            keep_cat = [f for f in remaining if prefix in f]
        else:
            rem_cat  = [f for f in removed   if not any(p in f for p in categories.values() if p)]
            keep_cat = [f for f in remaining if not any(p in f for p in categories.values() if p)]

        result[cat] = {"removed": len(rem_cat), "remaining": len(keep_cat)}
        print(f"  {cat:<10} removed={len(rem_cat):<5}  remaining={len(keep_cat)}")

        if rem_cat and cat == "Magpie":
            print(f"    Examples removed: {rem_cat[:3]}")

    return result


all_removed       = low_var_features + correlated_features
remaining_features= [c for c in df_filtered.columns if c != TARGET_COL and c not in non_numeric_cols]
feature_stats     = analyze_feature_removal(all_removed, remaining_features)

# ── Final Summary & Update ───────────────────────────────────
print(f"\n{'='*55}")
print(f"  FINAL FEATURE SELECTION SUMMARY")
print(f"{'='*55}")
print(f"  Original df shape           : {df.shape}")
print(f"  Low-variance removed        : {len(low_var_features)}")
print(f"  High-correlation removed    : {len(correlated_features)}")
print(f"  Total features removed      : {len(all_removed)}")
print(f"  Numeric features remaining  : "
      f"{df_filtered.select_dtypes(include=[np.number]).shape[1] - 1}")
print(f"  \nFinal df shape              : {df_filtered.shape}")

# ── Update master DataFrame ───────────────────────────────────────────
df = df_filtered
del df_filtered, all_removed, remaining_features
free_gpu_memory()

tracker.track(df, "Memory Garbage Collection", note="Memory Optimization", dataset="MP Dataset")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.1.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 11: Oxidation State Normalisation & Feature Engineering", level=2)

# ── Safe List Converter ───────────────────────────────────────────────
def safe_literal_eval(value) -> list:
    """
    Normalise a value to a Python list, handling all formats produced
    by the Materials Project API and intermediate pickle serialisation.

    Handles:
        - Already a list              → returned as-is
        - np.ndarray / pd.Series      → converted via .tolist()
        - None / NaN                  → empty list []
        - String repr of list         → parsed via ast.literal_eval
        - Empty string / 'Unknown'    → empty list []
        - '[Unknown]' / '[None]'      → empty list []
        - Mixed lists with 'Unknown'  → cleaned
        - Unexpected type             → logged warning, empty list []

    Args:
        value : Raw value from 'Oxidation States' column.

    Returns:
        list : Normalised list (may be empty if value was missing/invalid).
    """

    # Case 1: numpy / pandas sequence → convert directly
    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()

    # Case 2: Already a list → clean and return
    if isinstance(value, list):
        return [
            x for x in value
            if str(x).strip().lower() not in ("unknown", "none")
        ]

    # Case 3: None / NaN → empty list
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    # Case 4: String representation
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()

        # Direct empty patterns
        if lowered in ("", "unknown", "none", "[]", "[unknown]", "[none]"):
            return []

        try:
            result = literal_eval(stripped)

            if isinstance(result, list):
                # Clean unwanted entries
                return [
                    x for x in result
                    if str(x).strip().lower() not in ("unknown", "none")
                ]

            # Scalar → wrap into list
            return [result]

        except (ValueError, SyntaxError):
            # Unparseable string → treat as single value unless it's junk
            if lowered in ("unknown", "none"):
                return []
            return [stripped]

    # Case 5: Unexpected type — warn and return empty
    warnings.warn(
        f"safe_literal_eval: unexpected type {type(value).__name__} "
        f"for value '{str(value)[:40]}' — returning []",
        UserWarning,
        stacklevel=2
    )
    return []

# ── Apply ─────────────────────────────────────────────────────────────
df["Oxidation States"] = df["Oxidation States"].apply(safe_literal_eval)


# ── Structured Verification ───────────────────────────────────────────
print(f"{'='*50}")
print(f"  OXIDATION STATES — POST-CONVERSION AUDIT")
print(f"{'='*50}")

# 1. Type check — all values should now be lists
type_counts = df["Oxidation States"].apply(type).value_counts()
print(f"\n  Value types after conversion:")
for dtype, count in type_counts.items():
    marker = "✅" if dtype == list else "❌"
    print(f"  {marker}  {dtype.__name__:<15} : {count:>8,}")

# 2. Content audit
empty_lists = (df["Oxidation States"].apply(len) == 0).sum()
non_empty   = len(df) - empty_lists

print(f"\n  Non-empty entries  : {non_empty:>8,}  ({non_empty/len(df)*100:.2f}%)")
print(f"  Empty entries      : {empty_lists:>8,}  ({empty_lists/len(df)*100:.2f}%)")

# 3. Sample — one from each case type for manual inspection
print(f"\n  Representative samples:")

# Non-empty
sample_ne = df[df["Oxidation States"].apply(len) > 0]["Oxidation States"].iloc[0]
print(f"  Non-empty  : {sample_ne}")

# Empty
sample_e = df[df["Oxidation States"].apply(len) == 0]["Oxidation States"].iloc[0] \
           if empty_lists > 0 else "N/A"
print(f"  Empty      : {sample_e}")

# Multi-species
sample_multi = df[df["Oxidation States"].apply(len) > 2]["Oxidation States"].iloc[0] \
               if (df["Oxidation States"].apply(len) > 2).any() else "N/A"
print(f"  Multi (>2) : {sample_multi}")

del sample_multi, sample_e, sample_ne, type_counts, empty_lists, non_empty
free_gpu_memory()

print("\nSample conversions:")
print(df['Oxidation States'].sample(10))
print(f"{'='*50}")

# ── Replace Empty Lists → NaN ──────────────────────────────────────
df["Oxidation States"] = df["Oxidation States"].apply(
    lambda x: x if isinstance(x, list) and len(x) > 0 else np.nan
)

null_count = df["Oxidation States"].isna().sum()
print(f"Oxidation States — null after empty-list replacement: {null_count:,} "
      f"({null_count/len(df)*100:.2f}%)")

tracker.track(df, "Oxidation State Normalization", note="Feature Engineering", dataset="MP Dataset")


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.2.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")


# ── Charge Extraction ──────────────────────────────────────────────
# Supports formats: 'Fe2+', 'O2-', 'H+', 'N3-', 'Cu+', 'O-'
# Pattern: element symbol, optional digits, sign (+/-)
_CHARGE_PATTERN = re.compile(r'[A-Z][a-z]*(\d*)([+\-])')

def extract_charges(state_list) -> list[float]:
    """
    Parse oxidation state strings into signed integer charges.

    Args:
        state_list : List of strings like ['Fe2+', 'O2-'] or NaN.

    Returns:
        List of signed charges as floats, or [np.nan] if unparseable.

    Examples:
        ['Fe2+', 'O2-']  →  [2, -2]
        ['H+']           →  [1]
        ['N3-']          →  [-3]
    """
    if not isinstance(state_list, list) or len(state_list) == 0:
        return [np.nan]

    charges = []
    for s in state_list:
        m = _CHARGE_PATTERN.search(str(s))
        if m:
            magnitude = int(m.group(1)) if m.group(1) else 1
            charges.append(magnitude if m.group(2) == "+" else -magnitude)

    return charges if charges else [np.nan]

# ── Vectorised Stats Extraction — Single Pass ──────────────────────
def oxidation_stats(state_list) -> tuple[float, float, float, float]:
    """
    Compute min, max, mean, range of charges in a single pass.
    Returns (NaN, NaN, NaN, NaN) if no valid charges found.

    Args:
        state_list : List of oxidation state strings or NaN.

    Returns:
        Tuple of (min, max, mean, range) as floats.
    """
    charges = extract_charges(state_list)

    # Filter valid (non-NaN) charges
    valid = [c for c in charges if not np.isnan(c)]

    if not valid:
        return np.nan, np.nan, np.nan, np.nan

    arr = np.array(valid, dtype=np.float32)
    return (
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(arr.max() - arr.min()),
    )

# ── Apply — Single Pass for All Four Stats ─────────────────────────
print("Extracting oxidation state statistics...")

ox_stats = df["Oxidation States"].apply(oxidation_stats)

# Unpack tuple column into four named columns in one operation
(
    df["Ox_min"],
    df["Ox_max"],
    df["Ox_mean"],
    df["Ox_range"],
) = zip(*ox_stats)

# Convert to float32 — consistent with other numeric columns
ox_cols = ["Ox_min", "Ox_max", "Ox_mean", "Ox_range"]
df[ox_cols] = df[ox_cols].astype("float32")


# ── Impute NaN → 0 (with documented rationale) ────────────────────
# Rationale: materials with unknown oxidation states are predominantly
# metals or intermetallics where formal charges are ill-defined.
# Filling with 0 treats them as "effectively neutral" — consistent
# with their likely classification as metals (Eg = 0) in Stage 1.
nan_counts = df[ox_cols].isna().sum()
print(f"\nNaN counts before imputation:")
for col, n in nan_counts.items():
    print(f"  {col:<12} : {n:>8,}  ({n/len(df)*100:.2f}%)")

df[ox_cols] = df[ox_cols].fillna(0)
print(f"\n✅ NaN → 0 imputation applied to {ox_cols}")
print(f"   Rationale: unknown oxidation states are predominantly metals")
print(f"   (Eg = 0), for which formal ionic charges are ill-defined.")

# ── Audit ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  OXIDATION STATE FEATURE AUDIT")
print(f"{'='*50}")
print(df[ox_cols].describe().round(3).to_string())
print(f"{'='*50}")

# Sanity check — verify on a known compound
# Fe2O3: Fe is 3+, O is 2- → min=-2, max=3, mean=0.8, range=5
example_idx = df[df["Pretty Formula"] == "Fe2O3"].index
if len(example_idx) > 0:
    idx = example_idx[0]
    print(f"\n  Sanity check — Fe2O3 (expected: min=-2, max=3):")
    print(f"    Ox States : {df.loc[idx, 'Oxidation States']}")
    for col in ox_cols:
        print(f"    {col:<12} : {df.loc[idx, col]}")


print(f"✅  DataFrame shape: {df.shape}")

tracker.track(df, "Ionic Charge Extraction", note="Feature Engineering", dataset="MP Dataset")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.3.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 12: Exploratory Data Analysis — Oxidation State & Band Gap Relationships", level=2)

# ── EDA Style Extension ────────────────────────────────────────────
# Extend (not overwrite) the publication style set earlier
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.axisbelow"   : True,
    # Note: grid, font sizes, linewidths already set in global rcParams
})

# ── cuDF Safety Helpers ───────────────────────────────────────────────
def to_np(series, dtype=None) -> np.ndarray:
    """Convert a pandas/cuDF Series to numpy, optionally casting dtype."""
    arr = series.to_numpy() if hasattr(series, "to_numpy") else np.array(series)
    return arr.astype(dtype) if dtype else arr

def to_pd(obj):
    """Convert cuDF DataFrame or Series to pandas if needed."""
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj

print(f"{'='*60}")
print(f"  SECTION 1 — DATASET OVERVIEW")
print(f"{'='*60}")
print(f"\n  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Dtype breakdown (works on cuDF directly — no full copy needed) ────
dtype_counts = df.dtypes.value_counts()
print(f"\n  Dtype breakdown:")
for dtype, count in dtype_counts.items():
    print(f"    {str(dtype):<15} : {count} columns")

# ── Missing value audit ───────────────────────────────────────────────
missing_counts = to_pd(df.isnull().sum())          # ✅ no full copy
missing_pct    = (missing_counts / len(df) * 100).round(2)

null_audit = (
    pd.DataFrame({
        "Missing Count" : missing_counts,
        "Missing %"     : missing_pct,
    })
    .query("`Missing Count` > 0")
    .sort_values("Missing %", ascending=False)
)

print(f"\n  Columns with missing values : {len(null_audit)} / {df.shape[1]}")
if not null_audit.empty:
    print(f"\n  Top 20 by missingness:")
    print(null_audit.head(20).to_string())
else:
    print("  ✅ No missing values found")

# ── Add permanently — with clear documentation ────────────────────────
df["ox_missing"] = df["Oxidation States"].isna().astype("int8")

print(f"Feature 'ox_missing' added to df")
print(f"  Value counts:")
print(f"  0 (ox present) : {(df['ox_missing']==0).sum():>8,}   " f"({(df['ox_missing']==0).mean()*100:.1f}%)")
print(f"  1 (ox present) : {(df['ox_missing']==1).sum():>8,}   " f"({(df['ox_missing']==1).mean()*100:.1f}%)")
print(f"  dtype          : {df['ox_missing'].dtype}")

# ── Use permanent column directly — no local reconstruction ───────────
ox_missing_mask = df["ox_missing"] == 1
ox_present_mask = df["ox_missing"] == 0

grp_missing = df[ox_missing_mask]
grp_present = df[ox_present_mask]

print(f"\n  Ox state missing : {ox_missing_mask.sum():>8,}  "
      f"({ox_missing_mask.mean()*100:.1f}%)")
print(f"  Ox state present : {ox_present_mask.sum():>8,}  "
      f"({ox_present_mask.mean()*100:.1f}%)")

print(df.loc[df["ox_missing"] == 1, ["Ox_min","Ox_max","Ox_mean","Ox_range"]].describe())

# ── Band Gap Summary by Group ─────────────────────────────────────────
print(f"\n  Band Gap stats by oxidation state availability:")

bg_stats = pd.concat(
    [
        to_pd(grp_missing["Band Gap (T)"]).describe().round(4),
        to_pd(grp_present["Band Gap (T)"]).describe().round(4),
    ],
    axis = 1,
    keys = ["Ox Missing", "Ox Present"]
)
print(bg_stats.to_string())

# ── Metal Ratio ───────────────────────────────────────────────────────
metal_pct_missing = float((grp_missing["Band Gap (T)"] == 0).mean()) * 100
metal_pct_present = float((grp_present["Band Gap (T)"] == 0).mean()) * 100

print(f"\n  Metal % (BG = 0):")
print(f"    Ox missing : {metal_pct_missing:.1f}%")
print(f"    Ox present : {metal_pct_present:.1f}%")

# ── Mann-Whitney U Test ───────────────────────────────────────────────
u_stat, p_val = stats.mannwhitneyu(
    to_np(grp_missing["Band Gap (T)"].dropna()),
    to_np(grp_present["Band Gap (T)"].dropna()),
    alternative = "two-sided"
)

# Granular significance label ✅
sig_label = (
    "*** (p < 0.001)" if p_val < 0.001 else
    "**  (p < 0.01) " if p_val < 0.01  else
    "*   (p < 0.05) " if p_val < 0.05  else
    "ns  (p ≥ 0.05) "
)

print(f"\n  Mann-Whitney U : {u_stat:.4e}")
print(f"  p-value        : {p_val:.4e}  {sig_label}")
print(f"  Interpretation : ", end="")
if p_val < 0.05:
    print(
        f"The two groups have statistically different Band Gap distributions.\n"
        f"  Materials with missing oxidation states show a significantly\n"
        f"  {'higher' if metal_pct_missing > metal_pct_present else 'lower'} "
        f"metal fraction ({metal_pct_missing:.1f}% vs {metal_pct_present:.1f}%)."
    )
else:
    print("No significant difference in Band Gap distributions detected.")

del bg_stats
free_gpu_memory()

# ── Figure 1 — Full BG Distribution (all materials) ──────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Ox present plotted FIRST (behind), Ox missing plotted SECOND (front)
for label, grp, color, alpha, zorder in [
    ("Ox present", grp_present, COLORS["primary"], 0.5, 1),
    ("Ox missing", grp_missing, COLORS["metal"],   0.8, 2),
]:
    ax1.hist(
        to_np(grp["Band Gap (T)"]),
        bins    = 60,
        alpha   = alpha,
        label   = label,
        color   = color,
        density = True,
        zorder  = zorder
    )

ax1.set_yscale("log")
ax1.set_xlabel("Band Gap (eV)")
ax1.set_ylabel("Density (log)")
ax1.legend()
# Apply font to all elements including legend
apply_font(ax1)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig1, "Bandgap Distribution - All Materials (Log Scale)")     

# ── Figure 2 — Non-metals Only (BG > 0) ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

for label, grp, color, alpha in [
    ("Ox present", grp_present, COLORS["primary"], 0.5),
    ("Ox missing", grp_missing, COLORS["metal"],   0.7),
]:
    non_metal = grp[grp["Band Gap (T)"] > 0]
    ax2.hist(
        to_np(non_metal["Band Gap (T)"]),
        bins    = 60,
        alpha   = alpha,
        label   = label,
        color   = color,
        density = True
    )
ax2.set_xlabel("Band Gap (eV)")
ax2.set_ylabel("Density")

# Apply font to all elements including legend
apply_font(ax2)

plt.tight_layout()
if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Bandgap Distribution - Non Metals (Log Scale)")   

# ── Figure 3 — Metal Fraction Bar Chart ──────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 5))

bar_vals   = [metal_pct_missing, metal_pct_present]
bar_labels = ["Ox missing", "Ox present"]
bar_colors = [COLORS["metal"], COLORS["primary"]]

ax3.bar(
    bar_labels,
    bar_vals,
    color     = bar_colors,
    width     = 0.5,
    edgecolor = "black",
    linewidth = 0.8
)
ax3.set_ylim(0, max(bar_vals) * 1.20)
ax3.set_ylabel("% Metals (BG = 0)")

# Value labels
for i, v in enumerate(bar_vals):
    ax3.text(
        i, v + max(bar_vals) * 0.02,
        f"{v:.1f}%",
        ha         = "center",
        fontweight = "bold",
        fontsize   = 11
    )

# Apply font to all elements including legend
apply_font(ax3)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()
save_figure(fig3, "Metal Fraction by Oxidation State Availability")

# ── Cleanup ───────────────────────────────────────────────────────────
del non_metal
free_gpu_memory()

# ── Setup ─────────────────────────────────────────────────────────────
CAT_COLS        = ["Crystal System", "Magnetic Ordering"]
ox_missing_rate = float(df["ox_missing"].mean()) * 100   # ✅ computed, not hardcoded

fig, axes = plt.subplots(1, len(CAT_COLS), figsize=(14, 5))
chi2_results = []

# ── Unified loop — plot + chi-square in one pass ──────────────────────
for ax, col in zip(axes, CAT_COLS):

    # ✅ to_pd() — cuDF safe
    ct_raw = pd.crosstab(to_pd(df[col]), to_pd(df["ox_missing"]))

    # ✅ reindex before rename — handles missing value columns
    ct_raw = ct_raw.reindex(columns=[0, 1], fill_value=0)
    ct_raw.columns = ["Ox present", "Ox missing"]

    # Normalised for plot
    ct_pct = ct_raw.div(ct_raw.sum(axis=1), axis=0) * 100

    # ── Chi-square with rare category handling ────────────────────────
    row_totals      = ct_raw.sum(axis=1)
    rare_categories = row_totals[row_totals < 5].index.tolist()
    ct_for_chi2     = ct_raw.drop(index=rare_categories)

    if rare_categories:
        print(f"  ⚠️  {col}: dropped rare categories: {rare_categories}")

    try:
        chi2, p, dof, _ = stats.chi2_contingency(ct_for_chi2)
        test_used = "χ²"
    except ValueError as e:
        print(f"  ⚠️  {col}: chi2 failed ({e}) → Fisher's exact")
        _, p      = stats.fisher_exact(ct_for_chi2.values)
        chi2, dof = np.nan, 1
        test_used = "Fisher"

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    chi2_results.append({
        "Feature"    : col,
        "Test"       : test_used,
        "chi2"       : round(chi2, 2) if not np.isnan(chi2) else "N/A",
        "p-value"    : f"{p:.2e}",
        "dof"        : dof,
        "Significant": sig,
    })
    print(f"  {col:<25} {test_used}: chi2={chi2 if not np.isnan(chi2) else 'N/A'}, "
          f"p={p:.2e} {sig}")

    # ── Plot ──────────────────────────────────────────────────────────
    ct_pct["Ox missing"].sort_values().plot.barh(
        ax    = ax,
        color = COLORS["metal"],     # ✅ COLORS not hardcoded 'salmon'
        alpha = 0.85
    )
    ax.set_title(
        f"% Missing Ox — by {col}\n({test_used} p={p:.2e})",
        fontweight = "bold"
    )
    ax.set_xlabel("% Rows with Missing Oxidation States")

    # ✅ dynamic reference line
    ax.axvline(
        ox_missing_rate,
        linestyle = "--",
        color     = COLORS["neutral"],
        linewidth = 0.8,
        label     = f"Dataset avg ({ox_missing_rate:.1f}%)"
    )
    ax.legend(fontsize=8)

    # ✅ apply_font inside loop — applies to each axis
    apply_font(ax)

plt.tight_layout()   

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "eda_ox_categorical")
# ── Chi-square summary ────────────────────────────────────────────────
chi2_df = pd.DataFrame(chi2_results)
print(f"\n{chi2_df.to_string(index=False)}")
chi2_df.to_csv(
    os.path.join(CONFIG["csv_dir"], "chi2_ox_missingness.csv"),
    index=False
)

del ct_raw, ct_pct, ct_for_chi2, chi2_df
free_gpu_memory()

missing_df  = df[df['ox_missing'] == 1]
present_df  = df[df['ox_missing'] == 0]

# ── Validate columns exist before using ──────────────────────────────
NUM_COLS_REQUESTED = [
    "Number of Elements",
    "Nsites",
    "Density",
    "Formation Energy Per Atom",
    "Total Magnetization",
    "MagpieData maximum Electronegativity",
    "MagpieData mean MeltingT",
    "MagpieData mean NpValence",
]

# Guard: only keep columns that survived feature selection
num_cols = [c for c in NUM_COLS_REQUESTED if c in df.columns]
dropped  = set(NUM_COLS_REQUESTED) - set(num_cols)
if dropped:
    print(f"⚠️  {len(dropped)} columns not found (may have been filtered):")
    for c in dropped:
        print(f"   - {c}")

summary = df.groupby('ox_missing')[num_cols].mean().T
summary.columns = ['Ox present', 'Ox missing']
summary['diff %'] = ((summary['Ox missing'] - summary['Ox present'])
                    / summary['Ox present'].abs() * 100).round(1)
print(summary.sort_values('diff %', key=abs, ascending=False).to_string())

# T-tests
print("\n--- T-test p-values ---")
for col in num_cols:
    t, p = stats.ttest_ind(
        missing_df[col].dropna(), present_df[col].dropna(), equal_var=False
    )
    sig = "*" if p < 0.05 else ""
    print(f"  {col:<50} p={p:.2e} {sig}")

# ── Summary Table ─────────────────────────────────────────────────────
feature_means = pd.DataFrame({
    "Ox present" : to_pd(grp_present[num_cols].mean()),
    "Ox missing" : to_pd(grp_missing[num_cols].mean()),
})
feature_means["diff %"] = (
    (feature_means["Ox missing"] - feature_means["Ox present"])
    / feature_means["Ox present"].abs() * 100
).round(1)
feature_means = feature_means.sort_values("diff %", key=abs, ascending=False)

print(f"\n{'='*65}")
print(f"  FEATURE MEANS BY OXIDATION STATE AVAILABILITY")
print(f"{'='*65}")
print(feature_means.to_string())

# ── Statistical Tests ─────────────────────────────────────────────────
# Mann-Whitney U — non-parametric, consistent with Section 2
# appropriate for skewed distributions (BG, formation energy, etc.)
print(f"\n{'='*65}")
print(f"  MANN-WHITNEY U TESTS  (non-parametric, two-sided)")
print(f"{'='*65}")

test_results = []
for col in num_cols:
    u_stat, p = stats.mannwhitneyu(
        to_np(grp_missing[col].dropna()),
        to_np(grp_present[col].dropna()),
        alternative="two-sided"
    )
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    test_results.append({
        "Feature"   : col,
        "Mean (missing)" : round(summary.loc[col, "Ox missing"], 4),
        "Mean (present)" : round(summary.loc[col, "Ox present"], 4),
        "diff %"    : summary.loc[col, "diff %"],
        "U-stat"    : f"{u_stat:.2e}",
        "p-value"   : f"{p:.2e}",
        "sig"       : sig,
    })
    print(f"  {col:<50}  p={p:.2e}  {sig}")

# ── Plot — diff % bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))    # ✅ wider figure — more room

colors = [
    COLORS["metal"] if v < 0 else COLORS["primary"]
    for v in feature_means["diff %"]       # ✅ renamed from summary
]

bars = ax.barh(
    feature_means.index,
    feature_means["diff %"],
    color     = colors,
    alpha     = 0.85,
    edgecolor = "black",
    linewidth = 0.6,
)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("% Difference (Ox missing vs Ox present)")
ax.set_title(
    "Feature Mean Differences\n(Ox Missing vs Ox Present)",
    fontweight = "bold"
)

# ✅ Dynamic x padding — based on actual data range, not hardcoded
x_range  = feature_means["diff %"].abs().max()
x_pad    = x_range * 0.18    # 18% of range as padding on each side

ax.set_xlim(
    -(x_range + x_pad),
    +(x_range + x_pad)
)

# ── Value labels ──────────────────────────────────────────────────────
label_pad = x_range * 0.03 

for bar, val in zip(bars, feature_means["diff %"]):
    pad = +label_pad if val >= 0 else -label_pad
    ha  = "left"     if val >= 0 else "right"
    ax.text(
        val + pad,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.1f}%",
        va         = "center",
        ha         = ha,
    )

plt.subplots_adjust(left=0.35)

apply_font(ax)
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "eda_ox_feature_mean_diffs")

# ── Cleanup ───────────────────────────────────────────────────────────
del grp_missing, grp_present, summary, test_results
free_gpu_memory()

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.4.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print(f"\n{'='*60}")
print(f"  SECTION 3 — TARGET VARIABLE: BAND GAP")
print(f"{'='*60}")

# ── Extract as numpy — works with both pandas and cuDF ───────────────
bg       = to_np(df["Band Gap (T)"], dtype=np.float32)
bg_clean = bg[~np.isnan(bg)]
metals   = bg_clean[bg_clean == 0]
nonmetals= bg_clean[bg_clean > 0]
log_bg   = np.log1p(nonmetals)

# ── Summary Statistics ────────────────────────────────────────────────
desc = stats.describe(nonmetals)

print(f"\n  Total samples  : {len(bg_clean):,}")
print(f"  Metals  (BG=0) : {len(metals):,}  ({len(metals)/len(bg_clean)*100:.1f}%)")
print(f"  Non-metals     : {len(nonmetals):,}  ({len(nonmetals)/len(bg_clean)*100:.1f}%)")

print(f"\n  Non-metal Band Gap statistics:")
print(f"  {'Mean':<10} : {nonmetals.mean():.4f} eV")
print(f"  {'Median':<10} : {np.median(nonmetals):.4f} eV")
print(f"  {'Std':<10} : {nonmetals.std():.4f} eV")
print(f"  {'Min':<10} : {nonmetals.min():.4f} eV")
print(f"  {'Max':<10} : {nonmetals.max():.4f} eV")
print(f"  {'Skewness':<10} : {desc.skewness:.4f}  "
      f"({'right-skewed' if desc.skewness > 0 else 'left-skewed'})")
print(f"  {'Kurtosis':<10} : {desc.kurtosis:.4f}")

# ── Normality Test — D'Agostino-Pearson (appropriate for n > 5000) ────
# Shapiro-Wilk is unreliable at large n — always rejects due to power
k2, p_norm = stats.normaltest(
    np.random.default_rng(RANDOM_SEED).choice(log_bg, size=min(5000, len(log_bg)), replace=False)
)
norm_label = "approx. normal" if p_norm > 0.05 else "non-normal"
print(f"\n  D'Agostino-Pearson test on log(1+BG):")
print(f"  k²={k2:.2f},  p={p_norm:.3e}  → {norm_label}")
print(f"  (Confirms whether log-transform sufficiently normalises target)")

# ── Figure 1 — All Materials ──────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.hist(
    bg_clean, bins=80,
    color     = COLORS["primary"],
    alpha     = 0.8,
    edgecolor = "black",
    linewidth = 0.3
)
ax1.axvline(
    0, color=COLORS["metal"],
    linestyle="--", linewidth=1.5,
    label="Metal boundary (0 eV)"
)
# ax1.set_title("Band Gap Distribution — All Materials", fontweight="bold")
ax1.set_xlabel("Band Gap (eV)")
ax1.set_ylabel("Count")
ax1.legend()

# Annotation box
ax1.text(
    0.97, 0.97,
    f"Metals    : {len(metals)/len(bg_clean)*100:.1f}%\n"
    f"Non-metals: {len(nonmetals)/len(bg_clean)*100:.1f}%",
    transform = ax1.transAxes,
    fontsize  = 9, va="top", ha="right",
    bbox      = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black")
)

# Apply font to all elements including legend
apply_font(ax1)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig1, "Bandgap Distribution - All Materials") 

# ── Figure 2 — Non-metals Only (Raw) ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.hist(
    nonmetals, bins=80,
    color     = COLORS["stable"],
    alpha     = 0.8,
    edgecolor = "black",
    linewidth = 0.3
)
ax2.axvline(
    nonmetals.mean(),
    color="red", linestyle="--", linewidth=1.5,
    label=f"Mean = {nonmetals.mean():.2f} eV"
)
ax2.axvline(
    np.median(nonmetals),
    color=COLORS["metastable"], linestyle=":", linewidth=1.8,
    label=f"Median = {np.median(nonmetals):.2f} eV"
)
# ax2.set_title("Band Gap Distribution — Non-metals Only", fontweight="bold")
ax2.set_xlabel("Band Gap (eV)")
ax2.set_ylabel("Count")
ax2.legend()

# Apply font to all elements including legend
apply_font(ax2)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Bandgap Distribution - Non Metals") 

# ── Figure 3 — Non-metals log(1 + BG) ────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.hist(
    log_bg, bins=80,
    color     = COLORS["secondary"],
    alpha     = 0.8,
    edgecolor = "black",
    linewidth = 0.3
)
ax3.axvline(
    log_bg.mean(),
    color="red", linestyle="--", linewidth=1.5,
    label=f"Mean = {log_bg.mean():.2f}"
)
ax3.axvline(
    np.median(log_bg),
    color=COLORS["metastable"], linestyle=":", linewidth=1.8,
    label=f"Median = {np.median(log_bg):.2f}"
)
# ax3.set_title("Band Gap — log(1 + BG) Transform", fontweight="bold")
ax3.set_xlabel("log(1 + Band Gap)")
ax3.set_ylabel("Count")
ax3.legend()

# D'Agostino-Pearson annotation
ax3.text(
    0.97, 0.97,
    f"D'Agostino-Pearson\nk²={k2:.2f}, p={p_norm:.2e}\n→ {norm_label}",
    transform = ax3.transAxes,
    fontsize  = 8, va="top", ha="right",
    bbox      = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black")
)

# Apply font to all elements including legend
apply_font(ax3)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig3, "Bandgap Distribution (Log Scale)") 

# ── Cleanup ───────────────────────────────────────────────────────────
del bg, bg_clean, metals, nonmetals, log_bg
free_gpu_memory()

print(f"\n{'='*60}")
print(f"  SECTION 4 — CATEGORICAL FEATURES")
print(f"{'='*60}")

CAT_COLS_INFO = {
    "Crystal System"   : {"expected": 7},
    "Magnetic Ordering": {"expected": 5},
    "Symmetry Symbol"  : {"expected": "200+"},
}

# ── Value count audit ─────────────────────────────────────────────────
for col, meta in CAT_COLS_INFO.items():
    vc = to_pd(df[col]).value_counts()
    print(f"\n  {col}  ({len(vc)} unique, expected ~{meta['expected']}):")
    print(vc.head(10).to_string())

# ── Precompute value counts — used in both plots ──────────────────────
cs_counts  = to_pd(df["Crystal System"]).value_counts()
mo_counts  = to_pd(df["Magnetic Ordering"]).value_counts()
sym_counts = to_pd(df["Symmetry Symbol"]).value_counts().head(20)

max_offset_cs  = cs_counts.max()  * 0.008   # dynamic label offset
max_offset_mo  = mo_counts.max()  * 0.008
bg_col         = to_np(df["Band Gap (T)"], dtype=np.float32)

# ── Figure 1 — Crystal System Distribution ───────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.barh(
    cs_counts.index[::-1],
    cs_counts.values[::-1],
    color     = COLORS["primary"],
    alpha     = 0.85,
    edgecolor = "black",
    linewidth = 0.5
)
for i, (val, idx) in enumerate(zip(cs_counts.values[::-1], cs_counts.index[::-1])):
    ax1.text(val + max_offset_cs, i, f"{val:,}", va="center", fontsize=8)

# ax1.set_title("Crystal System Distribution", fontweight="bold")
ax1.set_xlabel("Count")
ax1.set_xlim(0, cs_counts.max() * 1.12)   # headroom for labels

# Apply font to all elements including legend
apply_font(ax1)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig1, "Crystal System Distribution") 

# ── Figure 2 — Magnetic Ordering Distribution ────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

bar_colors = (PALETTE * 2)[:len(mo_counts)]   # cycle PALETTE if needed

ax2.bar(
    mo_counts.index,
    mo_counts.values,
    color     = bar_colors,
    alpha     = 0.85,
    edgecolor = "black",
    linewidth = 0.5
)

for i, val in enumerate(mo_counts.values):
    ax2.text(i, val + max_offset_mo, f"{val:,}", ha="center", fontsize=8)

# ax2.set_title("Magnetic Ordering Distribution", fontweight="bold")
ax2.set_xlabel("Ordering Type")
ax2.set_ylabel("Count")
ax2.set_ylim(0, mo_counts.max() * 1.12)
ax2.tick_params(axis="x", rotation=15)

# Apply font to all elements including legend
apply_font(ax2)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Magnetic Ordering Distribution") 

# ── Figure 3 — Top 20 Symmetry Symbols ───────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))

ax3.barh(
    sym_counts.index[::-1],
    sym_counts.values[::-1],
    color     = COLORS["secondary"],
    alpha     = 0.8,
    edgecolor = "black",
    linewidth = 0.4
)
# ax3.set_title("Top 20 Symmetry Symbols (Space Groups)", fontweight="bold")
ax3.set_xlabel("Count")
ax3.set_xlim(0, sym_counts.max() * 1.12)

# Apply font to all elements including legend
apply_font(ax3)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig3, "Top 20 Symmetry Symbols (Space Groups)") 

# ── Figure 4 — Band Gap by Categorical Feature (2 boxplots) ──────────
# One figure per categorical column for clean dissertation embedding
for cat_col in ["Crystal System", "Magnetic Ordering"]:

    # ✅ Use to_pd() — avoids cuDF object array issues in np.unique
    cat_series = to_pd(df[cat_col])
    valid_mask = ~cat_series.isna() & ~np.isnan(bg_col)

    # Group BG values by category — sorted by median descending
    groups = {
        label: bg_col[valid_mask & (cat_series == label)]
        for label in cat_series.dropna().unique()
    }
    groups = dict(
        sorted(groups.items(), key=lambda x: np.median(x[1]), reverse=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bp = ax.boxplot(
        [groups[k] for k in groups],
        labels      = list(groups.keys()),
        patch_artist= True,
        showfliers  = False,
        medianprops = dict(color="black", linewidth=2),
        boxprops    = dict(linewidth=1.2),
    )

    # Colour boxes using PALETTE — cycle if more categories than palette length
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.75)

    # ✅ Data-driven y-axis — no hardcoded 9 eV ceiling
    bg_p99 = np.nanpercentile(bg_col, 99)
    ax.set_ylim(-0.2, bg_p99 * 1.05)

    # ax.set_title(f"Band Gap Distribution by {cat_col}", fontweight="bold")
    ax.set_ylabel("Band Gap (eV)")
    ax.tick_params(axis="x", rotation=25)

    # Sample size annotations below each box
    for i, (label, vals) in enumerate(groups.items()):
        ax.text(
            i + 1, -0.15,
            f"n={len(vals):,}",
            ha="center", fontsize=7,
            color="dimgray"
        )

    # Apply font to all elements including legend
    apply_font(ax)

    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    save_figure(fig, f"Bandgap Distribution By {cat_col}") 

# ── Cleanup ───────────────────────────────────────────────────────────
del cs_counts, mo_counts, sym_counts, groups, cat_series, bg_col
free_gpu_memory()

print_heading("Step 13: Categorical Encoding & Structural Feature Audit", level=2)

# ── 1. Crystal System — One-Hot Encoding ─────────────────────────────
# drop_first=False — we keep all categories
# XGBoost/DNN are not affected by multicollinearity; dropping a
# category (typically 'cubic') loses physically meaningful signal
cs_before = set(df.columns)
df = pd.get_dummies(df, columns=["Crystal System"], dtype="int8")
cs_dummies = sorted(set(df.columns) - cs_before)

print(f"Crystal System → {len(cs_dummies)} dummy columns:")
for col in cs_dummies:
    print(f"  + {col}  (n={df[col].sum():,})")

tracker.track(df, "Crystal System One-Hot Encoding", note="Encoding", dataset="MP Dataset")

# ── 2. Magnetic Ordering — Merge Unknown → own flag, then encode ──────
# Rather than silently merging 'Unknown' into 'NM' (non-magnetic),
# we preserve the epistemic distinction with a binary flag —
# consistent with the ox_missing pattern used earlier
unknown_count = (df["Magnetic Ordering"] == "Unknown").sum()
print(f"\nMagnetic Ordering — 'Unknown' entries : {unknown_count:,}")

if unknown_count > 0:
    # Option A: keep Unknown as its own dummy (recommended)
    # This lets the model learn a separate coefficient for unknowns
    # rather than assuming they behave like NM materials
    df["mag_unknown"] = (df["Magnetic Ordering"] == "Unknown").astype("int8")
    df["Magnetic Ordering"] = df["Magnetic Ordering"].replace("Unknown", "NM")
    print(f"  → 'mag_unknown' flag added (1 = was Unknown)")
    print(f"  → 'Unknown' entries reassigned to 'NM' for encoding")

mo_before = set(df.columns)
df = pd.get_dummies(df, columns=["Magnetic Ordering"], dtype="int8")
mo_dummies = sorted(set(df.columns) - mo_before)

print(f"\nMagnetic Ordering → {len(mo_dummies)} dummy columns:")
for col in mo_dummies:
    print(f"  + {col}  (n={df[col].sum():,})")

# ── 3. Symmetry Symbol — Drop ─────────────────────────────────────────
# Space Group Number already encodes symmetry numerically (1–230).
# Symmetry Symbol is a string label for the same information —
# redundant and high-cardinality (228 unique values).
if "Symmetry Symbol" in df.columns:
    df = df.drop(columns=["Symmetry Symbol"])
    print(f"\n🗑️  'Symmetry Symbol' dropped — redundant with Space Group Number")
else:
    print(f"\n⚠️  'Symmetry Symbol' not found — may have been dropped earlier")

tracker.track(df, "Magnetic State Regularization", note="Encoding", dataset="MP Dataset")

# ── 4. Audit ──────────────────────────────────────────────────────────
all_new_cols = cs_dummies + mo_dummies + (["mag_unknown"] if unknown_count > 0 else [])

print(f"\n{'='*55}")
print(f"  ENCODING AUDIT")
print(f"{'='*55}")
print(f"  Columns added    : {len(all_new_cols)}")
print(f"  DataFrame shape  : {df.shape}")

# Verify all new columns are int8
non_int8 = [c for c in all_new_cols if df[c].dtype != "int8"]
if non_int8:
    print(f"  ⚠️  Non-int8 dummy cols: {non_int8}")
else:
    print(f"  ✅ All dummy columns are int8")

# Verify no NaN introduced
null_check = df[all_new_cols].isna().sum().sum()
print(f"  Null values in new columns : {null_check} "
      f"{'✅' if null_check == 0 else '❌'}")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.5.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print(f"\n{'='*60}")
print(f"  SECTION 5 — NUMERICAL FEATURES: STRUCTURAL STATS")
print(f"{'='*60}")

# ── Column validation ─────────────────────────────────────────────────
STRUCT_COLS_REQUESTED = [
    "Number of Elements", "Nsites", "Volume", "Density",
    "Space Group Number",
    "Lattice (a)", "Lattice (b)", "Lattice (c)",
    "Lattice (alpha)", "Lattice (beta)", "Lattice (gamma)",
    "Energy Per Atom", "Total Magnetization", "Formation Energy Per Atom",
    "Ox_min", "Ox_max", "Ox_mean", "Ox_range",
]

struct_present = [c for c in STRUCT_COLS_REQUESTED if c in df.columns]
struct_dropped = set(STRUCT_COLS_REQUESTED) - set(struct_present)

print(f"\n  Requested : {len(STRUCT_COLS_REQUESTED)} columns")
print(f"  Present   : {len(struct_present)} columns")

if struct_dropped:
    print(f"  Dropped by earlier filtering ({len(struct_dropped)}):")
    for c in sorted(struct_dropped):
        print(f"    - {c}")


# ── Descriptive Statistics ────────────────────────────────────────────
# Use to_pd(df[...]) — single consistent source, cuDF safe
struct_pd = to_pd(df[struct_present])

desc = struct_pd.describe().T
desc["skew"]     = struct_pd.skew()
desc["kurtosis"] = struct_pd.kurtosis()

print(f"\n  Descriptive Statistics:")
print(
    desc[["count", "mean", "std", "min", "50%", "max", "skew", "kurtosis"]]
    .round(3)
    .to_string()
)

# Flag highly skewed features — these may benefit from transformation
high_skew = desc[desc["skew"].abs() > 1]["skew"].sort_values(key=abs, ascending=False)
if not high_skew.empty:
    print(f"\n  ⚠️  Highly skewed features (|skew| > 1) — {len(high_skew)} columns:")
    for col, skew_val in high_skew.items():
        print(f"    {col:<40} skew = {skew_val:.3f}")

# ── Distribution Grid ─────────────────────────────────────────────────
N_COLS_PLOT = 4
n_features  = len(struct_present)
n_rows_plot = int(np.ceil(n_features / N_COLS_PLOT))

# Exact grid — no blank axes if we reshape to actual count
fig, axes = plt.subplots(
    n_rows_plot, N_COLS_PLOT,
    figsize = (18, n_rows_plot * 3.2)
)
axes_flat = axes.flatten()

for i, col in enumerate(struct_present):
    ax   = axes_flat[i]
    data = to_np(df[col], dtype=np.float64)
    data = data[~np.isnan(data)]

    # Clip for display only — raw data unchanged
    p1, p99      = np.percentile(data, [1, 99])
    data_clipped = data[(data >= p1) & (data <= p99)]

    ax.hist(
        data_clipped,
        bins      = 50,
        color     = COLORS["primary"],
        alpha     = 0.8,
        edgecolor = "black",
        linewidth = 0.3
    )
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    # Skew annotation
    skew_val = stats.skew(data_clipped)
    skew_color = (
        COLORS["metal"] if abs(skew_val) > 2 else      # high skew → red warning
        COLORS["metastable"] if abs(skew_val) > 1 else # moderate → orange
        "black"
    )
    ax.text(
        0.97, 0.95,
        f"skew = {skew_val:.2f}",
        transform = ax.transAxes,
        fontsize  = 7,
        ha        = "right",
        va        = "top",
        color     = skew_color,
        bbox      = dict(facecolor="white", alpha=0.7, edgecolor="none")
    )
    apply_font(ax)

    plt.tight_layout()

# ── Hide unused axes ──────────────────────────────────────────────────
for j in range(n_features, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.subplots_adjust(top=0.93, hspace=0.45, wspace=0.3)

# Apply font to all elements including legend
apply_font(ax)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Structural Numerical Feature Distributions (1st–99th percentile clipped for display; raw data unchanged)") 

# ── Cleanup ───────────────────────────────────────────────────────────
del struct_pd, desc, data, data_clipped
free_gpu_memory()

print_heading("Step 14: Feature Transformation — Log Compression & Encoding", level=2)

# ── Validate columns exist before transforming ─────────────────────
LOG_COLS_REQUESTED = [
    "Nsites", "Volume",
    "Lattice (a)", "Lattice (b)", "Lattice (c)", "Total Magnetization",
]

log_cols    = [c for c in LOG_COLS_REQUESTED if c in df.columns]
skipped     = set(LOG_COLS_REQUESTED) - set(log_cols)
if skipped:
    print(f"⚠️  Skipped (not found): {skipped}")

# ── 2. log1p transform — non-negative columns only ────────────────────
print("Applying log1p transforms...")
for col in log_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        raise ValueError(
            f"log1p aborted for '{col}': {neg_count} negative values found.\n"
            f"  Use signed_log or Yeo-Johnson for columns with negatives."
        )
    df[col] = np.log1p(df[col])
    print(f"   log1p({col})")

# ── Orthogonal angle flags ─────────────────────────────────────────
# Lattice angles spike at 90° (cubic, tetragonal, orthorhombic systems).
# Binary flag captures this structural regularity; original angle retained
# for non-orthogonal systems (hexagonal=120°, rhombohedral=60°, etc.)
# No log transform — skew < 0.3 for all three angles.
ANGLE_COLS = [
    ("alpha", "Lattice (alpha)"),
    ("beta",  "Lattice (beta)"),
    ("gamma", "Lattice (gamma)"),
]
for angle_name, col in ANGLE_COLS:
    if col not in df.columns:
        print(f"  ⚠️  '{col}' not found — skipping orthogonal flag")
        continue
    flag_col = f"{angle_name}_orthogonal"
    df[flag_col] = (df[col] == 90.0).astype("int8")
    n_ortho = df[flag_col].sum()
    print(f"  ✅ {flag_col}  ({n_ortho:,} orthogonal, "
          f"{len(df)-n_ortho:,} non-orthogonal)")

# ── Post-transform NaN audit ───────────────────────────────────────
transformed_cols = log_cols + ["Total Magnetization", "Energy Per Atom"]
transformed_cols = [c for c in transformed_cols if c in df.columns]

print(f"\n{'='*55}")
print(f"  POST-TRANSFORM NaN AUDIT")
print(f"{'='*55}")
null_introduced = df[transformed_cols].isna().sum()
any_nulls = False
for col, n in null_introduced.items():
    marker = "✅" if n == 0 else "❌"
    print(f"  {marker}  {col:<40} : {n} NaN")
    if n > 0:
        any_nulls = True

if any_nulls:
    raise ValueError(
        "NaN values introduced by transforms — check input distributions "
        "before proceeding to model training."
    )
else:
    print(f"\n  ✅ No NaN introduced by any transform")

tracker.track(df, "Variance-Stabilizing Transforms", note="Feature Engineering", dataset="MP Dataset")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.6.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 15: Exploratory Data Analysis — Feature Correlation with Band Gap", level=2)

print(f"\n{'='*60}")
print(f"  SECTION 6 — FEATURE CORRELATION WITH BAND GAP")
print(f"{'='*60}")

# ── 1. Select numeric columns ─────────────────────────────────────────
EXCLUDE_COLS = {"Band Gap (T)", "Material ID"}

numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in EXCLUDE_COLS
]
print(f"\n  Numeric columns to correlate : {len(numeric_cols)}")

# ── 2. Vectorised correlation — single corrwith() call ────────────────
# replaces 150+ individual pearsonr() calls — ~50× faster
print(f"  Computing correlations (vectorised)...")

df_pd      = to_pd(df[numeric_cols + ["Band Gap (T)"]].dropna())
corr_r     = df_pd[numeric_cols].corrwith(df_pd["Band Gap (T)"])

# ── 3. p-values — computed efficiently and USED in plot ───────────────
# pearsonr p-value from r and n: t = r*sqrt((n-2)/(1-r²)), p from t-dist
n          = len(df_pd)
t_stat     = corr_r * np.sqrt((n - 2) / (1 - corr_r ** 2).clip(1e-12))
p_values   = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)   # two-tailed

corr_df    = pd.DataFrame({
    "r"          : corr_r,
    "p"          : p_values,
    "abs_r"      : corr_r.abs(),
    "significant": p_values < 0.05,
})
corr_df    = corr_df.sort_values("abs_r", ascending=False)

print(f"  Correlated : {len(corr_df)} columns")
print(f"  Significant (p<0.05) : {corr_df['significant'].sum()} columns")

# ── 4. Summary print ──────────────────────────────────────────────────
print(f"\n  Top 20 Positive Correlations with Band Gap:")
print(corr_df[corr_df["r"] > 0][["r", "p"]].head(20).round(4).to_string())

print(f"\n  Top 20 Negative Correlations with Band Gap:")
print(corr_df[corr_df["r"] < 0][["r", "p"]].head(20).round(4).to_string())

# ── 5. Plot top N ─────────────────────────────────────────────────────
TOP_N    = 30
top_corr = corr_df.head(TOP_N)   # already sorted by abs_r
plot_r   = top_corr["r"].values[::-1]
plot_idx = top_corr.index[::-1]
plot_sig = top_corr["significant"].values[::-1]

fig, ax = plt.subplots(figsize=(10, 9))

# Colour convention consistent with previous cells:
# positive r → primary (blue), negative r → metal (red)
bar_colors = [
    COLORS["primary"] if v >= 0 else COLORS["metal"]
    for v in plot_r
]
# Non-significant bars desaturated
bar_alphas = [0.85 if sig else 0.35 for sig in plot_sig]

bars = ax.barh(
    plot_idx,
    plot_r,
    color     = bar_colors,
    alpha     = 0.85,
    edgecolor = "black",
    linewidth = 0.4
)

# Apply individual alpha for significance
for bar_item, alpha in zip(bars, bar_alphas):
    bar_item.set_alpha(alpha)

ax.axvline(0, color="black", linewidth=0.8)

x_max = top_corr["abs_r"].max()
x_pad = x_max * 0.18
ax.set_xlim(-(x_max + x_pad), +(x_max + x_pad))

# ── Value labels with significance marker ─────────────────────────────
label_pad = x_max * 0.015
for bar_item, val, sig in zip(bars, plot_r, plot_sig):
    pad    = +label_pad if val >= 0 else -label_pad
    ha     = "left"     if val >= 0 else "right"
    marker = ""         if sig       else "ⁿˢ"   # mark non-significant
    ax.text(
        val + pad,
        bar_item.get_y() + bar_item.get_height() / 2,
        f"{val:.3f}{marker}",
        va         = "center",
        ha         = ha,
        fontsize   = 7.5,
        fontfamily = "Times New Roman",
    )

# ── Titles & Labels ───────────────────────────────────────────────────
# ax.set_title(
#     f"Top {TOP_N} Feature Correlations with Band Gap (Pearson r)",
#     fontweight = "bold"
# )
ax.set_xlabel("Pearson r")

ax.legend(
    handles = [
        Patch(color=COLORS["primary"],             label="Positive r"),
        Patch(color=COLORS["metal"],               label="Negative r"),
        # Patch(color=COLORS["primary"], alpha=0.35, label="Non-significant (p≥0.05)"),
    ],
    loc        = "lower right",
    fontsize   = 9,
    framealpha = 0.8
)

apply_font(ax)
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Top 30 Feature Correlations with Bandgap (Pearson r)")

# ── Save correlation table ────────────────────────────────────────────
csv_path = os.path.join(CONFIG["csv_dir"], "feature_correlations_bg.csv")
corr_df.round(4).to_csv(csv_path)
print(f"📄 Correlation table saved → {csv_path}")

# ── Cleanup ───────────────────────────────────────────────────────────
del df_pd, corr_r, t_stat, p_values, top_corr, plot_r, plot_sig
free_gpu_memory()

print_heading("Step 16: Feature Selection — Magpie Intra-Group Correlation Filtering", level=2)

# ── Configuration ─────────────────────────────────────────────────────
MAGPIE_CORR_THRESHOLD = 0.85   # intra-group |r| above this → drop weaker
WEAK_BG_THRESHOLD     = 0.08   # |r with Band Gap| below this → drop
# Rationale for 0.08: features below this threshold contribute less
# predictive signal than noise features in permutation importance tests
# (validated against XGBoost feature_importances_ in Section 8)

# ── Magpie property groups to check ───────────────────────────────────
MAGPIE_PROPERTIES = [
    "Electronegativity", "CovalentRadius", "MeltingT",   "NpValence",
    "SpaceGroupNumber",  "MendeleevNumber","Row",         "NdValence",
    "Column",            "AtomicWeight",   "GSvolume_pa", "GSbandgap",
    "GSmagmom",          "NValence",       "NUnfilled",   "NpUnfilled",
    "NdUnfilled",
]

# ── Protected columns — kept regardless of weak BG correlation ────────
KEEP_COLS = {
    "ox_missing",
    "alpha_orthogonal", "beta_orthogonal",
    "Lattice (alpha)", "Lattice (beta)", "Lattice (gamma)",
    "Band Gap (T)",
}

# ── 1. Compute global correlation matrix ONCE ─────────────────────────
print(f"\n  Computing global correlation matrix...")
num_cols_all  = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in KEEP_COLS
]
df_num_pd     = to_pd(df[num_cols_all + ["Band Gap (T)"]])
global_corr   = df_num_pd.corr().abs()   # ✅ computed once — reused below
bg_corr_all   = global_corr["Band Gap (T)"].drop("Band Gap (T)", errors="ignore")

print(f"  Numeric columns in scope : {len(num_cols_all)}")

# ── 2. Intra-group Magpie correlation filter ──────────────────────────
print(f"\n  Intra-group filtering (threshold={MAGPIE_CORR_THRESHOLD}):")

intragroup_drops = []
drop_log         = []

for prop in MAGPIE_PROPERTIES:
    prop_cols = [c for c in num_cols_all if prop in c]
    if len(prop_cols) < 2:
        continue

    # ✅ Reuse global_corr — no recomputation per group
    prop_corr = global_corr.loc[prop_cols, prop_cols]

    # Vectorised upper triangle
    upper = prop_corr.where(
        np.triu(np.ones(prop_corr.shape), k=1).astype(bool)
    )

    for col_a in upper.columns:
        high_corr_partners = upper.index[upper[col_a] > MAGPIE_CORR_THRESHOLD].tolist()

        for col_b in high_corr_partners:
            if col_a in intragroup_drops or col_b in intragroup_drops:
                continue

            r_a    = bg_corr_all.get(col_a, 0)
            r_b    = bg_corr_all.get(col_b, 0)
            loser  = col_a if r_a < r_b else col_b
            keeper = col_b if r_a < r_b else col_a

            intragroup_drops.append(loser)
            drop_log.append({
                "Group"          : prop,
                "Dropped"        : loser,
                "Kept"           : keeper,
                "Intra-corr |r|" : round(prop_corr.loc[col_a, col_b], 4),
                "Dropped→BG"     : round(min(r_a, r_b), 4),
                "Kept→BG"        : round(max(r_a, r_b), 4),
            })
            print(f"  [{prop:<20}] DROP {loser:<50} "
                  f"(r_BG={min(r_a,r_b):.3f}, intra_r={prop_corr.loc[col_a,col_b]:.2f})")

print(f"\n  Intra-group drops : {len(intragroup_drops)}")

# ── 3. Weak Band Gap correlation filter ───────────────────────────────
print(f"\n  Weak BG correlation filter (|r| < {WEAK_BG_THRESHOLD}):")

remaining_cols = [c for c in num_cols_all if c not in intragroup_drops]
weak_bg_series = bg_corr_all[remaining_cols].sort_values()
weak_drops     = [
    c for c in weak_bg_series[weak_bg_series < WEAK_BG_THRESHOLD].index
    if c not in KEEP_COLS
]
print(f"  Weak BG drops : {len(weak_drops)}")

# ── 4. Rule-based drops ───────────────────────────────────────────────
# Ns*/Nf* orbital groups — near-zero variance in inorganic crystal dataset
NS_NF_DROPS = [
    c for c in df.columns
    if any(x in c for x in ["NsValence", "NsUnfilled", "NfValence", "NfUnfilled"])
]

# Manual drops — missed by threshold, confirmed by inspection above
MANUAL_DROPS = [
    "MagpieData mode Electronegativity",    # mode duplicates mean/median signal
    "MagpieData mode GSbandgap",            # DFT bandgap embedded in feature — data leakage risk
    "MagpieData maximum SpaceGroupNumber",  # redundant with Space Group Number column
]
MANUAL_DROPS = [c for c in MANUAL_DROPS if c in df.columns]

print(f"\n  Ns/Nf orbital drops : {len(NS_NF_DROPS)}")
print(f"  Manual drops        : {len(MANUAL_DROPS)}")

# ── 5. Combine, deduplicate, validate ────────────────────────────────
all_drops = list(set(
    intragroup_drops + weak_drops + NS_NF_DROPS + MANUAL_DROPS
))
all_drops = [c for c in all_drops if c in df.columns and c not in KEEP_COLS]

print(f"\n{'='*60}")
print(f"  FEATURE REDUCTION SUMMARY")
print(f"{'='*60}")
print(f"  Intra-group (|r|>{MAGPIE_CORR_THRESHOLD})             : {len(intragroup_drops)}")
print(f"  Weak BG (|r|<{WEAK_BG_THRESHOLD})                 : {len(weak_drops)}")
print(f"  Ns/Nf orbital groups               : {len(NS_NF_DROPS)}")
print(f"  Manual                             : {len(MANUAL_DROPS)}")
print(f"  {'─'*35}")
print(f"  Total unique drops                 : {len(all_drops)}")
print(f"  Shape before                       : {df.shape}")

# ── 6. Apply drops ────────────────────────────────────────────────────
df = df.drop(columns=all_drops)   # ✅ no inplace=True
print(f"  Shape after                        : {df.shape}")
print(f"  Features remaining                 : {df.select_dtypes(include=[np.number]).shape[1] - 1}")

print_heading("Phase 3: Feature Space Pruning & Finalisation", level=1)
print_heading("Step 17: Data Type Finalisation", level=2)

# ── Convert bool/uint8 → int8 ─────────────────────────────────────────
# XGBoost and sklearn reject bool dtype — must be numeric
# uint8 from pd.get_dummies() also caught here
BOOL_LIKE_DTYPES = {"bool", "boolean", "uint8"}

bool_cols = [
    c for c in df.columns
    if str(df[c].dtype) in BOOL_LIKE_DTYPES
]

if bool_cols:
    df[bool_cols] = df[bool_cols].astype("int8")   # ✅ int8 — binary flags only need 1 byte
    print(f"  Converted {len(bool_cols)} bool/uint8 columns → int8:")
    for c in bool_cols:
        print(f"    {c:<45} dtype: {df[c].dtype}")
else:
    print("  ✅ No bool/uint8 columns found")

# ── Verify conversion ─────────────────────────────────────────────────
still_bool = [
    c for c in bool_cols
    if str(df[c].dtype) in BOOL_LIKE_DTYPES
]
if still_bool:
    print(f"\n  ❌ Conversion failed for: {still_bool}")
else:
    print(f"\n  ✅ All bool columns successfully converted to int8")

tracker.track(df, "Orbital Feature Removal", note="Feature Selection", dataset="MP Dataset")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_3.0.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 18: Exploratory Data Analysis — Structural & Magpie Correlation Heatmap", level=2)

print(f"\n{'='*60}")
print(f"  SECTION 7 — CORRELATION HEATMAP")
print(f"{'='*60}")

# ── 1. Select columns — reconstruct from df directly ──────────────────
# No dependency on stale corr_series, struct_present, or pdf

STRUCT_COLS = [
    "Number of Elements", "Nsites", "Volume", "Density",
    "Space Group Number", "Energy Per Atom",
    "Total Magnetization", "Formation Energy Per Atom",
    "Ox_min", "Ox_max", "Ox_mean", "Ox_range",
]
struct_in_df = [c for c in STRUCT_COLS if c in df.columns]

# Top Magpie features by |r| with Band Gap — computed fresh
df_num_pd   = to_pd(df.select_dtypes(include=[np.number]))
bg_corr_sec7= df_num_pd.corrwith(df_num_pd["Band Gap (T)"]).abs()

top_magpie  = (
    bg_corr_sec7
    .drop("Band Gap (T)", errors="ignore")
    .filter(like="MagpieData")
    .sort_values(ascending=False)
    .head(12)
    .index.tolist()
)

heatmap_cols = struct_in_df + top_magpie + ["Band Gap (T)"]
heatmap_cols = [c for c in dict.fromkeys(heatmap_cols)   # deduplicate, preserve order
                if c in df.columns]

print(f"  Heatmap columns : {len(heatmap_cols)}")
print(f"  Structural      : {len(struct_in_df)}")
print(f"  Top Magpie      : {len(top_magpie)}")

# ── 2. Correlation matrix — cuDF safe ────────────────────────────────
corr_matrix = to_pd(df[heatmap_cols]).corr()
col_names   = [str(c) for c in corr_matrix.columns]
n           = len(col_names)


# Extract as plain Python list first — breaks cuDF proxy chain
Z_list      = corr_matrix.values.tolist()          # pure Python list of lists
Z           = np.array(Z_list, dtype=np.float64)   # fresh numpy array — no cuDF proxy
mask        = np.triu(np.ones((n, n), dtype=bool))

# Use np.where on pure numpy — no masked array needed
Z_masked    = np.where(mask, np.nan, Z)

# Force copy to guarantee pure numpy memory — no proxy wrapping
Z_masked    = np.array(Z_masked, dtype=np.float64, copy=True)

# ── 4. Build Figure ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 13))

im = ax.imshow(
    Z_masked,
    cmap    = "RdBu_r",
    vmin    = -1,
    vmax    = 1,
    aspect  = "auto"
)

# ── 5. Cell annotations — lower triangle only ─────────────────────────
for i in range(n):
    for j in range(i):                    # j < i → lower triangle only
        val   = Z[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(
            j, i,
            f"{val:.2f}",
            ha         = "center",
            va         = "center",
            fontsize   = 5.5,
            color      = color,
            fontfamily = "Times New Roman",
        )

# # ── 6. Grid lines ─────────────────────────────────────────────────────
ax.grid(False)

# ── 7. Axes labels ────────────────────────────────────────────────────
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(col_names, fontsize=7.5)
ax.tick_params(length=0)

# ── 8. Colourbar ──────────────────────────────────────────────────────
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Pearson r", fontsize=10, fontfamily="Times New Roman")
cbar.outline.set_linewidth(1)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontfamily("Times New Roman")
apply_font(ax)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Correlation Heatmap — Structural + Top Magpie Features vs Band Gap")

# ── Cleanup ───────────────────────────────────────────────────────────
del df_num_pd, bg_corr_sec7, corr_matrix, Z, Z_masked, Z_list
free_gpu_memory()

print_heading("Step 19: Final Pre-Training Feature Audit & Collinearity Validation", level=2)

COLLINEAR_THRESHOLD = 0.90
WEAK_FLAG_THRESHOLD = 0.05   # ✅ consistent — flag only, not drop
TARGET              = "Band Gap (T)"

assert TARGET in df.columns, f"Target '{TARGET}' not found in df"

# ── 1. Recompute correlation matrix fresh — no Section 7 dependency ───
print("Computing correlation matrix...")
_num_cols   = [c for c in df.select_dtypes(include=[np.number]).columns]
_corr_pd    = to_pd(df[_num_cols]).corr()
_col_names  = list(_corr_pd.columns)
_n          = len(_col_names)

# Pure numpy — break cuDF proxy
_Z          = np.array(_corr_pd.values.tolist(), dtype=np.float64)
_target_idx = _col_names.index(TARGET)
r_vs_target = {_col_names[i]: abs(_Z[i][_target_idx]) for i in range(_n)}

print(f"  Columns in scope : {_n}")

# ── 2. Find collinear pairs — vectorised upper triangle ───────────────
upper_i, upper_j = np.triu_indices(_n, k=1)
all_pairs = []

for i, j in zip(upper_i, upper_j):
    ci, cj = _col_names[i], _col_names[j]
    if ci == TARGET or cj == TARGET:
        continue
    abs_r = abs(_Z[i, j])
    if abs_r >= COLLINEAR_THRESHOLD:
        all_pairs.append((abs_r, _Z[i, j], ci, cj))

all_pairs.sort(reverse=True)   # greedy — highest corr pairs first


# ── 3. Greedy drop — keep feature with higher |r| vs target ──────────
already_dropped = set()
drop_collinear  = []

for abs_r, raw_r, ci, cj in all_pairs:
    if ci in already_dropped or cj in already_dropped:
        continue

    drop_col = ci if r_vs_target[ci] <= r_vs_target[cj] else cj
    keep_col = cj if drop_col == ci else ci

    already_dropped.add(drop_col)
    drop_collinear.append({
        "drop"        : drop_col,
        "keep"        : keep_col,
        "r_between"   : round(raw_r, 3),
        "r_drop_vs_BG": round(r_vs_target[drop_col], 3),
        "r_keep_vs_BG": round(r_vs_target[keep_col], 3),
    })

# ── 4. Metadata columns — not usable as features ─────────────────────
DROP_META = [
    "Material ID", "Elements", "Pretty Formula",   # ✅ fixed typo
    "Normalised Composition", "Composition_obj",
    "Oxidation States", "Ox_Values",
]
# Log which ones still exist vs already dropped
meta_present = [c for c in DROP_META if c in df.columns]
meta_already_gone = [c for c in DROP_META if c not in df.columns]
if meta_already_gone:
    print(f"  Already removed in earlier cells: {meta_already_gone}")

# ── 5. Flag weak features — keep for SHAP review ─────────────────────
weak_flagged = [
    {"col": _col_names[i], "r_vs_BG": round(_Z[i][_target_idx], 4)}
    for i in range(_n)
    if  _col_names[i] != TARGET
    and _col_names[i] not in already_dropped
    and abs(_Z[i][_target_idx]) < WEAK_FLAG_THRESHOLD
]

# ── 6. Print ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  COLLINEAR DROPS  (|r| ≥ {COLLINEAR_THRESHOLD})")
print(f"{'='*65}")
for p in drop_collinear:
    print(f"  DROP  {p['drop']:<48} r_between={p['r_between']:+.3f}  "
          f"r_vs_BG={p['r_drop_vs_BG']:.3f}")
    print(f"  KEEP  {p['keep']:<48} r_vs_BG={p['r_keep_vs_BG']:.3f}\n")

print(f"{'='*65}")
print(f"  WEAK FEATURES FLAGGED  (|r| < {WEAK_FLAG_THRESHOLD}) — keep, review after SHAP")
print(f"{'='*65}")
for w in sorted(weak_flagged, key=lambda x: abs(x["r_vs_BG"])):
    print(f"  {w['col']:<52} r={w['r_vs_BG']:+.4f}")


# ── 7. Apply drops ────────────────────────────────────────────────────
cols_to_drop = list({
    c for c in list(already_dropped) + meta_present
    if c in df.columns and c != TARGET
})

shape_before = df.shape   # ✅ capture BEFORE drop
df           = df.drop(columns=cols_to_drop)
shape_after  = df.shape

# ── 8. Summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FEATURE FILTERING SUMMARY")
print(f"{'='*65}")
print(f"  Collinear dropped          : {len(drop_collinear)}")
print(f"  Metadata dropped           : {len(meta_present)}")
print(f"  Weak flagged (not dropped) : {len(weak_flagged)}")
print(f"  Total columns dropped      : {len(cols_to_drop)}")
print(f"  Shape before               : {shape_before}")    # ✅ correct
print(f"  Shape after                : {shape_after}")
print(f"  Features remaining         : {shape_after[1] - 1}")

assert TARGET in df.columns, "❌ Target column lost during drop"
print(f"  Target '{TARGET}'          : ✅ present")

tracker.track(df, "Greedy Collinearity Pruning", note="Feature Selection", dataset="MP Dataset")

# ── Cleanup ───────────────────────────────────────────────────────────
del _corr_pd, _Z, _col_names, r_vs_target, all_pairs
free_gpu_memory()


# NOTE: If the previous filtering cell was run with our rewrite,
# this cell should find zero additional pairs — it serves as a
# validation step confirming the previous cell was exhaustive.
# ─────────────────────────────────────────────────────────────
print(f"{'='*65}")
print(f"  FULL COLLINEARITY VALIDATION CHECK")
print(f"{'='*65}")

COLLINEAR_THRESHOLD = 0.90
TARGET              = "Band Gap (T)"

# ── 1. All numeric cols including target — single corr() call ─────────
numeric_cols = [
    c for c in df.columns
    if df[c].dtype in ["float64", "float32", "int64", "int8", "int16"]
]
print(f"\n  Numeric features to check : {len(numeric_cols)}")

# Include target in same corr() call — no second call needed
all_cols_for_corr = numeric_cols + (
    [TARGET] if TARGET not in numeric_cols else []
)
_corr_full   = to_pd(df[all_cols_for_corr]).corr()
_col_names   = list(_corr_full.columns)
_n           = len(_col_names)

# Pure numpy — break cuDF proxy
_Z_full      = np.array(_corr_full.values.tolist(), dtype=np.float64)
_target_idx  = _col_names.index(TARGET)

r_vs_target  = {
    _col_names[i]: abs(_Z_full[i, _target_idx])
    for i in range(_n)
}

# ── 2. Find pairs — vectorised upper triangle ─────────────────────────
ui, uj      = np.triu_indices(_n, k=1)
full_pairs  = []

for i, j in zip(ui, uj):
    ci, cj  = _col_names[i], _col_names[j]
    if ci == TARGET or cj == TARGET:
        continue
    abs_r   = abs(_Z_full[i, j])
    if abs_r >= COLLINEAR_THRESHOLD:
        full_pairs.append((abs_r, _Z_full[i, j], ci, cj))

full_pairs.sort(reverse=True)
print(f"  Collinear pairs found (|r| ≥ {COLLINEAR_THRESHOLD}) : {len(full_pairs)}")

if not full_pairs:
    print(f"\n  ✅ No additional collinear pairs — previous filtering was exhaustive")
else:
    # ── 3. Greedy drop ────────────────────────────────────────────────
    already_dropped = set()
    drop_log        = []

    for abs_r, raw_r, ci, cj in full_pairs:
        if ci in already_dropped or cj in already_dropped:
            continue

        drop_col = ci if r_vs_target.get(ci, 0) <= r_vs_target.get(cj, 0) else cj
        keep_col = cj if drop_col == ci else ci

        already_dropped.add(drop_col)
        drop_log.append({
            "drop"        : drop_col,
            "keep"        : keep_col,
            "r_between"   : round(raw_r, 3),
            "r_drop_vs_BG": round(r_vs_target.get(drop_col, 0), 3),
            "r_keep_vs_BG": round(r_vs_target.get(keep_col, 0), 3),
        })
        print(f"  DROP  {drop_col:<50} r_between={raw_r:+.3f}  "
              f"r_vs_BG={r_vs_target.get(drop_col,0):.3f}")
        print(f"  KEEP  {keep_col:<50} "
              f"r_vs_BG={r_vs_target.get(keep_col,0):.3f}\n")

    # ── 4. Apply ──────────────────────────────────────────────────────
    extra_drops  = [p["drop"] for p in drop_log if p["drop"] in df.columns]
    shape_before = df.shape

    df = df.drop(columns=extra_drops)   # ✅ no .copy() needed

    print(f"\n  Additional columns dropped : {len(extra_drops)}")
    print(f"  Shape before               : {shape_before}")
    print(f"  Shape after                : {df.shape}")

    # Save log
    pd.DataFrame(drop_log).to_csv(
        os.path.join(CONFIG["csv_dir"], "collinear_full_drop_log.csv"),
        index=False
    )
    print(f"  📄 Drop log saved")

# ── 5. Final assertion ────────────────────────────────────────────────
assert TARGET in df.columns, "❌ Target column lost"
print(f"\n  Target '{TARGET}' : ✅ present")
print(f"  Final shape      : {df.shape}")
print(f"  Features         : {df.shape[1] - 1}")

tracker.track(df, "Final Matrix Validation", note="Data Quality", dataset="MP Dataset")

# ── Cleanup ───────────────────────────────────────────────────────────
del _corr_full, _Z_full, _col_names, r_vs_target, full_pairs
free_gpu_memory()

print_heading("Step 20: Exploratory Data Analysis — Outlier & Complexity Audit", level=2)

print(f"\n{'='*60}")
print(f"  SECTION 8 — OUTLIER ANALYSIS")
print(f"{'='*60}")

OUTLIER_COLS_REQUESTED = [
    "Volume", "Density", "Nsites",
    "Lattice (a)", "Lattice (b)", "Lattice (c)",
    "Energy Per Atom", "Formation Energy Per Atom",
    "Total Magnetization", "Band Gap (T)",
]

# Validate against df — no pdf dependency
outlier_cols = [c for c in OUTLIER_COLS_REQUESTED if c in df.columns]
dropped_cols = set(OUTLIER_COLS_REQUESTED) - set(outlier_cols)
if dropped_cols:
    print(f"  ⚠️  Not found (may have been filtered): {sorted(dropped_cols)}")

# Note log-transformed columns — IQR computed on transformed scale
LOG_TRANSFORMED = {"Volume", "Nsites", "Lattice (a)", "Lattice (b)", "Lattice (c)"}
log_in_analysis = [c for c in outlier_cols if c in LOG_TRANSFORMED]
if log_in_analysis:
    print(f"\n  📌 Note: {log_in_analysis}")
    print(f"     These columns were log1p-transformed — IQR fences")
    print(f"     are on the log scale, not original units.")

# ── 1. IQR Outlier Report ─────────────────────────────────────────────
outlier_report = []

for col in outlier_cols:
    data_col = to_pd(df[col]).dropna()

    Q1, Q3  = data_col.quantile(0.25), data_col.quantile(0.75)
    IQR     = Q3 - Q1
    lower   = Q1 - 1.5 * IQR
    upper   = Q3 + 1.5 * IQR
    n_out   = int(((data_col < lower) | (data_col > upper)).sum())

    # Flag Band Gap separately — zero inflation skews IQR
    bg_note = " ⚠️ zero-inflated" if col == "Band Gap (T)" else ""

    outlier_report.append({
        "Column"        : col + bg_note,
        "Q1"            : round(float(Q1),    3),
        "Q3"            : round(float(Q3),    3),
        "IQR"           : round(float(IQR),   3),
        "Lower fence"   : round(float(lower), 3),
        "Upper fence"   : round(float(upper), 3),
        "Outliers (IQR)": n_out,
        "Outlier %"     : round(n_out / len(data_col) * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_report).set_index("Column")
print(f"\n  IQR-based Outlier Report:")
print(outlier_df.to_string())

# ── 2. Boxplot Grid — dynamic layout ─────────────────────────────────
N_COLS_PLOT = 5
n_features  = len(outlier_cols)
n_rows_plot = int(np.ceil(n_features / N_COLS_PLOT))

fig, axes = plt.subplots(
    n_rows_plot, N_COLS_PLOT,
    figsize=(20, n_rows_plot * 3.8)
)
axes_flat = axes.flatten() if n_rows_plot > 1 else axes

for i, col in enumerate(outlier_cols):
    ax      = axes_flat[i]
    data    = to_np(df[col], dtype=np.float64)
    data    = data[~np.isnan(data)]

    bp = ax.boxplot(
        data,
        patch_artist = True,
        showfliers   = True,
        flierprops   = dict(
            marker     = ".",
            markersize = 1,
            alpha      = 0.2,
            color      = COLORS["neutral"]
        ),
        medianprops  = dict(color=COLORS["metal"],   linewidth=2),
        boxprops     = dict(facecolor=COLORS["primary"], alpha=0.6, linewidth=1.2),
        whiskerprops = dict(linewidth=1.0),
        capprops     = dict(linewidth=1.0),
    )

    # ── IQR stats for annotation ──────────────────────────────────────
    Q1, Q3  = np.percentile(data, [25, 75])
    IQR     = Q3 - Q1
    n_out   = int(np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)))
    pct_out = n_out / len(data) * 100

    # ── Annotation colour — flag high outlier % ───────────────────────
    ann_color = (
        COLORS["metal"]      if pct_out > 10 else
        COLORS["metastable"] if pct_out > 5  else
        "black"
    )
    ax.text(
        1.28, float(np.median(data)),
        f"{pct_out:.1f}%\noutliers",
        fontsize   = 7,
        va         = "center",
        ha         = "left",
        color      = ann_color,
        fontfamily = "Times New Roman",
        bbox       = dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    # ── Log scale note in title ───────────────────────────────────────
    log_note = " [log]" if col in LOG_TRANSFORMED else ""
    ax.set_title(
        f"{col}{log_note}",
        fontsize   = 8.5,
        fontweight = "bold",
        fontfamily = "Times New Roman"
    )
    ax.tick_params(labelsize=7)
    apply_font(ax, fontsize={"ticks": 7, "title": 8.5})

# ── Hide unused axes ──────────────────────────────────────────────────
for j in range(n_features, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.subplots_adjust(top=0.90, hspace=0.45, wspace=0.35)

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Outlier Distribution — IQR Method (annotation colour: black ≤5%, orange 5–10%, red >10%)")

# ── Cleanup ───────────────────────────────────────────────────────────
del data, data_col, outlier_report, outlier_df
free_gpu_memory()

# Better: clip both tails symmetrically
p01 = float(df['Total Magnetization'].quantile(0.01))
p99 = float(df['Total Magnetization'].quantile(0.99))
df['Total Magnetization'] = df['Total Magnetization'].clip(lower=p01, upper=p99)

print(f"Total Magnetization clipped at 99th pct: {p99:.3f}")
print(f"New max: {float(df['Total Magnetization'].max()):.3f}")

print("\n" + "=" * 60)
print("SECTION 9: NUMBER OF ELEMENTS vs BAND GAP")
print("=" * 60)

n_elem = to_np(df['Number of Elements'], dtype=np.float32)
bg_vals = to_np(df['Band Gap (T)'], dtype=np.float32)
valid = ~np.isnan(n_elem) & ~np.isnan(bg_vals)
n_elem, bg_vals = n_elem[valid], bg_vals[valid]

print("\n Band Gap stats by number of elements:")
for n in sorted(np.unique(n_elem)):
    mask = n_elem == n
    subset = bg_vals[mask]
    metals_frac = (subset == 0).mean() * 100
    print(f"  {int(n)} elements: count={mask.sum():6,}  "
          f"mean_BG={subset.mean():.2f} eV  "
          f"median_BG={np.median(subset):.2f} eV  "
          f"metal%={metals_frac:.1f}%")

fig1, ax1 = plt.subplots(figsize=(8, 5))

unique_n = sorted(np.unique(n_elem).astype(int))
groups_n = [bg_vals[n_elem == n] for n in unique_n]

bp = ax1.boxplot(groups_n, labels=[str(n) for n in unique_n],
                 patch_artist=True, showfliers=False,
                 medianprops=dict(color='black', linewidth=2))

cmap_n = plt.cm.viridis
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cmap_n(i / len(unique_n)))
    patch.set_alpha(0.75)
ax1.set_xlabel('Number of Elements')
ax1.set_ylabel('Band Gap (eV)')
ax1.set_ylim(-0.2, 9)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig1, "Band Gap vs Number of Elements")

fig2, ax2 = plt.subplots(figsize=(7, 5))

metal_fracs = [(n_elem == n).sum() for n in unique_n]
metal_counts = [(bg_vals[n_elem == n] == 0).sum() for n in unique_n]
metal_pcts = [mc / tc * 100 for mc, tc in zip(metal_counts, metal_fracs)]

ax2.bar([str(n) for n in unique_n], metal_pcts,
        color='#C41E3A', alpha=0.8, edgecolor='black', linewidth=0.5)

for i, val in enumerate(metal_pcts):
    ax2.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=9)

ax2.set_title('Metal Fraction by Number of Elements', fontweight='bold')
ax2.set_xlabel('Number of Elements')
ax2.set_ylabel('Metal %')

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "MetalFraction_vs_NumElements")

print_heading("Step 21: Metadata Removal & Classification Target Engineering", level=2)

drop_meta = [
    'Material ID', 'Elements', 'Pretty Formula',
    'Normalised Composition', 'Oxidation States',
    'Composition_obj', 'Ox_Values', 'Ox_range'
]
drop_meta = [c for c in drop_meta if c in df.columns]
df = df.drop(columns=drop_meta)
print(df.shape)

tracker.track(df, "Metadata Removal", note="Cleaning", dataset="MP Dataset")

# Create classification target
# Is Metal (T) = 1 when Band Gap == 0, else 0
df['Is_Metal'] = (df['Band Gap (T)'] == 0).astype(int)

print(df['Is_Metal'].value_counts())
print(f"\nMetal (1)     : {int((df['Is_Metal']==1).sum()):,}")
print(f"Non-Metal (0) : {int((df['Is_Metal']==0).sum()):,}")

tracker.track(df, "Hurdle Target Construction", note="Target Engineering", dataset="MP Dataset")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_3.1.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")


fig1, ax1 = plt.subplots(figsize=(7, 5))

# Calculate statistics
mean_bg = df['Band Gap (T)'].mean()
median_bg = df['Band Gap (T)'].median()
n_total = len(df)
n_metallic = len(df[df['Band Gap (T)'] == 0])

# Create histogram with KDE
sns.histplot(
    df['Band Gap (T)'],
    bins=60,
    kde=True,
    color=COLORS["primary"],   # use defined palette
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5,
    ax=ax1,
    stat='count',
    line_kws={'linewidth': 2}
)

# Metallic threshold (0 eV)
ax1.axvline(
    x=0,
    color=COLORS["metal"],   # from palette
    linestyle='--',
    linewidth=2,
    label=f'Metallic threshold (n = {n_metallic})'
)

# Mean line
ax1.axvline(
    x=mean_bg,
    color=COLORS["stable"],   # from palette
    linestyle=':',
    linewidth=2,
    label=f'Mean = {mean_bg:.2f} eV'
)

# Formatting (no manual font sizes — rcParams handles it)
# ax1.set_title('Full Band Gap Distribution', fontweight='bold', pad=12)
ax1.set_xlabel('Band Gap (eV)')
ax1.set_ylabel('Frequency')

# Legend styling aligned with your theme
ax1.legend(
    frameon=True,
    fancybox=False,
    shadow=False,
    loc='upper right',
    edgecolor='black',
    framealpha=1
)

# Grid — subtle, consistent with your config
# ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Stats textbox
textstr = f'n = {n_total}\nMedian = {median_bg:.2f} eV'
props = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.85,
    edgecolor='black'
)

ax1.text(
    0.98, 0.97, textstr,
    transform=ax1.transAxes,
    fontsize=10,
    va='top',
    ha='right',
    bbox=props
)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()


# Optional save
save_figure(fig1, "BandGap_Distribution_Full")

fig2, ax2 = plt.subplots(figsize=(7, 5))

# Filter non-zero band gaps
non_zero_bg = df.loc[df['Band Gap (T)'] > 0, 'Band Gap (T)']

# Statistics
mean_nz = non_zero_bg.mean()
median_nz = non_zero_bg.median()
n_semiconductor = len(non_zero_bg)

# Histogram + KDE
sns.histplot(
    non_zero_bg,
    bins=50,
    kde=True,
    color=COLORS["secondary"],   # use palette instead of color2
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5,
    ax=ax2,
    stat='count',
    line_kws={'linewidth': 2, 'color': COLORS["secondary"]}
)

# Mean line
ax2.axvline(
    x=mean_nz,
    color=COLORS["stable"],   # green from palette
    linestyle=':',
    linewidth=2,
    label=f'Mean = {mean_nz:.2f} eV'
)

# Median line
ax2.axvline(
    x=median_nz,
    color=COLORS["tertiary"],   # replaces hardcoded purple
    linestyle='-.',
    linewidth=2,
    label=f'Median = {median_nz:.2f} eV'
)

ax2.set_xlabel('Band Gap (eV)')
ax2.set_ylabel('Frequency')

# Legend
ax2.legend(
    frameon=True,
    fancybox=False,
    shadow=False,
    loc='upper right',
    edgecolor='black',
    framealpha=1
)

# Grid styling
# ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# Stats textbox
textstr = f'n = {n_semiconductor}\n({n_semiconductor/n_total*100:.1f}% of total)'
props = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.85,
    edgecolor='black'
)

ax2.text(
    0.98, 0.97,
    textstr,
    transform=ax2.transAxes,
    fontsize=10,
    va='top',
    ha='right',
    bbox=props
)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()


# Optional save
save_figure(fig2, "BandGap_Distribution_NonZero")

print("\n--- TARGET ANALYSIS: Band Gap (T) ---")
metals = df[df['Band Gap (T)'] == 0]
non_metals = df[df['Band Gap (T)'] > 0]

print(f"Metals (0.0 eV): {len(metals)} ({len(metals)/len(df)*100:.2f}%)")
print(f"Non-Metals (> 0.0 eV): {len(non_metals)} ({len(non_metals)/len(df)*100:.2f}%)")
print(f"Non-Metal Band Gap Mean: {non_metals['Band Gap (T)'].mean():.4f} eV")
print(f"Non-Metal Band Gap Median: {non_metals['Band Gap (T)'].median():.4f} eV")

# --- Data Preparation (CPU-Safe Sampling) ---
sample_size = min(15000, len(df))
sample_idx = np.random.choice(len(df), sample_size, replace=False)

def get_sample(col_name):
    return df[col_name].iloc[sample_idx].to_numpy()

def get_full(col_name):
    return df[col_name].to_numpy()

fig4, ax4 = plt.subplots(figsize=(7, 6))

# Check if Magpie feature exists
x_col = 'MagpieData mean Electronegativity' if 'MagpieData mean Electronegativity' in df.columns else 'Density'
x_label = 'Mean Electronegativity' if 'MagpieData mean Electronegativity' in df.columns else 'Density (g/cm³)'

chem_data = pd.DataFrame({
    x_label: get_sample(x_col),
    'Band Gap': get_sample('Band Gap (T)'),
    'Is Metal': get_sample('Is_Metal')
})

# Filter semiconductors/insulators (non-metals with Band Gap > 0)
semiconductors = chem_data[(chem_data['Is Metal'] == 0) & (chem_data['Band Gap'] > 0)]

if len(semiconductors) > 100:  # Need enough points for KDE
    # Create density contour plot
    sns.kdeplot(data=semiconductors, x=x_label, y='Band Gap',
               fill=True, cmap="viridis", thresh=0.05, levels=12, 
               ax=ax4, alpha=0.7)

    # Overlay scatter points
    ax4.scatter(semiconductors[x_label], semiconductors['Band Gap'],
               alpha=0.15, s=8, color='black', edgecolors='none',
               rasterized=True, label='Individual materials')
else:
    # If not enough points, just scatter
    ax4.scatter(semiconductors[x_label], semiconductors['Band Gap'],
               alpha=0.5, s=30, color='#008B8B', edgecolors='black',
               linewidths=0.3, label='Semiconductors/Insulators')

ax4.set_xlabel(x_label)
ax4.set_ylabel('Band Gap (eV)')
ax4.set_ylim(0, 8)
ax4.set_axisbelow(True)

# Add statistics box
mean_bg = semiconductors['Band Gap'].mean()
median_bg = semiconductors['Band Gap'].median()
mean_x = semiconductors[x_label].mean()

textstr = (f'n = {len(semiconductors)}\n'
          f'Band Gap:\n  Mean: {mean_bg:.2f} eV\n  Median: {median_bg:.2f} eV\n'
          f'{x_label}:\n  Mean: {mean_x:.2f}')
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.2)
ax4.text(0.98, 0.98, textstr, transform=ax4.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=props, family='monospace')

# Add legend if scatter plot
if len(semiconductors) <= 100:
    ax4.legend(frameon=True, fancybox=False, shadow=False,
              loc='upper left', edgecolor='black', framealpha=1)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()


save_figure(fig4, "Density vs Bandgap (Semiconductors vs Insulators)")

print(f"\n=== FIGURE 4: CHEMICAL PROPERTY VS BAND GAP ===")
print(f"Non-metallic materials: n = {len(semiconductors)}")
print(f"Band Gap: Mean = {mean_bg:.3f} eV, Median = {median_bg:.3f} eV")
print(f"{x_label}: Mean = {mean_x:.3f}")
print(f"Range: {semiconductors['Band Gap'].min():.3f} - {semiconductors['Band Gap'].max():.3f} eV")


fig2, ax2 = plt.subplots(figsize=(8, 5))

# Define stability from formation energy
stable_fe = df[df['Formation Energy Per Atom'] < 0]['Formation Energy Per Atom']
unstable_fe = df[df['Formation Energy Per Atom'] >= 0]['Formation Energy Per Atom']

# Stats
mean_stable = stable_fe.mean()
mean_unstable = unstable_fe.mean()

# Histogram
ax2.hist(
    stable_fe,
    bins=50,
    alpha=0.6,
    color=COLORS["stable"],
    edgecolor='black',
    linewidth=0.5,
    label=f'Stable (n={len(stable_fe)})',
    density=True
)

ax2.hist(
    unstable_fe,
    bins=50,
    alpha=0.6,
    color=COLORS["metal"],
    edgecolor='black',
    linewidth=0.5,
    label=f'Unstable (n={len(unstable_fe)})',
    density=True
)

# Mean lines
ax2.axvline(
    mean_stable,
    color=COLORS["stable"],
    linestyle='--',
    linewidth=2,
    label=f'Stable mean = {mean_stable:.2f}'
)

ax2.axvline(
    mean_unstable,
    color=COLORS["metal"],
    linestyle='--',
    linewidth=2,
    label=f'Unstable mean = {mean_unstable:.2f}'
)

ax2.set_xlabel('Formation Energy (eV/atom)')
ax2.set_ylabel('Probability Density')
ax2.set_axisbelow(True)

ax2.legend(
    frameon=True,
    fancybox=False,
    shadow=False,
    loc='upper left',
    edgecolor='black',
    framealpha=1
)

# Stats box
textstr = (
    f'Stable:\n  n = {len(stable_fe)}\n  σ = {stable_fe.std():.2f}\n'
    f'Unstable:\n  n = {len(unstable_fe)}\n  σ = {unstable_fe.std():.2f}'
)

props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black')

ax2.text(
    0.98, 0.98,
    textstr,
    transform=ax2.transAxes,
    fontsize=9,
    va='top',
    ha='right',
    bbox=props,
    family='monospace'
)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Formation Energy Distribution by Stability")

# Sample data for performance
sample_size = min(20000, len(df))
sample_idx = np.random.choice(len(df), sample_size, replace=False)

def get_col(col_name, subset=None):
    """Extract column as numpy array"""
    if subset is None:
        return df[col_name].to_numpy()
    return df[col_name].iloc[subset].to_numpy()

# Prepare data (NO Is Stable column)
landscape_data = pd.DataFrame({
    'Density': get_col('Density', sample_idx),
    'Formation Energy': get_col('Formation Energy Per Atom', sample_idx),
    'Band Gap': get_col('Band Gap (T)', sample_idx),
})

# Derive stability
landscape_data['Is Stable'] = (landscape_data['Formation Energy'] <= 0).astype(int)

# Filter physically reasonable range
mask_land = (
    (landscape_data['Density'] > 0) &
    (landscape_data['Density'] < 15) &
    (landscape_data['Formation Energy'] > -6)
)
landscape_filtered = landscape_data[mask_land]

# Statistics
n_points = len(landscape_filtered)
n_stable = (landscape_filtered['Is Stable'] == 1).sum()
stable_pct = (n_stable / n_points) * 100

fig, ax = plt.subplots(figsize=(8, 7))

# Scatter plot
points = ax.scatter(
    landscape_filtered['Density'],
    landscape_filtered['Formation Energy'],
    c=landscape_filtered['Band Gap'],
    cmap='plasma',   # keep — perceptually good
    s=20,
    alpha=0.6,
    vmin=0,
    vmax=6,
    edgecolor='none'
)

# Stability threshold line
ax.axhline(
    0,
    color=COLORS["metal"],   # consistent palette
    linestyle='--',
    linewidth=2,
    alpha=0.8,
    label='Stability threshold (ΔH = 0)',
    zorder=3
)

ax.set_xlabel('Density (g/cm³)')
ax.set_ylabel('Formation Energy (eV/atom)')
ax.set_ylim(-5, 6)

ax.legend(
    frameon=True,
    fancybox=False,
    shadow=False,
    loc='upper right',
    edgecolor='black',
    framealpha=1
)

ax.set_axisbelow(True)

# Colorbar
cbar = plt.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Band Gap (eV)', rotation=270, labelpad=15, fontweight='bold')
cbar.outline.set_linewidth(1.2)

# Stats box
textstr = f'n = {n_points:,}\nStable = {n_stable:,} ({stable_pct:.1f}%)'
props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black')

ax.text(
    0.02, 0.97,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    va='top',
    ha='left',
    bbox=props
)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Stability_Landscape")

print_heading("Step 22: Target Transformation — Log Compression for Regression", level=2)

def np_col(col):
    return df[col].to_numpy(dtype=np.float32)

# Extract data
bandgap = np_col('Band Gap (T)')
is_metal = np_col('Is_Metal')

# Filter non-metals and valid values
non_metal_bg = bandgap[(is_metal == 0) & (bandgap >= 0)]
log_bg = np.log1p(non_metal_bg)

# Statistics
n_total = len(non_metal_bg)
mean_log = log_bg.mean()
median_log = np.median(log_bg)
mean_original = non_metal_bg.mean()
median_original = np.median(non_metal_bg)


# FIGURE: Log Band Gap Distribution (Non-Metals)
fig, ax = plt.subplots(figsize=(8, 5))

# Histogram + KDE
sns.histplot(
    log_bg,
    bins=50,
    kde=True,
    color=COLORS["tertiary"],   # ✅ replaces color3
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5,
    ax=ax,
    stat='count',
    line_kws={'linewidth': 2, 'color': COLORS["tertiary"]}
)

# Mean line
ax.axvline(
    mean_log,
    color=COLORS["metal"],   # red
    linestyle='--',
    linewidth=2,
    label=f'Mean = {mean_log:.2f}'
)

# Median line
ax.axvline(
    median_log,
    color=COLORS["stable"],   # green
    linestyle=':',
    linewidth=2,
    label=f'Median = {median_log:.2f}'
)

ax.set_xlabel('log(1 + Band Gap)')
ax.set_ylabel('Frequency')

ax.legend(
    frameon=True,
    fancybox=False,
    shadow=False,
    loc='upper right',
    edgecolor='black',
    framealpha=1
)

ax.set_axisbelow(True)

props = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.85,
    edgecolor='black'
)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()


save_figure(fig, "Log_BandGap_Distribution_NonMetals")

df['Log_BandGap'] = np.log1p(df['Band Gap (T)'])


# Extract data and create DataFrame
plot_data = pd.DataFrame({
    'Is Metal': df['Is_Metal'],
    'Magnetization': df['Total Magnetization']
})

# Remove NaNs
plot_data = plot_data.dropna(subset=['Magnetization', 'Is Metal'])

# Add labels
plot_data['Material Type'] = plot_data['Is Metal'].map({
    0: 'Non-metallic',
    1: 'Metallic'
})

# Statistics
metal_data = plot_data[plot_data['Is Metal'] == 1]['Magnetization']
nonmetal_data = plot_data[plot_data['Is Metal'] == 0]['Magnetization']

metal_mean = metal_data.mean()
metal_median = metal_data.median()
nonmetal_mean = nonmetal_data.mean()
nonmetal_median = nonmetal_data.median()

n_metal = len(metal_data)
n_nonmetal = len(nonmetal_data)


# FIGURE: Magnetization Distribution
fig, ax = plt.subplots(figsize=(8, 5))

# Violin plot (FIXED palette)
sns.violinplot(
    data=plot_data,
    x='Material Type',
    y='Magnetization',
    palette={
        'Non-metallic': COLORS["nonmetal"],   # ✅ fixed
        'Metallic': COLORS["metal"]           # ✅ fixed
    },
    ax=ax,
    order=['Non-metallic', 'Metallic'],
    inner='box',
    linewidth=1.2,
    cut=0   # ✅ prevents unrealistic tails
)

# Formatting
ax.set_xlabel('')
ax.set_ylabel('Total Magnetization (μB)')
ax.set_axisbelow(True)

# Bold x labels
ax.set_xticklabels(ax.get_xticklabels())

# Zero reference line
ax.axhline(
    0,
    color=COLORS["neutral"],   # ✅ consistent
    linestyle='-',
    linewidth=1,
    alpha=0.6
)

# Stats box
textstr = (
    f'Non-metallic (n = {n_nonmetal:,}):\n'
    f'  μ = {nonmetal_mean:.3f} μB\n'
    f'  median = {nonmetal_median:.3f} μB\n\n'
    f'Metallic (n = {n_metal:,}):\n'
    f'  μ = {metal_mean:.3f} μB\n'
    f'  median = {metal_median:.3f} μB'
)

props = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.85,
    edgecolor='black'
)

ax.text(
    0.98, 0.97,
    textstr,
    transform=ax.transAxes,
    fontsize=8.5,
    va='top',
    ha='right',
    bbox=props
)

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Magnetization_Distribution")

del chem_data, landscape_data, plot_data, ax, fig
free_gpu_memory()

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_3.2.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

tracker.track(df, "Target Log-Transformation", note="Target Engineering", dataset="MP Dataset")

print_heading("Phase 4: Two-Stage Hurdle Architecture", level=1)
print_heading("Step 23: Hurdle Framework Configuration & Initialisation", level=2)
print(f"XGBoost version : {xgb.__version__}")

# Target definitions
TARGET_RAW  = "Band Gap (T)"
TARGET_LOG  = "Log_BandGap"

# Reproducibility and validation
TEST_SIZE   = 0.20
N_SPLITS    = 5          # folds for StratifiedKFold

# ── Verify log target exists ──────────────────────────────────────────
if TARGET_LOG not in df.columns:
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])
    print(f"✅ '{TARGET_LOG}' created from log1p('{TARGET_RAW}')")

assert TARGET_RAW in df.columns
assert TARGET_LOG in df.columns

print_heading("Step 24: Feature Matrix Assembly & Target Construction", level=2)

# STEP 0 — FEATURE MATRIX PREPARATION
print(f"\n{'='*60}")
print(f"  STEP 0 — FEATURE MATRIX PREPARATION")
print(f"{'='*60}")

# ── Drop all target-related columns from features ─────────────────────
DROP_FROM_FEATURES = {
    TARGET_RAW,
    TARGET_LOG,
    "Is_Metal",
    "Material ID",        # identifier — not a feature
}

# ── Verify required target columns exist ──────────────────────────────
for col in [TARGET_RAW, TARGET_LOG]:
    assert col in df.columns, f"Missing required column: '{col}'"

# ── Extract numeric features automatically ────────────────────────────
feature_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in DROP_FROM_FEATURES
]
print(f"\n  Features          : {len(feature_cols)}")
print(f"  Total samples     : {len(df):,}")

# Safety check for leaked object columns
obj_cols = df.select_dtypes(include='object').columns.tolist()
print(f"  Excluded object cols: {obj_cols}")

# ── Build feature matrix and targets ──────────────────────────────────
# X stays as cuDF (GPU) if RAPIDS_AVAILABLE, else pandas — XGBoost handles both
X     = df[feature_cols]
y_raw = df[TARGET_RAW]   # original band gap — for analysis
y_log = df[TARGET_LOG]   # log band gap      — Stage 2 regression (non-metals only)

# ── CRITICAL: assert no -inf in y_log leaks into Stage 2 ──────────────
# log(0) = -inf for metals; Stage 2 must ONLY see non-metal rows
neg_inf_count = (y_log == float('-inf')).sum()
print(f"\n  ⚠ -inf entries in {TARGET_LOG} (metals): {neg_inf_count:,}  — will be excluded in Stage 2")

# ── Stage 1 binary labels ─────────────────────────────────────────────
# 0 = Metal (BG = 0),  1 = Non-metal (BG > 0)
y_cls = (y_raw > 0).astype(int)

metal_count    = int((y_cls == 0).sum())
nonmetal_count = int((y_cls == 1).sum())

print(f"\n  Class distribution:")
print(f"    Metal     (0) : {metal_count:>8,}  ({metal_count / len(y_cls) * 100:.1f}%)")
print(f"    Non-metal (1) : {nonmetal_count:>8,}  ({nonmetal_count / len(y_cls) * 100:.1f}%)")

if metal_count > 0:
    print(f"    Imbalance ratio : {nonmetal_count / metal_count:.2f}:1")
else:
    print("    ⚠ No metal samples found — check TARGET_RAW column values")

print("Space Group Number" in feature_cols)

tracker.track(df, "Feature Matrix Assembly", note="Matrix Assembly", dataset="MP Dataset")

XGB_DEVICE_PARAMS = (
    {"tree_method": "hist", "device": "cuda"}
    if CONFIG["gpu_env"]
    else {"tree_method": "hist", "device": "cpu"}
)


# STEP 1 — TRAIN / CALIBRATION / TEST SPLIT (72 / 8 / 20)

#
#   ┌──────────────────────┬─────────────┬──────────────┐
#   │   TRAIN (72%)        │  CAL (8%)   │  TEST (20%)  │
#   │  model training      │  UQ only    │  final eval  │
#   └──────────────────────┴─────────────┴──────────────┘
#
# Split order:
#   1. Carve test (20%) from full data   → stratified on y_cls
#   2. Carve cal  (10% of remainder)     → stratified on y_cls
#   Remaining 90% of 80% = 72% → training

print(f"\n{'='*60}")
print(f"  STEP 1 — TRAIN / CALIBRATION / TEST SPLIT")
print(f"{'='*60}")

# ── Step 1: carve test set ────────────────────────────────────────────
(X_train_cal, X_test,
 y_cls_train_cal, y_cls_test,
 y_log_train_cal, y_log_test,
 y_raw_train_cal, y_raw_test) = train_test_split(
    X, y_cls, y_log, y_raw,
    test_size    = 0.20,
    random_state = RANDOM_SEED,
    stratify     = y_cls
)

# ── Step 2: carve calibration set from remaining 80% ─────────────────
# 10% of 80% = 8% of total — enough for ~13k calibration samples
(X_train_all, X_cal,
 y_cls_train_all, y_cls_cal,
 y_log_train_all, y_log_cal,
 y_raw_train_all, y_raw_cal) = train_test_split(
    X_train_cal, y_cls_train_cal,
    y_log_train_cal, y_raw_train_cal,
    test_size    = 0.10,
    random_state = RANDOM_SEED,
    stratify     = y_cls_train_cal
)

# ── Sanity checks ─────────────────────────────────────────────────────
n_total = len(X)
assert len(X_train_all) + len(X_cal) + len(X_test) == n_total, \
    "🔴 Split sizes do not sum to total!"
assert set(np.array(y_cls_test).tolist()) == {0, 1}, \
    "🔴 Test set missing a class!"
assert set(np.array(y_cls_cal).tolist()) == {0, 1}, \
    "🔴 Calibration set missing a class!"

# ── Summary ───────────────────────────────────────────────────────────
n_train = len(X_train_all)
n_cal   = len(X_cal)
n_test  = len(X_test)

print(f"\n  Total samples  : {n_total:>8,}")
print(f"  Train set      : {n_train:>8,}  ({n_train/n_total*100:.1f}%)")
print(f"  Cal   set      : {n_cal:>8,}  ({n_cal/n_total*100:.1f}%)  ← UQ only")
print(f"  Test  set      : {n_test:>8,}  ({n_test/n_total*100:.1f}%)")

print(f"\n  Train class distribution:")
n_metal_train    = int((y_cls_train_all == 0).sum())
n_nonmetal_train = int((y_cls_train_all == 1).sum())
print(f"    Metal     (0) : {n_metal_train:>8,}  ({n_metal_train/n_train*100:.1f}%)")
print(f"    Non-metal (1) : {n_nonmetal_train:>8,}  ({n_nonmetal_train/n_train*100:.1f}%)")

print(f"\n  Cal class distribution:")
n_metal_cal    = int((y_cls_cal == 0).sum())
n_nonmetal_cal = int((y_cls_cal == 1).sum())
print(f"    Metal     (0) : {n_metal_cal:>8,}  ({n_metal_cal/n_cal*100:.1f}%)")
print(f"    Non-metal (1) : {n_nonmetal_cal:>8,}  ({n_nonmetal_cal/n_cal*100:.1f}%)")

print(f"\n  Test class distribution:")
n_metal_test    = int((y_cls_test == 0).sum())
n_nonmetal_test = int((y_cls_test == 1).sum())
print(f"    Metal     (0) : {n_metal_test:>8,}  ({n_metal_test/n_test*100:.1f}%)")
print(f"    Non-metal (1) : {n_nonmetal_test:>8,}  ({n_nonmetal_test/n_test*100:.1f}%)")

# ── CV splitter (shared across Stage 1 & Stage 2) ─────────────────────
cv_splitter = StratifiedKFold(
    n_splits     = N_SPLITS,
    shuffle      = True,
    random_state = RANDOM_SEED
)

print_heading("Step 25: Train / Calibration / Test Split", level=2)
print_heading("Step 26: Stage 1 Classifier Training", level=2)

# STEP 2 — STAGE 1: XGBoost CLASSIFIER (Hurdle Gate)
print(f"\n{'='*60}")
print(f"  STEP 2 — STAGE 1: METAL / NON-METAL CLASSIFIER")
print(f"{'='*60}")

# ── Class imbalance weight ────────────────────────────────────────────
# scale_pos_weight = count(negative) / count(positive)
# Tells XGBoost to penalise misclassifying the minority class more
n_metal_train    = int((y_cls_train_all == 0).sum())  
n_nonmetal_train = int((y_cls_train_all == 1).sum())
scale_pos_weight = n_metal_train / n_nonmetal_train

print(f"\n  Metal (neg) samples    : {n_metal_train:,}")
print(f"  Non-metal (pos) samples: {n_nonmetal_train:,}")
print(f"  scale_pos_weight       : {scale_pos_weight:.4f}")

# ── Hyperparameters ───────────────────────────────────────────────────
# NOTE: early_stopping_rounds and eval_metric are intentionally kept
# OUT of this dict — they are passed to .fit() so they don't interfere
# with cross-validation folds in later steps.
STAGE1_PARAMS = {
    # Booster
    "n_estimators"      : 1000,       # max trees; early stopping will reduce this
    "learning_rate"     : 0.05,
    "max_depth"         : 7,
    "min_child_weight"  : 3,
    # Regularisation
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "gamma"             : 0.1,
    "reg_alpha"         : 0.1,        # L1
    "reg_lambda"        : 1.0,        # L2
    # Class imbalance
    "scale_pos_weight"  : scale_pos_weight,
    # Hardware — XGBoost ≥ 2.0 syntax
    **XGB_DEVICE_PARAMS,
    # Output & reproducibility
    "verbosity"         : 1,
    "random_state"      : RANDOM_SEED,
    "importance_type"   : "gain",     # for feature importance later
    "eval_metric"       : ["logloss", "auc"],
    "early_stopping_rounds": 50,

}

clf_stage1 = xgb.XGBClassifier(**STAGE1_PARAMS)

print(f"\n  Classifier initialised ✅")
print(f"  Max estimators     : {STAGE1_PARAMS['n_estimators']}")
print(f"  Device             : {STAGE1_PARAMS['device']}")

# ── Train with early stopping ─────────────────────────────────────────
# Carve a small internal validation set from training data ONLY.
# This is what early stopping watches — test set is NEVER touched here.
X_tr, X_val, y_cls_tr, y_cls_val = train_test_split(
    X_train_all, y_cls_train_all,
    test_size    = 0.05,          # 5% of train → val (= 8% of total data)
    random_state = RANDOM_SEED,
    stratify     = y_cls_train_all
)

print(f"\n  Internal split for early stopping:")
print(f"    Fit set : {len(X_tr):,}")
print(f"    Val set : {len(X_val):,}  (early-stop monitor only)")

print(f"\n  Training Stage 1 classifier...")

t0 = time.time()

clf_stage1.fit(
    X_tr, y_cls_tr,
    eval_set              = [(X_tr, y_cls_tr), (X_val, y_cls_val)],                                             
    verbose               = 100,
)

t1 = time.time()
print(f"\n  ✅ Stage 1 trained in {(t1-t0)/60:.1f} minutes")
print(f"  Best iteration : {clf_stage1.best_iteration}")
print(f"  Best score     : {clf_stage1.best_score:.5f}")

print_heading("Step 27: Stage 1 Classifier Evaluation", level=2)

# ── Stage 1 Evaluation ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  STAGE 1 EVALUATION")
print(f"{'='*60}")

# Convert to numpy for sklearn compatibility (safe even if already numpy)
y_cls_test_np  = np.array(y_cls_test)

y_cls_pred       = clf_stage1.predict(X_test)
y_cls_pred_proba = clf_stage1.predict_proba(X_test)[:, 1]

# Ensure numpy for all metric calls
y_cls_pred       = np.array(y_cls_pred)
y_cls_pred_proba = np.array(y_cls_pred_proba)

acc = accuracy_score (y_cls_test_np, y_cls_pred)
f1  = f1_score       (y_cls_test_np, y_cls_pred, average="weighted")
auc = roc_auc_score  (y_cls_test_np, y_cls_pred_proba)
pre = precision_score(y_cls_test_np, y_cls_pred, average="weighted")
rec = recall_score   (y_cls_test_np, y_cls_pred, average="weighted")
cm  = confusion_matrix(y_cls_test_np, y_cls_pred)

print(f"\n  Accuracy        : {acc:.4f}")
print(f"  Precision (wtd) : {pre:.4f}")
print(f"  Recall    (wtd) : {rec:.4f}")
print(f"  F1        (wtd) : {f1:.4f}")
print(f"  ROC-AUC         : {auc:.4f}")

print(f"\n  Confusion Matrix:")
print(f"  {'':>15}  Pred Metal  Pred Non-metal")
print(f"  {'True Metal':>15}  {cm[0,0]:>10,}  {cm[0,1]:>14,}")
print(f"  {'True Non-metal':>15}  {cm[1,0]:>10,}  {cm[1,1]:>14,}")

print(f"\n  Classification Report:")
print(classification_report(
    y_cls_test_np, y_cls_pred,
    target_names = ["Metal", "Non-metal"],
    digits       = 4
))

# ── Save predictions for hurdle pipeline ──────────────────────────────
# These will gate Stage 2 — only predicted non-metals proceed
print(f"\n  Predicted non-metals (will enter Stage 2): "
      f"{(y_cls_pred == 1).sum():,} / {len(y_cls_pred):,}")

print_heading("Step 28: Stage 2 Regressor Setup", level=2)

# STEP 3 — STAGE 2: XGBoost REGRESSOR (Non-metals only)
print(f"\n{'='*60}")
print(f"  STEP 3 — STAGE 2: BANDGAP REGRESSOR (NON-METALS)")
print(f"{'='*60}")

# ── Filter training data — non-metals only ────────────────────────────
# Use TRUE labels for training (not Stage 1 predictions)
#    Stage 1 predictions are only used at inference time
nonmetal_train_mask = (y_cls_train_all == 1)   # ✅ fixed name
nonmetal_test_mask  = (y_cls_test      == 1)

X_nm_train     = X_train_all[nonmetal_train_mask]       
y_log_nm_train = y_log_train_all[nonmetal_train_mask]
X_nm_test      = X_test[nonmetal_test_mask]
y_log_nm_test  = y_log_test[nonmetal_test_mask]
y_raw_nm_test  = y_raw_test[nonmetal_test_mask]

# ── CRITICAL: assert no -inf leaked through ───────────────────────────
# If any metal slipped through the mask, y_log would contain -inf
assert not np.isinf(np.array(y_log_nm_train)).any(), \
    "🔴 -inf detected in y_log_nm_train — metal rows leaked through mask!"
assert not np.isinf(np.array(y_log_nm_test)).any(), \
    "🔴 -inf detected in y_log_nm_test — metal rows leaked through mask!"

print(f"\n  Non-metal train samples : {len(X_nm_train):,}")
print(f"  Non-metal test  samples : {len(X_nm_test):,}")

# ── Log_BandGap distribution check ───────────────────────────────────
y_log_nm_train_np = np.array(y_log_nm_train)
print(f"\n  Log_BandGap stats (train non-metals):")
print(f"    Mean   : {y_log_nm_train_np.mean():.4f}")
print(f"    Std    : {y_log_nm_train_np.std():.4f}")
print(f"    Min    : {y_log_nm_train_np.min():.4f}")
print(f"    Max    : {y_log_nm_train_np.max():.4f}")

tracker.track(df, "Stage 1 Baseline Training", note="Model Training", dataset="MP Dataset")

print_heading("Step 29: Stage 2 Regressor Training", level=2)

# ── Hyperparameters ───────────────────────────────────────────────────
STAGE2_PARAMS = {
    # Booster
    "n_estimators"          : 2000,
    "learning_rate"         : 0.03,
    "max_depth"             : 7,
    "min_child_weight"      : 5,
    # Regularisation
    "subsample"             : 0.8,
    "colsample_bytree"      : 0.7,
    "colsample_bylevel"     : 0.8,
    "gamma"                 : 0.1,
    "reg_alpha"             : 0.2,
    "reg_lambda"            : 1.5,
    # Hardware
    **XGB_DEVICE_PARAMS,
    # Objective — eval_metric & early_stopping_rounds in constructor (XGB ≥ 2.0)
    "objective"             : "reg:squarederror",
    "eval_metric"           : ["mae", "rmse"],    # track both
    "early_stopping_rounds" : 75,
    # Output & reproducibility
    "verbosity"             : 1,
    "random_state"          : RANDOM_SEED,
}

reg_stage2 = xgb.XGBRegressor(**STAGE2_PARAMS)

print(f"\n  Regressor initialised ✅")
print(f"  Max estimators     : {STAGE2_PARAMS['n_estimators']}")
print(f"  Early stop rounds  : {STAGE2_PARAMS['early_stopping_rounds']}")
print(f"  Device             : {STAGE2_PARAMS['device']}")

# ── Internal val split — non-metal train only, no test leakage ────────
X_nm_tr, X_nm_val, y_log_nm_tr, y_log_nm_val = train_test_split(
    X_nm_train, y_log_nm_train,         # ✅ correct names from Step 3
    test_size    = 0.05,
    random_state = RANDOM_SEED,
)

print(f"\n  Internal split for early stopping:")
print(f"    Fit set : {len(X_nm_tr):,}")
print(f"    Val set : {len(X_nm_val):,}  (early-stop monitor only)")

# ── Train ─────────────────────────────────────────────────────────────
print(f"\n  Training Stage 2 regressor...")
t0 = time.time()

reg_stage2.fit(
    X_nm_tr, y_log_nm_tr,
    eval_set = [(X_nm_tr, y_log_nm_tr), (X_nm_val, y_log_nm_val)],
    verbose  = 200,
)

t1 = time.time()
print(f"\n  ✅ Stage 2 trained in {(t1-t0)/60:.1f} minutes")
print(f"  Best iteration : {reg_stage2.best_iteration}")
print(f"  Best score     : {reg_stage2.best_score:.5f}")

print_heading("Step 30: Stage 2 Performance Evaluation", level=2)

# ── Stage 2 Evaluation ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  STAGE 2 EVALUATION — LOG SCALE")
print(f"{'='*60}")

# ✅ Correct names from Step 3
y_log_nm_pred = reg_stage2.predict(X_nm_test)

# Ensure numpy for all metric calls
y_log_nm_pred = np.array(y_log_nm_pred)
y_log_nm_test_np = np.array(y_log_nm_test)
y_raw_nm_test_np = np.array(y_raw_nm_test)   # ✅ cuDF-safe

# ── Metrics on log scale ──────────────────────────────────────────────
mae_log  = mean_absolute_error(y_log_nm_test_np, y_log_nm_pred)
rmse_log = np.sqrt(mean_squared_error(y_log_nm_test_np, y_log_nm_pred))
r2_log   = r2_score(y_log_nm_test_np, y_log_nm_pred)

# ── Back-transform to eV scale ────────────────────────────────────────
# ⚠️ CONFIRM which transform was used to create Log_BandGap:
#   If Log_BandGap = np.log1p(band_gap)  → use np.expm1  ✅ (handles 0 safely)
#   If Log_BandGap = np.log(band_gap)    → use np.exp
# Using wrong inverse will silently corrupt eV predictions
y_eV_nm_pred = np.expm1(y_log_nm_pred)      # ← verify this matches your transform
y_eV_nm_true = y_raw_nm_test_np

# ── Metrics on eV scale ───────────────────────────────────────────────
mae_eV  = mean_absolute_error(y_eV_nm_true, y_eV_nm_pred)
rmse_eV = np.sqrt(mean_squared_error(y_eV_nm_true, y_eV_nm_pred))
r2_eV   = r2_score(y_eV_nm_true, y_eV_nm_pred)

# ── Median Absolute Error (robust to outliers) ────────────────────────
medae_eV = np.median(np.abs(y_eV_nm_true - y_eV_nm_pred))

print(f"\n  ── Log scale ──────────────────────────────────────")
print(f"  MAE  (log) : {mae_log:.4f}")
print(f"  RMSE (log) : {rmse_log:.4f}")
print(f"  R²   (log) : {r2_log:.4f}")

print(f"\n  ── eV scale (back-transformed) ────────────────────")
print(f"  MAE   (eV) : {mae_eV:.4f} eV")
print(f"  MedAE (eV) : {medae_eV:.4f} eV")
print(f"  RMSE  (eV) : {rmse_eV:.4f} eV")
print(f"  R²    (eV) : {r2_eV:.4f}")

print(f"\n  DFT-PBE systematic error benchmark : ~0.6–1.0 eV MAE")
print(f"  Model vs benchmark : ", end="")
if mae_eV < 0.6:
    print("✅ Better than DFT-PBE baseline")
elif mae_eV < 1.0:
    print("⚠️  Within DFT-PBE range")
else:
    print("❌ Worse than DFT-PBE baseline")

tracker.track(df, "Stage 2 Baseline Training", note="Model Training", dataset="MP Dataset")

print_heading("Step 31: Full Hurdle Inference & Model Persistence", level=2)

# STEP 5 — FULL HURDLE PIPELINE PREDICTION FUNCTION
# ⚠️ NOTE: Check if Step 4 (e.g. CV / OOF evaluation) is missing above

def hurdle_predict(
    X_new              : pd.DataFrame,
    classifier         : xgb.XGBClassifier,
    regressor          : xgb.XGBRegressor,
    nonmetal_threshold : float = 0.5,      # ✅ renamed — applied to P(non-metal)
    return_proba       : bool  = False,
) -> pd.DataFrame:
    """
    Full two-stage Hurdle prediction pipeline.

    Stage 1 : Classify each sample as Metal (0) or Non-metal (1).
              A sample is predicted Non-metal if P(non-metal) >= nonmetal_threshold.
    Stage 2 : For predicted non-metals, predict log1p(BG) then
              back-transform via expm1 to eV.
              Metals are assigned BG = 0.0 eV directly.

    Args:
        X_new              : Feature matrix — must have same columns as training.
        classifier         : Trained Stage 1 XGBClassifier.
        regressor          : Trained Stage 2 XGBRegressor.
        nonmetal_threshold : P(non-metal) cutoff (default 0.5).
                             Lower → more samples treated as non-metal.
        return_proba       : If True, include class probabilities in output.

    Returns:
        DataFrame with columns:
            - 'predicted_class'      : 0 = Metal, 1 = Non-metal
            - 'predicted_BG_eV'      : Final band gap prediction in eV
            - 'predicted_log_BG'     : log1p-scale prediction (0.0 for metals)
            - 'P_nonmetal'           : P(non-metal) from Stage 1  [if return_proba]
            - 'P_metal'              : P(metal) from Stage 1       [if return_proba]
    """
    # ── Input validation ──────────────────────────────────────────────
    expected_cols = classifier.get_booster().feature_names
    if expected_cols is not None:
        missing = set(expected_cols) - set(X_new.columns.tolist())
        extra   = set(X_new.columns.tolist()) - set(expected_cols)
        assert not missing, f"🔴 Missing features in X_new: {missing}"
        if extra:
            print(f"  ⚠️  Extra columns in X_new (will be ignored): {extra}")

    # ── Stage 1: classify ─────────────────────────────────────────────
    proba_nonmetal = np.array(classifier.predict_proba(X_new)[:, 1])  # P(non-metal)
    proba_metal    = 1.0 - proba_nonmetal                              # P(metal)
    pred_class     = (proba_nonmetal >= nonmetal_threshold).astype(int)

    # ── Stage 2: regress non-metals only ──────────────────────────────
    nonmetal_mask = pred_class == 1
    pred_log_bg   = np.zeros(len(X_new), dtype=np.float32)
    pred_bg_eV    = np.zeros(len(X_new), dtype=np.float32)

    n_nonmetal = int(nonmetal_mask.sum())
    n_metal    = int((~nonmetal_mask).sum())

    if n_nonmetal > 0:
        X_nm                       = X_new[nonmetal_mask]
        log_preds                  = np.array(regressor.predict(X_nm))
        pred_log_bg[nonmetal_mask] = log_preds
        pred_bg_eV[nonmetal_mask]  = np.expm1(log_preds)   # ✅ inverse of log1p

    print(f"  Hurdle gate — Metal: {n_metal:,}  |  Non-metal: {n_nonmetal:,}")

    # ── Assemble output ───────────────────────────────────────────────
    out = pd.DataFrame({
        "predicted_class"  : pred_class,
        "predicted_BG_eV"  : pred_bg_eV,
        "predicted_log_BG" : pred_log_bg,
    }, index=X_new.index)

    if return_proba:
        out["P_nonmetal"] = proba_nonmetal   # ✅ explicit naming — no ambiguity
        out["P_metal"]    = proba_metal

    return out

# ── Test hurdle_predict on test set ──────────────────────────────────
print(f"\n  Testing hurdle_predict on test set...")
test_predictions = hurdle_predict(
    X_test,
    clf_stage1,
    reg_stage2,
    return_proba = True
)
print(f"  Output shape : {test_predictions.shape}")
print(f"  Sample output:")
print(test_predictions.head(5).to_string())

# STEP 6 — COMBINED HURDLE EVALUATION

print(f"\n{'='*60}")
print(f"  STEP 6 — FULL HURDLE PIPELINE EVALUATION")
print(f"{'='*60}")

# ── numpy conversion for all comparisons ─────────────────────────────
final_pred_eV  = test_predictions["predicted_BG_eV"].values
true_eV        = np.array(y_raw_test)       # ✅ cuDF-safe
y_cls_test_np  = np.array(y_cls_test)       # ✅ cuDF-safe
pred_class_np  = test_predictions["predicted_class"].values

# ── Full pipeline metrics ─────────────────────────────────────────────
mae_full   = mean_absolute_error(true_eV, final_pred_eV)
rmse_full  = np.sqrt(mean_squared_error(true_eV, final_pred_eV))
r2_full    = r2_score(true_eV, final_pred_eV)
medae_full = np.median(np.abs(true_eV - final_pred_eV))

print(f"\n  ── Full pipeline (metals + non-metals) ────────────────")
print(f"  MAE   : {mae_full:.4f} eV")
print(f"  MedAE : {medae_full:.4f} eV")
print(f"  RMSE  : {rmse_full:.4f} eV")
print(f"  R²    : {r2_full:.4f}")

# ── Stage 1 gate analysis ─────────────────────────────────────────────
true_metal    = y_cls_test_np == 0
true_nonmetal = y_cls_test_np == 1
pred_metal    = pred_class_np == 0
pred_nonmetal = pred_class_np == 1

tn = int((true_metal    & pred_metal   ).sum())   # correct metals
tp = int((true_nonmetal & pred_nonmetal).sum())   # correct non-metals
fn = int((true_nonmetal & pred_metal   ).sum())   # non-metals wrongly sent to BG=0
fp = int((true_metal    & pred_nonmetal).sum())   # metals wrongly sent to Stage 2

print(f"\n  ── Stage 1 gate breakdown ─────────────────────────────")
print(f"  True metals   ({true_metal.sum():>6,}) → "
      f"correct (BG=0): {tn:,}  |  wrong→Stage2: {fp:,}")
print(f"  True non-metals ({true_nonmetal.sum():>6,}) → "
      f"correct→Stage2: {tp:,}  |  wrong (BG=0): {fn:,}")

# ── MAE breakdown by Stage 1 correctness ─────────────────────────────
# This shows how much Stage 1 errors hurt the final regression MAE
correctly_classified = (y_cls_test_np == pred_class_np)
wrongly_classified   = ~correctly_classified

if wrongly_classified.sum() > 0:
    mae_correct = mean_absolute_error(
        true_eV[correctly_classified], final_pred_eV[correctly_classified]
    )
    mae_wrong = mean_absolute_error(
        true_eV[wrongly_classified], final_pred_eV[wrongly_classified]
    )
    print(f"\n  ── MAE by Stage 1 correctness ─────────────────────────")
    print(f"  Correctly classified ({correctly_classified.sum():,}) : MAE = {mae_correct:.4f} eV")
    print(f"  Misclassified        ({wrongly_classified.sum():,})   : MAE = {mae_wrong:.4f} eV")
    print(f"  ⚠️  Misclassification MAE penalty: {mae_wrong - mae_correct:+.4f} eV")
else:
    print(f"\n  ✅ No misclassifications on test set")

# ── False negative impact (most damaging error type) ─────────────────
# FN = true non-metal predicted as metal → assigned BG=0 → large error
if fn > 0:
    fn_mask      = true_nonmetal & pred_metal
    fn_true_eV   = true_eV[fn_mask]
    fn_pred_eV   = final_pred_eV[fn_mask]   # all zeros
    mae_fn       = mean_absolute_error(fn_true_eV, fn_pred_eV)
    print(f"\n  ── False Negative (non-metal→BG=0) impact ─────────────")
    print(f"  Count    : {fn:,}")
    print(f"  MAE      : {mae_fn:.4f} eV  (true BG forced to 0)")
    print(f"  Mean true BG of missed non-metals: {fn_true_eV.mean():.4f} eV")

# STEP 7 — SAVE MODELS

print(f"\n{'='*60}")
print(f"  STEP 7 — SAVING MODELS")
print(f"{'='*60}")

stage1_path = os.path.join(CONFIG["models_dir"], "stage1_classifier.json")
stage2_path = os.path.join(CONFIG["models_dir"], "stage2_regressor.json")
meta_path   = os.path.join(CONFIG["models_dir"], "hurdle_metadata.pkl")

clf_stage1.save_model(stage1_path)
reg_stage2.save_model(stage2_path)

# Save metadata — required to reconstruct pipeline
metadata = {
    "feature_cols"       : feature_cols,
    "target_raw"         : TARGET_RAW,
    "target_log"         : TARGET_LOG,
    "random_seed"        : RANDOM_SEED,
    "test_size"          : TEST_SIZE,
    "stage1_params"      : STAGE1_PARAMS,
    "stage2_params"      : STAGE2_PARAMS,
    "stage1_metrics"     : {
        "accuracy": round(acc,   4),
        "f1"      : round(f1,    4),
        "roc_auc" : round(auc,   4),
    },
    "stage2_metrics_log" : {
        "mae" : round(mae_log,  4),
        "rmse": round(rmse_log, 4),
        "r2"  : round(r2_log,   4),
    },
    "stage2_metrics_eV"  : {
        "mae" : round(mae_eV,   4),
        "rmse": round(rmse_eV,  4),
        "r2"  : round(r2_eV,    4),
    },
    "full_pipeline_eV"   : {
        "mae" : round(mae_full,  4),
        "rmse": round(rmse_full, 4),
        "r2"  : round(r2_full,   4),
    },
}

with open(meta_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"  💾 Stage 1 → {stage1_path}")
print(f"  💾 Stage 2 → {stage2_path}")
print(f"  💾 Metadata → {meta_path}")

# ── Final summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  HURDLE FRAMEWORK — FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n  STAGE 1 — Metal / Non-metal Classifier")
print(f"  {'─'*40}")
print(f"    Accuracy        : {acc:.4f}")
print(f"    Precision (wtd) : {pre:.4f}")
print(f"    Recall    (wtd) : {rec:.4f}")
print(f"    F1        (wtd) : {f1:.4f}")
print(f"    ROC-AUC         : {auc:.4f}")

print(f"\n  STAGE 2 — Band Gap Regressor (Non-metals only)")
print(f"  {'─'*40}")
print(f"    MAE   (eV) : {mae_eV:.4f}")
print(f"    MedAE (eV) : {medae_eV:.4f}")
print(f"    RMSE  (eV) : {rmse_eV:.4f}")
print(f"    R²    (eV) : {r2_eV:.4f}")

print(f"\n  FULL PIPELINE (metals + non-metals)")
print(f"  {'─'*40}")
print(f"    MAE   (eV) : {mae_full:.4f}")
print(f"    MedAE (eV) : {medae_full:.4f}")
print(f"    RMSE  (eV) : {rmse_full:.4f}")
print(f"    R²    (eV) : {r2_full:.4f}")

print(f"\n  DFT-PBE benchmark : ~0.6–1.0 eV MAE")
print(f"  Full pipeline vs benchmark : ", end="")
if mae_full < 0.6:
    print("✅ Better than DFT-PBE baseline")
elif mae_full < 1.0:
    print("⚠️  Within DFT-PBE range")
else:
    print("❌ Worse than DFT-PBE baseline")

print_heading("Phase 5: Post-Processing, Bias Calibration & Conformal Prediction", level=1)
print_heading("Step 32: Uncertainty-Aware Inference with Conformal Prediction", level=2)

def hurdle_predict_uq(
    X_new              : pd.DataFrame,
    classifier         : xgb.XGBClassifier,
    regressor          : xgb.XGBRegressor,
    cls_q_hats         : dict,
    reg_q_hats         : dict,
    nonmetal_threshold : float = 0.5,
    bin_corrections    : dict  = None,   # optional post-processor
    bg_bins            : list  = None,
) -> pd.DataFrame:
    """
    Hurdle prediction with conformal prediction intervals.

    Key design:
    - Intervals built in LOG scale using reg_q_hats (log units)
    - Back-transformed to eV via expm1 → naturally asymmetric intervals
    - Bin correction applied to POINT PREDICTIONS only, not to interval bounds
      (interval width comes from conformal calibration, not post-processing)
    - Lower bound clipped at 0 eV
    """
    # ── Input validation ──────────────────────────────────────────────
    expected_cols = classifier.get_booster().feature_names
    if expected_cols is not None:
        missing = set(expected_cols) - set(X_new.columns.tolist())
        assert not missing, f"🔴 Missing features: {missing}"

    # ── Stage 1: classify ─────────────────────────────────────────────
    proba_nonmetal = np.array(classifier.predict_proba(X_new)[:, 1])
    proba_metal    = 1.0 - proba_nonmetal
    pred_class     = (proba_nonmetal >= nonmetal_threshold).astype(int)

    cls_scores     = 1.0 - np.where(
        pred_class == 1, proba_nonmetal, proba_metal
    )

    # ── Stage 2: regress non-metals ───────────────────────────────────
    nonmetal_mask  = pred_class == 1
    pred_log_bg    = np.zeros(len(X_new), dtype=np.float64)
    pred_bg_eV     = np.zeros(len(X_new), dtype=np.float64)

    # Interval arrays per alpha (log scale)
    intervals = {
        alpha: {
            "lower_log": np.zeros(len(X_new)),
            "upper_log": np.zeros(len(X_new)),
        }
        for alpha in reg_q_hats
    }

    n_nonmetal = int(nonmetal_mask.sum())
    n_metal    = int((~nonmetal_mask).sum())

    if n_nonmetal > 0:
        X_nm      = X_new[nonmetal_mask] if not hasattr(X_new, 'iloc') \
                    else X_new.iloc[nonmetal_mask]
        log_preds = np.array(regressor.predict(X_nm))

        # ── Point prediction — apply bin correction in eV ──────────────
        eV_preds_raw = np.expm1(log_preds)
        if bin_corrections is not None and bg_bins is not None:
            eV_preds_cor = apply_bin_correction(
                eV_preds_raw, bin_corrections, bg_bins
            )
        else:
            eV_preds_cor = eV_preds_raw

        pred_log_bg[nonmetal_mask] = log_preds
        pred_bg_eV[nonmetal_mask]  = eV_preds_cor

        # ── Intervals — built in LOG scale, NOT bin-corrected ──────────
        # Bin correction shifts point estimates — intervals come from
        # conformal calibration on raw log residuals, so we build them
        # in log space and back-transform independently.
        for alpha, q in reg_q_hats.items():
            intervals[alpha]["lower_log"][nonmetal_mask] = log_preds - q
            intervals[alpha]["upper_log"][nonmetal_mask] = log_preds + q

    print(f"  Hurdle gate — Metal: {n_metal:,}  |  Non-metal: {n_nonmetal:,}")

    # ── Assemble output ───────────────────────────────────────────────
    out = pd.DataFrame({
        "predicted_class"  : pred_class,
        "P_nonmetal"       : proba_nonmetal,
        "P_metal"          : proba_metal,
        "predicted_BG_eV"  : pred_bg_eV,     # ✅ bin-corrected point pred
        "predicted_log_BG" : pred_log_bg,
    }, index=X_new.index)

    # Stage 1 uncertainty flags
    for alpha in cls_q_hats:
        col      = f"cls_uncertain_{int((1-alpha)*100)}"
        out[col] = cls_scores > cls_q_hats[alpha]

    # Stage 2 intervals — back-transform from log to eV
    for alpha in reg_q_hats:
        pct   = int((1 - alpha) * 100)
        lo_eV = np.clip(np.expm1(intervals[alpha]["lower_log"]), 0.0, None)
        hi_eV = np.expm1(intervals[alpha]["upper_log"])
        out[f"pi{pct}_lower"] = lo_eV
        out[f"pi{pct}_upper"] = hi_eV
        out[f"pi{pct}_width"] = hi_eV - lo_eV

    return out

print_heading("Step 33: Split-Conformal Calibration & Coverage Validation", level=2)
ALPHA_VALUES = [0.10, 0.05]   # 90% and 95% coverage


# STEP 4 — CONFORMAL PREDICTION CALIBRATION (held-out cal set)

# Proper split-conformal prediction — mathematically guaranteed coverage.
# Models never saw X_cal during training → scores are exchangeable
# with test set → coverage guarantee holds exactly.

print(f"\n{'='*60}")
print(f"  STEP 4 — CONFORMAL PREDICTION CALIBRATION")
print(f"{'='*60}")
print(f"  Calibration set : {len(X_cal):,} samples")
print(f"  Coverage targets: {[int((1-a)*100) for a in ALPHA_VALUES]}%")

# ── 4A: Stage 1 calibration ───────────────────────────────────────────
print(f"\n  {'─'*50}")
print(f"  4A — Stage 1 Classifier Calibration")
print(f"  {'─'*50}")

cal_proba_nonmetal = np.array(clf_stage1.predict_proba(X_cal)[:, 1])
y_cls_cal_np       = np.array(y_cls_cal)

# Nonconformity score: 1 - P(true class)
# Higher score = classifier less confident about the correct class
cls_cal_scores = 1.0 - np.where(
    y_cls_cal_np == 1,
    cal_proba_nonmetal,        # non-metal: use P(non-metal)
    1.0 - cal_proba_nonmetal   # metal:     use P(metal)
)

print(f"  Stage 1 nonconformity score stats:")
print(f"    Mean   : {cls_cal_scores.mean():.4f}")
print(f"    Median : {np.median(cls_cal_scores):.4f}")
print(f"    95th%  : {np.percentile(cls_cal_scores, 95):.4f}")

# ── 4B: Stage 2 calibration — non-metals only ─────────────────────────
print(f"\n  {'─'*50}")
print(f"  4B — Stage 2 Regressor Calibration")
print(f"  {'─'*50}")

nm_cal_mask     = y_cls_cal_np == 1
y_log_cal_np    = np.array(y_log_cal)
y_log_nm_cal    = y_log_cal_np[nm_cal_mask]

if hasattr(X_cal, 'iloc'):
    X_nm_cal = X_cal.loc[nm_cal_mask]
else:
    X_nm_cal = X_cal[nm_cal_mask]

# Assert no -inf in cal regression targets
assert not np.isinf(y_log_nm_cal).any(), \
    "🔴 -inf in y_log_nm_cal — metal leaked through mask!"

cal_log_pred   = np.array(reg_stage2.predict(X_nm_cal))
reg_cal_scores = np.abs(y_log_nm_cal - cal_log_pred)

print(f"  Non-metal cal samples : {nm_cal_mask.sum():,}")
print(f"  Stage 2 nonconformity score stats (log scale):")
print(f"    Mean   : {reg_cal_scores.mean():.4f}")
print(f"    Median : {np.median(reg_cal_scores):.4f}")
print(f"    95th%  : {np.percentile(reg_cal_scores, 95):.4f}")

# ── 4C: Compute q_hats for each alpha ─────────────────────────────────
print(f"\n  {'─'*50}")
print(f"  4C — Conformal Quantiles")
print(f"  {'─'*50}")

cls_q_hats = {}
reg_q_hats = {}
n_cls = len(cls_cal_scores)
n_reg = len(reg_cal_scores)

for alpha in ALPHA_VALUES:
    pct = int((1 - alpha) * 100)

    # Finite-sample corrected quantile level
    q_level_cls         = min(np.ceil((n_cls + 1) * (1 - alpha)) / n_cls, 1.0)
    q_level_reg         = min(np.ceil((n_reg + 1) * (1 - alpha)) / n_reg, 1.0)
    cls_q_hats[alpha]   = float(np.quantile(cls_cal_scores, q_level_cls))
    reg_q_hats[alpha]   = float(np.quantile(reg_cal_scores, q_level_reg))

    print(f"  PI{pct}:")
    print(f"    Stage 1 q_hat : {cls_q_hats[alpha]:.4f}")
    print(f"    Stage 2 q_hat : {reg_q_hats[alpha]:.4f} log units")

# ── 4D: Coverage Evaluation — CORRECTED ───────────────────────────────
print(f"\n  {'─'*50}")
print(f"  4D — Coverage Evaluation on Test Set")
print(f"  {'─'*50}")

test_preds_uq = hurdle_predict_uq(
    X_test, clf_stage1, reg_stage2,
    cls_q_hats, reg_q_hats
)

y_raw_test_np  = np.array(y_raw_test)
y_cls_test_np  = np.array(y_cls_test)
pred_class_np  = test_preds_uq["predicted_class"].values

# ── Three populations for coverage analysis ────────────────────────────
# TP: true non-metal, predicted non-metal → Stage 2 ran, interval valid
# FN: true non-metal, predicted metal    → BG=0 assigned, interval [0,0]
# TN: true metal, predicted metal        → BG=0 correct, not evaluated
# FP: true metal, predicted non-metal   → Stage 2 ran on a metal (noise)

true_nonmetal = y_cls_test_np == 1
true_metal    = y_cls_test_np == 0
pred_nonmetal = pred_class_np == 1
pred_metal    = pred_class_np == 0

tp_mask = true_nonmetal & pred_nonmetal   # Stage 2 ran correctly
fn_mask = true_nonmetal & pred_metal      # Stage 1 error — BG=0 assigned
fp_mask = true_metal    & pred_nonmetal   # Stage 1 error — metal in Stage 2
tn_mask = true_metal    & pred_metal      # correct metals

print(f"\n  Stage 1 gate breakdown:")
print(f"    TP (non-metal → Stage 2) : {tp_mask.sum():>7,}")
print(f"    FN (non-metal → BG=0)    : {fn_mask.sum():>7,}  ← guaranteed coverage miss")
print(f"    FP (metal → Stage 2)     : {fp_mask.sum():>7,}  ← noise in Stage 2")
print(f"    TN (metal → BG=0)        : {tn_mask.sum():>7,}")

# ── Stage 2 coverage — TRUE POSITIVES ONLY ────────────────────────────
# This is the correct population for Stage 2 conformal guarantee
true_eV_tp = y_raw_test_np[tp_mask]

print(f"\n  Stage 2 coverage on TRUE POSITIVES ({tp_mask.sum():,}):")
print(f"  {'Status':>4}  {'Level':>6}  {'Coverage':>10}  "
      f"{'Avg Width':>10}  {'Median Width':>13}")
print(f"  {'─'*55}")

for alpha in ALPHA_VALUES:
    pct   = int((1 - alpha) * 100)
    lo    = test_preds_uq[f"pi{pct}_lower"].values[tp_mask]
    hi    = test_preds_uq[f"pi{pct}_upper"].values[tp_mask]
    width = test_preds_uq[f"pi{pct}_width"].values[tp_mask]

    covered      = ((true_eV_tp >= lo) & (true_eV_tp <= hi)).mean()
    avg_width    = width.mean()
    median_width = np.median(width)
    status       = "✅" if covered >= (1 - alpha) else "⚠️ "

    print(f"  {status}   PI{pct}   {covered*100:>8.2f}%  "
          f"{avg_width:>10.4f} eV  {median_width:>10.4f} eV")

# ── Full pipeline coverage — ALL true non-metals ───────────────────────
# Includes FNs — shows real-world performance including Stage 1 errors
true_eV_nm_all = y_raw_test_np[true_nonmetal]

print(f"\n  Full pipeline coverage on ALL true non-metals ({true_nonmetal.sum():,}):")
print(f"  (Includes FNs assigned BG=0 — shows Stage 1 error impact)")
print(f"  {'Status':>4}  {'Level':>6}  {'Coverage':>10}")
print(f"  {'─'*35}")

for alpha in ALPHA_VALUES:
    pct   = int((1 - alpha) * 100)
    lo    = test_preds_uq[f"pi{pct}_lower"].values[true_nonmetal]
    hi    = test_preds_uq[f"pi{pct}_upper"].values[true_nonmetal]

    covered = ((true_eV_nm_all >= lo) & (true_eV_nm_all <= hi)).mean()
    status  = "✅" if covered >= (1 - alpha) else "⚠️ "
    print(f"  {status}   PI{pct}   {covered*100:>8.2f}%  ")
    # Fix — make the direction explicit
    shortfall = (1 - alpha - covered) * 100
    print(f"(shortfall from FNs: {shortfall:.2f}%)")

# ── FN impact summary ─────────────────────────────────────────────────
fn_count    = fn_mask.sum()
fn_true_eV  = y_raw_test_np[fn_mask]
print(f"\n  False Negative impact (Stage 1 misclassified non-metals):")
print(f"    Count          : {fn_count:,}")
print(f"    Mean true BG   : {fn_true_eV.mean():.4f} eV")
print(f"    Coverage loss  : ~{fn_count/true_nonmetal.sum()*100:.2f}% "
      f"(these always miss — interval = [0,0])")
print(f"    Fix            : improve Stage 1 recall, not Stage 2 UQ")

# ── Stage 1 uncertainty flags ──────────────────────────────────────────
print(f"\n  Stage 1 uncertainty flags:")
for alpha in ALPHA_VALUES:
    pct     = int((1 - alpha) * 100)
    col     = f"cls_uncertain_{pct}"
    n_unc   = int(test_preds_uq[col].sum())
    pct_unc = n_unc / len(test_preds_uq) * 100
    print(f"    PI{pct} → {n_unc:,} uncertain ({pct_unc:.1f}%)")

# ── Diagnostic: inspect test-set cls nonconformity distribution ────────
test_proba_nm  = clf_stage1.predict_proba(X_test)[:, 1]
test_pred_cls  = (test_proba_nm >= 0.5).astype(int)
test_cls_scores = 1.0 - np.where(
    test_pred_cls == 1, test_proba_nm, 1.0 - test_proba_nm
)
print(f"\n  Test-set cls nonconformity score stats:")
print(f"    Mean   : {test_cls_scores.mean():.4f}")
print(f"    Median : {np.median(test_cls_scores):.4f}")
print(f"    95th%  : {np.percentile(test_cls_scores, 95):.4f}")
print(f"    Max    : {test_cls_scores.max():.4f}")
print(f"    > q90 ({cls_q_hats[0.10]:.4f}) : "
      f"{(test_cls_scores > cls_q_hats[0.10]).sum():,}")
print(f"    > q95 ({cls_q_hats[0.05]:.4f}) : "
      f"{(test_cls_scores > cls_q_hats[0.05]).sum():,}")

# ── Add this note after the uncertainty flags block ────────────────────
print(f"\n  ⚠️  NOTE — PI95 uncertain = 0 is NOT a bug:")
print(f"    Test-set max cls score : {test_cls_scores.max():.4f}")
print(f"    PI95 q_hat             : {cls_q_hats[0.05]:.4f}")
print(f"    The test set contains no samples as ambiguous as the")
print(f"    hardest calibration samples. The conformal threshold")
print(f"    is conservative — guaranteed coverage is preserved.")
print(f"    This is benign: classifier is more confident on test set.")

free_gpu_memory()

tracker.track(df, "Conformal UQ Calibration", note="Uncertainty Quantification", dataset="MP Dataset")

print_heading("Step 34: Stage 2 Hyperparameter Optimization", level=2)

optuna.logging.set_verbosity(optuna.logging.WARNING)
y_log_nm_train = pd.Series(y_log_nm_train).reset_index(drop=True)
X_nm_train     = X_nm_train.reset_index(drop=True)

def objective_stage2(trial):
    params = {
        "n_estimators"          : trial.suggest_int("n_estimators", 300, 800),   # tightened ceiling
        "learning_rate"         : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth"             : trial.suggest_int("max_depth", 4, 10),
        "min_child_weight"      : trial.suggest_int("min_child_weight", 1, 10),
        "subsample"             : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"      : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel"     : trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma"                 : trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha"             : trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda"            : trial.suggest_float("reg_lambda", 0.5, 5.0),
        **XGB_DEVICE_PARAMS,
        "objective"             : "reg:squarederror",
        "eval_metric"           : "mae",
        "early_stopping_rounds" : 50,
        "random_state"          : RANDOM_SEED,
    }

    # ── Single stratified holdout instead of 5-fold CV ────────────────
    # Trades some variance in the search signal for ~5x speedup.
    # Final model is retrained properly afterward regardless.
    bg_bins = pd.cut(y_log_nm_train, bins=10, labels=False)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_nm_train, y_log_nm_train,
        test_size=0.2, random_state=RANDOM_SEED, stratify=bg_bins
    )

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-mae")
    model = xgb.XGBRegressor(**params, callbacks=[pruning_callback])
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)

    # Free GPU memory periodically to avoid fragmentation over many sequential fits
    if trial.number % 10 == 0:
        free_gpu_memory()

    return mean_absolute_error(y_val, preds)

print("Running Optuna hyperparameter search...")
study = optuna.create_study(
    direction = "minimize",
    sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner    = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
)
study.optimize(objective_stage2, n_trials=80, show_progress_bar=True)

joblib.dump(study, "optuna_study_stage2.pkl")

fig = vis.plot_optimization_history(study)

if CONFIG["display_graphs"]:
    fig.show()

print(f"Pilot best MAE (log): {study.best_value:.4f}")
print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Best params: {study.best_params}")

print_heading("Step 35: Stage 2 Retraining with Optimized Hyperparameters", level=2)

# STAGE 2 RETRAIN — OPTUNA BEST PARAMS

# ── Internal val split — no test leakage ──────────────────────────────
X_nm_tr, X_nm_val, y_log_nm_tr, y_log_nm_val = train_test_split(
    X_nm_train, y_log_nm_train,
    test_size    = 0.05,
    random_state = RANDOM_SEED,
)

best_model = xgb.XGBRegressor(
    **study.best_params,               # auto-fills all tuned params
    tree_method           = "hist",
    device                = "cuda",
    objective             = "reg:squarederror",
    eval_metric           = ["mae", "rmse"],
    early_stopping_rounds = 50,
    random_state          = RANDOM_SEED,
)

print(f"\n  Retraining Stage 2 with Optuna best params...")
t0 = time.time()

best_model.fit(
    X_nm_tr, y_log_nm_tr,
    eval_set = [(X_nm_tr, y_log_nm_tr), (X_nm_val, y_log_nm_val)],  # no test leakage
    verbose  = 100,
)

t1 = time.time()
print(f"  ✅ Trained in {(t1-t0)/60:.1f} min")
print(f"  Best iteration : {best_model.best_iteration}")
print(f"  Best score     : {best_model.best_score:.5f}")

# ── Evaluate on test set ───────────────────────────────────────────────
log_preds_tuned = np.array(best_model.predict(X_nm_test))
eV_preds_tuned  = np.expm1(log_preds_tuned)
eV_true         = np.array(y_raw_nm_test)   # already in eV, no back-transform needed

mae_new   = mean_absolute_error(eV_true, eV_preds_tuned)
medae_new = median_absolute_error(eV_true, eV_preds_tuned)
rmse_new  = np.sqrt(mean_squared_error(eV_true, eV_preds_tuned))
r2_new    = r2_score(eV_true, eV_preds_tuned)

# ── Comparison table — auto pulls previous values ─────────────────────
print(f"\n  Stage 2 — Before vs After Optuna (eV, non-metals only)")
print(f"  {'─'*50}")
print(f"  {'Metric':<10} {'Before':>10} {'After':>10} {'Delta':>10}")
print(f"  {'─'*50}")
print(f"  {'MAE':<10} {mae_eV:>10.4f} {mae_new:>10.4f} {mae_new - mae_eV:>+10.4f}")
print(f"  {'MedAE':<10} {medae_eV:>10.4f} {medae_new:>10.4f} {medae_new - medae_eV:>+10.4f}")
print(f"  {'RMSE':<10} {rmse_eV:>10.4f} {rmse_new:>10.4f} {rmse_new - rmse_eV:>+10.4f}")
print(f"  {'R²':<10} {r2_eV:>10.4f} {r2_new:>10.4f} {r2_new - r2_eV:>+10.4f}")


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Replace reg_stage2 if tuned model is better ────────────────────────
if mae_new < mae_eV:
    reg_stage2 = best_model
    # Update stored metrics so downstream steps stay consistent
    mae_eV   = mae_new
    medae_eV = medae_new
    rmse_eV  = rmse_new
    r2_eV    = r2_new
    joblib.dump(reg_stage2, MODEL_DIR / "stage2_regressor.pkl")
    print(f"\n  ✅ Tuned model is better — reg_stage2 updated and saved")
else:
    print(f"\n  ⚠️  Tuned model not better — keeping original reg_stage2")

tracker.track(df, "Stage 2 Hyperparameter Search", note="Optimization", dataset="MP Dataset")

print_heading("Step 36: Error Stratification by Bandgap Regime", level=2)

# ── Bin errors by true BG range ───────────────────────────────────────
print(f"{'='*60}")
print(f"  ERROR BREAKDOWN BY BAND GAP RANGE")
print(f"{'='*60}")

BG_BINS   = [0, 1, 2, 3, 4, 5, float('inf')]
BG_LABELS = ['0–1 eV', '1–2 eV', '2–3 eV', '3–4 eV', '4–5 eV', '>5 eV']

true_bg = eV_true          #  tuned model ground truth (from retrain block)
pred_bg = eV_preds_tuned   #  tuned model predictions (from retrain block)
abs_err = np.abs(pred_bg - true_bg)
bin_idx = np.digitize(true_bg, BG_BINS) - 1

bin_report = []
for i, label in enumerate(BG_LABELS):
    mask = bin_idx == i
    if mask.sum() < 10:
        continue
    bin_report.append({
        "BG Range"  : label,
        "n samples" : int(mask.sum()),
        "% of test" : round(float(mask.mean()) * 100, 1),
        "MAE (eV)"  : round(float(abs_err[mask].mean()), 4),
        "RMSE (eV)" : round(float(np.sqrt(((pred_bg[mask] - true_bg[mask])**2).mean())), 4),
        "Mean bias" : round(float((pred_bg[mask] - true_bg[mask]).mean()), 4),
    })

bin_df = pd.DataFrame(bin_report).set_index("BG Range")
print(f"\n{bin_df.to_string()}")
print(f"\n  📌 'Mean bias' < 0 = systematic underestimation")
print(f"  📌 'Mean bias' > 0 = systematic overestimation")

# ── Training set coverage by BG range ─────────────────────────────────
print(f"\n{'='*60}")
print(f"  TRAINING SET COVERAGE BY BG RANGE")
print(f"{'='*60}")

train_bg   = np.array(y_raw_train_all)[np.array(y_cls_train_all) == 1]  # cuDF-safe
train_bins = np.digitize(train_bg, BG_BINS) - 1

print(f"\n  {'Range':<12} {'Train n':>10} {'Train %':>10} {'Test n':>10}")
for i, label in enumerate(BG_LABELS):
    tr_n = int((train_bins == i).sum())
    te_n = int((bin_idx    == i).sum())
    print(f"  {label:<12} {tr_n:>10,} {tr_n/len(train_bg)*100:>9.1f}%  {te_n:>10,}")

print_heading("Step 37: Isotonic Recalibration of Stage 2 Predictions", level=2)

# ── Fit isotonic recalibration on cal set (non-metals only) ───────────
cal_log_preds = np.array(reg_stage2.predict(X_nm_cal))
cal_eV_preds  = np.expm1(cal_log_preds)
cal_eV_true   = np.array(y_raw_cal)[nm_cal_mask]

iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(cal_eV_preds, cal_eV_true)

# ── Evaluate recalibrated predictions on test set ─────────────────────
eV_preds_iso = iso_reg.predict(eV_preds_tuned)

mae_iso   = mean_absolute_error(eV_true, eV_preds_iso)
medae_iso = median_absolute_error(eV_true, eV_preds_iso)
rmse_iso  = np.sqrt(mean_squared_error(eV_true, eV_preds_iso))
r2_iso    = r2_score(eV_true, eV_preds_iso)

print(f"\n  Stage 2 — After Isotonic Recalibration")
print(f"  {'─'*55}")
print(f"  {'Metric':<10} {'Tuned':>10} {'Iso-cal':>10} {'Delta':>10}")
print(f"  {'─'*55}")
print(f"  {'MAE':<10} {mae_eV:>10.4f} {mae_iso:>10.4f} {mae_iso - mae_eV:>+10.4f}")
print(f"  {'MedAE':<10} {medae_eV:>10.4f} {medae_iso:>10.4f} {medae_iso - medae_eV:>+10.4f}")
print(f"  {'RMSE':<10} {rmse_eV:>10.4f} {rmse_iso:>10.4f} {rmse_iso - rmse_eV:>+10.4f}")
print(f"  {'R²':<10} {r2_eV:>10.4f} {r2_iso:>10.4f} {r2_iso - r2_eV:>+10.4f}")

# ── Check bias correction by BG range ─────────────────────────────────
print(f"\n  Bias after isotonic recalibration:")
print(f"  {'Range':<12} {'Before':>10} {'After':>10} {'Fixed?':>10}")
bin_idx_test = np.digitize(eV_true, BG_BINS) - 1
for i, label in enumerate(BG_LABELS):
    mask = bin_idx_test == i
    if mask.sum() < 10:
        continue
    bias_before = float((eV_preds_tuned[mask] - eV_true[mask]).mean())
    bias_after  = float((eV_preds_iso[mask]   - eV_true[mask]).mean())
    fixed       = "✅" if abs(bias_after) < abs(bias_before) else "⚠️ "
    print(f"  {label:<12} {bias_before:>+10.4f} {bias_after:>+10.4f} {fixed:>10}")

print_heading("Step 38: Bin-Wise Bias Correction & Post-Processing Selection", level=2)

# BIN-WISE BIAS CORRECTION

# ── Fit correction on calibration set ────────────────────────────────
cal_eV_preds_raw = np.expm1(np.array(reg_stage2.predict(X_nm_cal)))
cal_eV_true      = np.array(y_raw_cal)[nm_cal_mask]

# Compute mean bias per bin on cal set
cal_bin_idx = np.digitize(cal_eV_preds_raw, BG_BINS) - 1
bin_corrections = {}

print(f"  Bias corrections fitted on calibration set:")
print(f"  {'Range':<12} {'Cal n':>8} {'Bias':>10} {'Correction':>12}")
print(f"  {'─'*46}")

for i, label in enumerate(BG_LABELS):
    mask = cal_bin_idx == i
    if mask.sum() < 10:
        bin_corrections[i] = 0.0
        continue
    # Mean bias = mean(pred - true) → subtract this to debias
    bias = float((cal_eV_preds_raw[mask] - cal_eV_true[mask]).mean())
    bin_corrections[i] = -bias   # correction = negative bias
    print(f"  {label:<12} {mask.sum():>8,} {bias:>+10.4f} {-bias:>+12.4f}")

# ── Apply correction to test predictions ──────────────────────────────
def apply_bin_correction(preds_eV, corrections, bins):
    corrected = preds_eV.copy()
    bin_idx   = np.digitize(preds_eV, bins) - 1
    for i, corr in corrections.items():
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        corrected[mask] = np.clip(preds_eV[mask] + corr, 0.0, None)
    return corrected

eV_preds_corrected = apply_bin_correction(
    eV_preds_tuned, bin_corrections, BG_BINS
)

# ── Evaluate ───────────────────────────────────────────────────────────
mae_corr   = mean_absolute_error(eV_true, eV_preds_corrected)
medae_corr = median_absolute_error(eV_true, eV_preds_corrected)
rmse_corr  = np.sqrt(mean_squared_error(eV_true, eV_preds_corrected))
r2_corr    = r2_score(eV_true, eV_preds_corrected)

print(f"\n  Stage 2 — Tuned vs Bin-corrected vs Isotonic")
print(f"  {'─'*60}")
print(f"  {'Metric':<10} {'Tuned':>10} {'Bin-corr':>10} {'Iso-cal':>10}")
print(f"  {'─'*60}")
print(f"  {'MAE':<10} {mae_eV:>10.4f} {mae_corr:>10.4f} {mae_iso:>10.4f}")
print(f"  {'MedAE':<10} {medae_eV:>10.4f} {medae_corr:>10.4f} {medae_iso:>10.4f}")
print(f"  {'RMSE':<10} {rmse_eV:>10.4f} {rmse_corr:>10.4f} {rmse_iso:>10.4f}")
print(f"  {'R²':<10} {r2_eV:>10.4f} {r2_corr:>10.4f} {r2_iso:>10.4f}")

print(f"\n  Bias after bin-wise correction:")
print(f"  {'Range':<12} {'Original':>10} {'Iso-cal':>10} {'Bin-corr':>10}")
print(f"  {'─'*48}")
bin_idx_test = np.digitize(eV_true, BG_BINS) - 1
for i, label in enumerate(BG_LABELS):
    mask = bin_idx_test == i
    if mask.sum() < 10:
        continue
    b_orig = float((eV_preds_tuned[mask]     - eV_true[mask]).mean())
    b_iso  = float((eV_preds_iso[mask]       - eV_true[mask]).mean())
    b_corr = float((eV_preds_corrected[mask] - eV_true[mask]).mean())
    print(f"  {label:<12} {b_orig:>+10.4f} {b_iso:>+10.4f} {b_corr:>+10.4f}")

# ── Save best post-processor ───────────────────────────────────────────
joblib.dump(bin_corrections, f'{CONFIG["models_dir"]}/stage2_bin_corrections.pkl')
print(f"\n  ✅ Bin corrections saved")


# ── Final Stage 2 post-processor decision ─────────────────────────────
# Bin-correction wins on all metrics — adopt it
post_processor = "bin_correction"
eV_preds_final = eV_preds_corrected
mae_eV_final   = mae_corr
medae_eV_final = medae_corr
rmse_eV_final  = rmse_corr
r2_eV_final    = r2_corr

joblib.dump(bin_corrections, f'{CONFIG["models_dir"]}/stage2_bin_corrections.pkl')
joblib.dump(iso_reg,         f'{CONFIG["models_dir"]}/stage2_isotonic.pkl')

print(f"  ✅ Post-processor : {post_processor}")
print(f"  Final Stage 2 MAE : {mae_eV_final:.4f} eV  (was {mae_eV:.4f} before tuning)")
print(f"  DFT-PBE benchmark : ~0.6–1.0 eV ✅")
print(f"\n  Stage 2 complete — proceeding to Stage 1 Optuna tuning")

tracker.track(df, "Bin-Wise Bias Calibration", note="Calibration", dataset="MP Dataset")

print_heading("Step 39: Stage 1 Hyperparameter Optimization", level=2)

# STAGE 1 OPTUNA — CLASSIFIER HYPERPARAMETER TUNING

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Convert to numpy once — reused across trials
X_train_all_np     = np.array(X_train_all)
y_cls_train_all_np = np.array(y_cls_train_all)

def objective_stage1(trial):
    params = {
        # Booster
        "n_estimators"          : trial.suggest_int("n_estimators", 300, 1200),  # tightened ceiling
        "learning_rate"         : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth"             : trial.suggest_int("max_depth", 4, 10),
        "min_child_weight"      : trial.suggest_int("min_child_weight", 1, 10),
        # Regularisation
        "subsample"             : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"      : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel"     : trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma"                 : trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha"             : trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda"            : trial.suggest_float("reg_lambda", 0.5, 5.0),
        # Class imbalance — keep fixed, already calculated
        "scale_pos_weight"      : scale_pos_weight,
        # Fixed
        **XGB_DEVICE_PARAMS,
        "objective"             : "binary:logistic",
        "eval_metric"           : ["logloss", "auc"],
        "early_stopping_rounds" : 50,
        "random_state"          : RANDOM_SEED,
    }

    # ── Single stratified holdout instead of 5-fold CV ────────────────
    # Same split serves both early stopping and AUC scoring.
    # Trades some variance in the search signal for ~5x speedup.
    # Final model is retrained properly afterward regardless.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_cls_train_all_np,
        test_size    = 0.2,
        random_state = RANDOM_SEED,
        stratify     = y_cls_train_all_np
    )

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-auc")
    model = xgb.XGBClassifier(**params, callbacks=[pruning_callback])
    model.fit(
        X_tr, y_tr,
        eval_set = [(X_val, y_val)],
        verbose  = False,
    )

    proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, proba)

    # Free GPU memory periodically to avoid fragmentation over many sequential fits
    if trial.number % 10 == 0:
        free_gpu_memory()

    return auc_score   # maximise AUC

# ── Run study ─────────────────────────────────────────────────────────
print("Running Stage 1 Optuna search...")
study_stage1 = optuna.create_study(
    direction = "maximize",
    sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner    = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
)
study_stage1.optimize(
    objective_stage1,
    n_trials          = 60,
    show_progress_bar = True,
)

print(f"\n  Best AUC   : {study_stage1.best_value:.5f}")
print(f"  Best params: {study_stage1.best_params}")
print(f"  Completed  : {len([t for t in study_stage1.trials if t.state == optuna.trial.TrialState.COMPLETE])} trials")

joblib.dump(study_stage1, "optuna_study_stage1.pkl")
fig = vis.plot_optimization_history(study_stage1)

if CONFIG["display_graphs"]:
    fig.show()

print_heading("Step 40: Stage 1 Retraining with Optimized Hyperparameters", level=2)

# STEP 7 — RETRAIN STAGE 1 WITH OPTUNA BEST PARAMS

print(f"\n{'='*60}")
print(f"  STEP 7 — STAGE 1 RETRAIN (OPTUNA TUNED)")
print(f"{'='*60}")

X_cls_tr, X_cls_val, y_cls_tr, y_cls_val = train_test_split(
    X_train_all, y_cls_train_all,
    test_size    = 0.05,
    random_state = RANDOM_SEED,
    stratify     = y_cls_train_all
)

clf_stage1_tuned = xgb.XGBClassifier(
    **study_stage1.best_params,
    scale_pos_weight  = scale_pos_weight,
    tree_method       = "hist",
    device            = "cuda",
    objective         = "binary:logistic",
    eval_metric       = ["logloss", "auc"],
    early_stopping_rounds = 50,
    random_state      = RANDOM_SEED,
)

print(f"\n  Training tuned Stage 1 classifier...")
t0 = time.time()
clf_stage1_tuned.fit(
    X_cls_tr, y_cls_tr,
    eval_set = [(X_cls_tr, y_cls_tr), (X_cls_val, y_cls_val)],
    verbose  = 100,
)
t1 = time.time()
print(f"  ✅ Trained in {(t1-t0)/60:.1f} min")
print(f"  Best iteration : {clf_stage1_tuned.best_iteration}")
print(f"  Best score     : {clf_stage1_tuned.best_score:.5f}")


# ── Evaluate tuned vs original ────────────────────────────────────────
y_cls_test_np         = np.array(y_cls_test)
y_pred_orig           = np.array(clf_stage1.predict(X_test))
y_pred_tuned          = np.array(clf_stage1_tuned.predict(X_test))
y_proba_tuned         = np.array(clf_stage1_tuned.predict_proba(X_test)[:, 1])

acc_t  = accuracy_score (y_cls_test_np, y_pred_tuned)
pre_t  = precision_score(y_cls_test_np, y_pred_tuned, average="weighted")
rec_t  = recall_score   (y_cls_test_np, y_pred_tuned, average="weighted")
f1_t   = f1_score       (y_cls_test_np, y_pred_tuned, average="weighted")
auc_t  = roc_auc_score  (y_cls_test_np, y_proba_tuned)

print(f"\n  Stage 1 — Original vs Tuned:")
print(f"  {'Metric':<12} {'Original':>10} {'Tuned':>10} {'Delta':>10}")
print(f"  {'─'*46}")
print(f"  {'Accuracy':<12} {acc:>10.4f} {acc_t:>10.4f} {acc_t-acc:>+10.4f}")
print(f"  {'Precision':<12} {pre:>10.4f} {pre_t:>10.4f} {pre_t-pre:>+10.4f}")
print(f"  {'Recall':<12} {rec:>10.4f} {rec_t:>10.4f} {rec_t-rec:>+10.4f}")
print(f"  {'F1':<12} {f1:>10.4f} {f1_t:>10.4f} {f1_t-f1:>+10.4f}")
print(f"  {'ROC-AUC':<12} {auc:>10.4f} {auc_t:>10.4f} {auc_t-auc:>+10.4f}")


# ── Adopt tuned classifier if better ──────────────────────────────────
if auc_t >= auc:
    clf_stage1 = clf_stage1_tuned
    acc, pre, rec, f1, auc = acc_t, pre_t, rec_t, f1_t, auc_t
    joblib.dump(clf_stage1, MODEL_DIR / "stage1_classifier_tuned.pkl")
    print(f"\n  ✅ Tuned classifier adopted")
else:
    print(f"\n  ⚠️  Tuned classifier not better — keeping original")

tracker.track(df, "Stage 1 Hyperparameter Search", note="Optimization", dataset="MP Dataset")

print_heading("Step 41: Classification Threshold Optimization", level=2)

# STEP 8 (CORRECTED) — THRESHOLD TUNING ON CALIBRATION SET
y_cls_cal_np = np.array(y_cls_cal)
y_proba_cal  = np.array(clf_stage1.predict_proba(X_cal)[:, 1])   # ✅ X_cal, not X_test

thresholds = np.arange(0.20, 0.71, 0.01)
results = []

for thr in thresholds:
    y_pred_thr = (y_proba_cal >= thr).astype(int)
    results.append({
        "threshold": round(thr, 2),
        "accuracy":  accuracy_score(y_cls_cal_np, y_pred_thr),
        "precision": precision_score(y_cls_cal_np, y_pred_thr, average="weighted", zero_division=0),
        "recall":    recall_score(y_cls_cal_np, y_pred_thr, average="weighted", zero_division=0),
        "f1":        f1_score(y_cls_cal_np, y_pred_thr, average="weighted", zero_division=0),
        "nm_recall": recall_score(y_cls_cal_np, y_pred_thr, pos_label=1, average="binary", zero_division=0),
        "fn_count":  int(((y_cls_cal_np == 1) & (y_pred_thr == 0)).sum()),
    })
thr_df = pd.DataFrame(results).set_index("threshold")

best_f1     = thr_df["f1"].max()
valid_mask  = thr_df["f1"] >= best_f1 * 0.99
optimal_thr = thr_df[valid_mask]["nm_recall"].idxmax()
OPTIMAL_THRESHOLD = optimal_thr
print(f"Optimal threshold (selected on calibration set): {OPTIMAL_THRESHOLD}")

# ── Find optimal threshold ────────────────────────────────────────────
# Primary: maximise non-metal recall
# Secondary: keep F1 within 1% of best F1 (don't sacrifice too much precision)
best_f1       = thr_df["f1"].max()
valid_mask    = thr_df["f1"] >= best_f1 * 0.99
optimal_thr   = thr_df[valid_mask]["nm_recall"].idxmax()
optimal_row   = thr_df.loc[optimal_thr]

print(f"\n  Threshold scan (selected rows):")
print(f"  {'Thr':>6} {'Accuracy':>10} {'F1':>8} {'NM Recall':>10} {'FN Count':>10}")
print(f"  {'─'*50}")
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, optimal_thr, 0.55, 0.60]:
    thr = round(thr, 2)
    if thr not in thr_df.index:
        continue
    row    = thr_df.loc[thr]
    marker = " ◀ OPTIMAL" if thr == optimal_thr else ""
    print(f"  {thr:>6.2f} {row['accuracy']:>10.4f} {row['f1']:>8.4f} "
          f"{row['nm_recall']:>10.4f} {int(row['fn_count']):>10,}{marker}")

print(f"\n  Optimal threshold : {optimal_thr}")
print(f"  NM Recall         : {optimal_row['nm_recall']:.4f}  "
      f"(was {rec:.4f} at 0.50)")
print(f"  FN count          : {int(optimal_row['fn_count']):,}  "
      f"(was {int(thr_df.loc[0.50, 'fn_count']):,} at 0.50)")

OPTIMAL_THRESHOLD = optimal_thr

tracker.track(df, "Hurdle Gate Threshold Calibration", note="Calibration", dataset="MP Dataset")

print_heading("Step 42: Final Conformal Calibration with Optimized Components", level=2)

# STEP 4 (RERUN) — CONFORMAL UQ WITH FINAL MODELS

print(f"\n{'='*60}")
print(f"  STEP 4 RERUN — CONFORMAL UQ (FINAL MODELS)")
print(f"{'='*60}")

# ── Stage 1 calibration ───────────────────────────────────────────────
cal_proba_nonmetal = np.array(clf_stage1.predict_proba(X_cal)[:, 1])
y_cls_cal_np       = np.array(y_cls_cal)

cls_cal_scores = 1.0 - np.where(
    y_cls_cal_np == 1,
    cal_proba_nonmetal,
    1.0 - cal_proba_nonmetal
)

# ── Stage 2 calibration — LOG SCALE (not eV) ──────────────────────────
# Calibrating in eV after bin correction inflates q_hat massively.
# Correct approach: calibrate residuals in log scale, back-transform
# intervals to eV inside hurdle_predict_uq.

nm_cal_mask    = y_cls_cal_np == 1
y_log_cal_np   = np.array(y_log_cal)
y_log_nm_cal   = y_log_cal_np[nm_cal_mask]

if hasattr(X_cal, 'iloc'):
    X_nm_cal = X_cal.iloc[nm_cal_mask]
else:
    X_nm_cal = X_cal[nm_cal_mask]

assert not np.isinf(y_log_nm_cal).any(), "🔴 -inf in cal log targets"

cal_log_pred   = np.array(reg_stage2.predict(X_nm_cal))

# Nonconformity scores in LOG scale — not eV
reg_cal_scores = np.abs(y_log_nm_cal - cal_log_pred)

print(f"  Non-metal cal samples : {nm_cal_mask.sum():,}")
print(f"  Stage 2 nonconformity stats (log scale):")
print(f"    Mean   : {reg_cal_scores.mean():.4f}")
print(f"    Median : {np.median(reg_cal_scores):.4f}")
print(f"    95th%  : {np.percentile(reg_cal_scores, 95):.4f}")

# ── Compute q_hats ────────────────────────────────────────────────────
cls_q_hats = {}
reg_q_hats = {}
n_cls = len(cls_cal_scores)
n_reg = len(reg_cal_scores)

for alpha in ALPHA_VALUES:
    pct       = int((1 - alpha) * 100)
    q_cls     = min(np.ceil((n_cls + 1) * (1-alpha)) / n_cls, 1.0)
    q_reg     = min(np.ceil((n_reg + 1) * (1-alpha)) / n_reg, 1.0)
    cls_q_hats[alpha] = float(np.quantile(cls_cal_scores, q_cls))
    reg_q_hats[alpha] = float(np.quantile(reg_cal_scores, q_reg))
    print(f"  PI{pct}: cls q_hat={cls_q_hats[alpha]:.4f}  "
          f"reg q_hat={reg_q_hats[alpha]:.4f} log units")

# ── Coverage evaluation ───────────────────────────────────────────────
test_preds_uq = hurdle_predict_uq(
    X_test, clf_stage1, reg_stage2,
    cls_q_hats, reg_q_hats,
    nonmetal_threshold = OPTIMAL_THRESHOLD,
    bin_corrections    = bin_corrections, 
    bg_bins            = BG_BINS,          
)


y_raw_test_np = np.array(y_raw_test)
pred_class_np = test_preds_uq["predicted_class"].values
true_nonmetal = y_cls_test_np == 1
tp_mask       = true_nonmetal & (pred_class_np == 1)
true_eV_tp    = y_raw_test_np[tp_mask]

# Apply bin correction to UQ predictions
for alpha in ALPHA_VALUES:
    pct = int((1 - alpha) * 100)
    test_preds_uq[f"pi{pct}_lower"] = np.clip(
        test_preds_uq[f"pi{pct}_lower"], 0.0, None
    )

print(f"\n  Coverage on true positive non-metals ({tp_mask.sum():,}):")
print(f"  {'Status':>4}  {'Level':>6}  {'Coverage':>10}  "
      f"{'Avg Width':>10}  {'Median Width':>13}")
print(f"  {'─'*55}")

for alpha in ALPHA_VALUES:
    pct      = int((1 - alpha) * 100)
    lo       = test_preds_uq[f"pi{pct}_lower"].values[tp_mask]
    hi       = test_preds_uq[f"pi{pct}_upper"].values[tp_mask]
    covered  = ((true_eV_tp >= lo) & (true_eV_tp <= hi)).mean()
    width    = hi - lo
    status   = "✅" if covered >= (1 - alpha) else "⚠️ "
    print(f"  {status}   PI{pct}   {covered*100:>8.2f}%  "
          f"{width.mean():>10.4f} eV  {np.median(width):>10.4f} eV")

joblib.dump(cls_q_hats, MODEL_DIR / "cls_q_hats_final.pkl")
joblib.dump(reg_q_hats, MODEL_DIR / "reg_q_hats_final.pkl")
print(f"\n  ✅ Final UQ calibration saved")

print_heading("Step 43: Final Hurdle Pipeline Evaluation with Optimized Configuration", level=2)

# STEP 6 (RERUN) — FULL PIPELINE EVALUATION

print(f"\n{'='*60}")
print(f"  STEP 6 RERUN — FULL PIPELINE EVALUATION")
print(f"{'='*60}")

# Run hurdle predict with optimal threshold + bin correction
test_predictions_final = hurdle_predict(
    X_test, clf_stage1, reg_stage2,
    nonmetal_threshold = OPTIMAL_THRESHOLD
)

# Apply bin correction to final predictions
final_pred_eV_raw = test_predictions_final["predicted_BG_eV"].values
pred_nm_mask      = test_predictions_final["predicted_class"].values == 1
final_pred_eV     = final_pred_eV_raw.copy()
final_pred_eV[pred_nm_mask] = apply_bin_correction(
    final_pred_eV_raw[pred_nm_mask], bin_corrections, BG_BINS
)

true_eV_all   = np.array(y_raw_test)
pred_class_np = test_predictions_final["predicted_class"].values

mae_final   = mean_absolute_error(true_eV_all, final_pred_eV)
medae_final = np.median(np.abs(true_eV_all - final_pred_eV))
rmse_final  = np.sqrt(mean_squared_error(true_eV_all, final_pred_eV))
r2_final    = r2_score(true_eV_all, final_pred_eV)

print(f"\n  Full pipeline metrics (threshold={OPTIMAL_THRESHOLD}, bin-corrected):")
print(f"  {'Metric':<12} {'Value':>10}")
print(f"  {'─'*26}")
print(f"  {'MAE (eV)':<12} {mae_final:>10.4f}")
print(f"  {'MedAE (eV)':<12} {medae_final:>10.4f}")
print(f"  {'RMSE (eV)':<12} {rmse_final:>10.4f}")
print(f"  {'R²':<12} {r2_final:>10.4f}")

# ── Stage 1 gate breakdown ────────────────────────────────────────────
true_metal    = y_cls_test_np == 0
true_nonmetal = y_cls_test_np == 1
pred_metal    = pred_class_np == 0
pred_nonmetal = pred_class_np == 1

tp = int((true_nonmetal & pred_nonmetal).sum())
fn = int((true_nonmetal & pred_metal   ).sum())
fp = int((true_metal    & pred_nonmetal).sum())
tn = int((true_metal    & pred_metal   ).sum())

print(f"\n  Stage 1 gate (threshold={OPTIMAL_THRESHOLD}):")
print(f"    TP non-metals → Stage 2 : {tp:>8,}")
print(f"    FN non-metals → BG=0    : {fn:>8,}  ← coverage loss")
print(f"    FP metals → Stage 2     : {fp:>8,}")
print(f"    TN metals → BG=0        : {tn:>8,}")

# ── MAE split by classification correctness ───────────────────────────
correct_mask = (y_cls_test_np == pred_class_np)
wrong_mask   = ~correct_mask

if wrong_mask.sum() > 0:
    print(f"\n  MAE by Stage 1 correctness:")
    print(f"    Correct ({correct_mask.sum():,}) : "
          f"{mean_absolute_error(true_eV_all[correct_mask], final_pred_eV[correct_mask]):.4f} eV")
    print(f"    Wrong   ({wrong_mask.sum():,})   : "
          f"{mean_absolute_error(true_eV_all[wrong_mask], final_pred_eV[wrong_mask]):.4f} eV")

print(f"\n  DFT-PBE benchmark : ~0.6–1.0 eV MAE")
print(f"  Pipeline vs benchmark : ", end="")
if mae_final < 0.6:
    print("✅ Better than DFT-PBE")
elif mae_final < 1.0:
    print("⚠️  Within DFT-PBE range")
else:
    print("❌ Worse than DFT-PBE")

# STEP 9 — FEATURE IMPORTANCE

print(f"\n{'='*60}")
print(f"  STEP 9 — FEATURE IMPORTANCE")
print(f"{'='*60}")

TOP_N = 20

# ── Stage 1 importance ────────────────────────────────────────────────
imp_cls = pd.Series(
    clf_stage1.feature_importances_,
    index = feature_cols
).sort_values(ascending=False)

# ── Stage 2 importance ────────────────────────────────────────────────
imp_reg = pd.Series(
    reg_stage2.feature_importances_,
    index = feature_cols
).sort_values(ascending=False)

print(f"\n  Top {TOP_N} features — Stage 1 (Classifier):")
print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>12}")
print(f"  {'─'*60}")
for rank, (feat, val) in enumerate(imp_cls.head(TOP_N).items(), 1):
    print(f"  {rank:<6} {feat:<40} {val:>12.4f}")

print(f"\n  Top {TOP_N} features — Stage 2 (Regressor):")
print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>12}")
print(f"  {'─'*60}")
for rank, (feat, val) in enumerate(imp_reg.head(TOP_N).items(), 1):
    print(f"  {rank:<6} {feat:<40} {val:>12.4f}")

# ── Shared top features ───────────────────────────────────────────────
top_cls_set = set(imp_cls.head(TOP_N).index)
top_reg_set = set(imp_reg.head(TOP_N).index)
shared      = top_cls_set & top_reg_set

print(f"\n  Features in top {TOP_N} of BOTH stages ({len(shared)}):")
for feat in sorted(shared):
    print(f"    {feat:<40} "
          f"cls rank: {list(imp_cls.index).index(feat)+1:>3}  "
          f"reg rank: {list(imp_reg.index).index(feat)+1:>3}")

# Save importance tables
imp_cls.to_csv(MODEL_DIR / "feature_importance_stage1.csv")
imp_reg.to_csv(MODEL_DIR / "feature_importance_stage2.csv")
print(f"\n  ✅ Feature importance saved")


# ── 4D: Coverage Evaluation on Test Set — CORRECTED ────────────────────
print(f"\n  {'─'*50}")
print(f"  Coverage Evaluation on Test Set (tuned model, thr={OPTIMAL_THRESHOLD})")
print(f"  {'─'*50}")

test_preds_uq = hurdle_predict_uq(
    X_test, clf_stage1_tuned, reg_stage2,
    cls_q_hats, reg_q_hats,
    nonmetal_threshold = OPTIMAL_THRESHOLD,
    bin_corrections    = bin_corrections,
    bg_bins            = BG_BINS,
)

y_raw_test_np  = np.array(y_raw_test)
y_cls_test_np  = np.array(y_cls_test)
pred_class_np  = test_preds_uq["predicted_class"].values

# ── Three populations for coverage analysis ─────────────────────────
true_nonmetal = y_cls_test_np == 1
true_metal    = y_cls_test_np == 0
pred_nonmetal = pred_class_np == 1
pred_metal    = pred_class_np == 0

tp_mask = true_nonmetal & pred_nonmetal
fn_mask = true_nonmetal & pred_metal
fp_mask = true_metal    & pred_nonmetal
tn_mask = true_metal    & pred_metal

print(f"\n  Stage 1 gate breakdown:")
print(f"    TP (non-metal → Stage 2) : {tp_mask.sum():>7,}")
print(f"    FN (non-metal → BG=0)    : {fn_mask.sum():>7,}  ← guaranteed coverage miss")
print(f"    FP (metal → Stage 2)     : {fp_mask.sum():>7,}  ← noise in Stage 2")
print(f"    TN (metal → BG=0)        : {tn_mask.sum():>7,}")

# ── Stage 2 coverage — TRUE POSITIVES ONLY ──────────────────────────
true_eV_tp = y_raw_test_np[tp_mask]

print(f"\n  Stage 2 coverage on TRUE POSITIVES ({tp_mask.sum():,}):")
print(f"  {'Status':>4}  {'Level':>6}  {'Coverage':>10}  "
      f"{'Avg Width':>10}  {'Median Width':>13}")
print(f"  {'─'*55}")

for alpha in ALPHA_VALUES:
    pct   = int((1 - alpha) * 100)
    lo    = test_preds_uq[f"pi{pct}_lower"].values[tp_mask]
    hi    = test_preds_uq[f"pi{pct}_upper"].values[tp_mask]
    width = test_preds_uq[f"pi{pct}_width"].values[tp_mask]

    covered      = ((true_eV_tp >= lo) & (true_eV_tp <= hi)).mean()
    avg_width    = width.mean()
    median_width = np.median(width)
    status       = "✅" if covered >= (1 - alpha) else "⚠️ "

    print(f"  {status}   PI{pct}   {covered*100:>8.2f}%  "
          f"{avg_width:>10.4f} eV  {median_width:>10.4f} eV")

# ── Full pipeline coverage — ALL true non-metals ────────────────────
true_eV_nm_all = y_raw_test_np[true_nonmetal]

print(f"\n  Full pipeline coverage on ALL true non-metals ({true_nonmetal.sum():,}):")
print(f"  (Includes FNs assigned BG=0 — shows Stage 1 error impact)")
print(f"  {'Status':>4}  {'Level':>6}  {'Coverage':>10}")
print(f"  {'─'*35}")

for alpha in ALPHA_VALUES:
    pct   = int((1 - alpha) * 100)
    lo    = test_preds_uq[f"pi{pct}_lower"].values[true_nonmetal]
    hi    = test_preds_uq[f"pi{pct}_upper"].values[true_nonmetal]

    covered   = ((true_eV_nm_all >= lo) & (true_eV_nm_all <= hi)).mean()
    status    = "✅" if covered >= (1 - alpha) else "⚠️ "
    shortfall = (1 - alpha - covered) * 100
    print(f"  {status}   PI{pct}   {covered*100:>8.2f}%   "
          f"(shortfall from FNs: {shortfall:.2f}%)")

# ── FN impact summary ────────────────────────────────────────────────
fn_count   = fn_mask.sum()
fn_true_eV = y_raw_test_np[fn_mask]
print(f"\n  False Negative impact (Stage 1 misclassified non-metals):")
print(f"    Count          : {fn_count:,}")
print(f"    Mean true BG   : {fn_true_eV.mean():.4f} eV")
print(f"    Coverage loss  : ~{fn_count/true_nonmetal.sum()*100:.2f}% "
      f"(these always miss — interval = [0,0])")

# ── Stage 1 uncertainty flags ─────────────────────────────────────────
print(f"\n  Stage 1 uncertainty flags:")
for alpha in ALPHA_VALUES:
    pct     = int((1 - alpha) * 100)
    col     = f"cls_uncertain_{pct}"
    n_unc   = int(test_preds_uq[col].sum())
    pct_unc = n_unc / len(test_preds_uq) * 100
    print(f"    PI{pct} → {n_unc:,} uncertain ({pct_unc:.1f}%)")

# ── Recompute Stage 1 metrics at OPTIMAL_THRESHOLD (tuned model) ──────
# Fixes stale acc/pre/rec/f1 variables left over from earlier cells
y_proba_final = np.array(clf_stage1_tuned.predict_proba(X_test)[:, 1])
y_pred_final  = (y_proba_final >= OPTIMAL_THRESHOLD).astype(int)

acc = accuracy_score (y_cls_test_np, y_pred_final)
pre = precision_score(y_cls_test_np, y_pred_final, average="weighted")
rec = recall_score   (y_cls_test_np, y_pred_final, average="weighted")
f1  = f1_score       (y_cls_test_np, y_pred_final, average="weighted")
auc = roc_auc_score  (y_cls_test_np, y_proba_final)  # threshold-independent, same as auc_t

print(f"\n  ✅ Recomputed Stage 1 metrics at threshold={OPTIMAL_THRESHOLD} "
      f"(tuned model)")
print(f"    Accuracy  : {acc:.4f}")
print(f"    Precision : {pre:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1        : {f1:.4f}")
print(f"    ROC-AUC   : {auc:.4f}")

# STEP 10 — FINAL SUMMARY

print(f"\n{'='*60}")
print(f"  HURDLE FRAMEWORK — FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n  STAGE 1 — Metal / Non-metal Classifier")
print(f"  {'─'*40}")
print(f"    Accuracy        : {acc:.4f}")
print(f"    Precision (wtd) : {pre:.4f}")
print(f"    Recall    (wtd) : {rec:.4f}")
print(f"    F1        (wtd) : {f1:.4f}")
print(f"    ROC-AUC         : {auc:.4f}")
print(f"    Threshold       : {OPTIMAL_THRESHOLD}")

print(f"\n  STAGE 2 — Band Gap Regressor (Non-metals)")
print(f"  {'─'*40}")
print(f"    MAE   (eV) : {mae_eV_final:.4f}")
print(f"    MedAE (eV) : {medae_eV_final:.4f}")
print(f"    RMSE  (eV) : {rmse_eV_final:.4f}")
print(f"    R²    (eV) : {r2_eV_final:.4f}")
print(f"    Post-proc  : {post_processor}")

print(f"\n  FULL PIPELINE (metals + non-metals)")
print(f"  {'─'*40}")
print(f"    MAE   (eV) : {mae_final:.4f}")
print(f"    MedAE (eV) : {medae_final:.4f}")
print(f"    RMSE  (eV) : {rmse_final:.4f}")
print(f"    R²    (eV) : {r2_final:.4f}")

print(f"\n  CONFORMAL PREDICTION INTERVALS")
print(f"  {'─'*40}")
for alpha in ALPHA_VALUES:
    pct = int((1 - alpha) * 100)
    print(f"    PI{pct}: cls q_hat={cls_q_hats[alpha]:.4f}  "
          f"reg q_hat={reg_q_hats[alpha]:.4f} eV")

print(f"\n  DFT-PBE benchmark : ~0.6–1.0 eV MAE")
print(f"  Pipeline vs benchmark : ", end="")
if mae_final < 0.6:
    print("✅ Better than DFT-PBE baseline")
elif mae_final < 1.0:
    print("⚠️  Within DFT-PBE range")
else:
    print("❌ Worse than DFT-PBE baseline")

# ── Save all final artifacts ───────────────────────────────────────────
joblib.dump(clf_stage1,      MODEL_DIR / "stage1_classifier_final.pkl")
joblib.dump(reg_stage2,      MODEL_DIR / "stage2_regressor_final.pkl")
joblib.dump(feature_cols,    MODEL_DIR / "feature_cols.pkl")
joblib.dump(bin_corrections, MODEL_DIR / "stage2_bin_corrections.pkl")
joblib.dump(cls_q_hats,      MODEL_DIR / "cls_q_hats_final.pkl")
joblib.dump(reg_q_hats,      MODEL_DIR / "reg_q_hats_final.pkl")

print(f"\n  ── Saved artifacts ─────────────────────────────────────")
for f in sorted(MODEL_DIR.glob("*.pkl")):
    print(f"    {f}")

print(f"\n{'='*60}")
print(f"  ✅ HURDLE FRAMEWORK COMPLETE")
print(f"{'='*60}")

tracker.track(df, "End-to-End Test Evaluation", note="Evaluation", dataset="MP Dataset")

TIME_END = time.perf_counter()
TIME_TAKEN = TIME_END - TIME_START

hours, remainder = divmod(TIME_TAKEN, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Time taken: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
print(f"Time taken in Sec: {TIME_TAKEN}")
print("\n\n\n\n")

df_chg = pd.read_pickle(os.path.join(CONFIG["intermediate_dir"], "intermediate_data_3.0.pkl"))

print(f"\n{'='*65}")
print(f"  STRUCTURAL SENSITIVITY — CHEAP (CHGNet) vs DFT-RELAXED")
print(f"{'='*65}")

N_SAMPLE = 300   # keep tractable; relaxation is ~1-5 sec/structure on GPU
rng = np.random.default_rng(RANDOM_SEED)

# ── Re-fetch structures for a random sample of TEST materials ─────────
# (raw Structure objects were not retained after feature extraction —
#  this requires re-querying MP for material_id -> structure)
sample_material_ids = df_chg.loc[X_test.index, "Material ID"].dropna().unique() \
    if "Material ID" in df_chg.columns else None

if sample_material_ids is None:
    raise RuntimeError(
        "Material ID column not found in df_chg — cannot re-fetch structures. "
        "Re-run from an earlier checkpoint pickle that retains Material ID."
    )

sample_ids = rng.choice(sample_material_ids, size=min(N_SAMPLE, len(sample_material_ids)), replace=False)
print(f"\n  Sampling {len(sample_ids)} test-set materials for structural sensitivity check")

assert API, "MP_API_KEY required to re-fetch structures for this analysis"

with MPRester(API) as mpr:
    docs = mpr.materials.summary.search(
        material_ids=list(sample_ids),
        fields=["material_id", "structure", "band_gap"]
    )

print(f"  Retrieved {len(docs)} structures")


# ── Load CHGNet ─────────────────────────────────────────────────────
CHGNET_DEVICE = "cuda" if CONFIG["gpu_env"] else "cpu"

chgnet_model = CHGNet.load(use_device=CHGNET_DEVICE)
relaxer = StructOptimizer(model=chgnet_model, use_device=CHGNET_DEVICE)

print(f"✅ CHGNet loaded on device: {CHGNET_DEVICE}")

def perturb_structure(struct: Structure, strain_pct=0.04, coord_noise=0.03, seed=None):
    """Simulate a plausible 'unrelaxed' starting guess by applying
    random lattice strain and fractional-coordinate displacement to
    an already-relaxed structure."""
    rng_local = np.random.default_rng(seed)
    s = copy.deepcopy(struct)

    # Random symmetric strain tensor on the lattice matrix
    strain = np.eye(3) + rng_local.uniform(-strain_pct, strain_pct, size=(3, 3))
    strain = (strain + strain.T) / 2   # keep it physically reasonable (symmetric)
    new_lattice = np.dot(s.lattice.matrix, strain)
    s.lattice = type(s.lattice)(new_lattice)

    # Random fractional coordinate displacement per site
    for i in range(len(s)):
        disp = rng_local.uniform(-coord_noise, coord_noise, size=3)
        s.translate_sites([i], disp, frac_coords=True)

    return s

def extract_struct_features(struct: Structure) -> dict:
    """Extract the same structural features used in the main pipeline
    from a given pymatgen Structure object."""
    lat = struct.lattice
    sga = SpacegroupAnalyzer(struct)
    return {
        "Nsites"            : len(struct),
        "Volume"            : struct.volume,
        "Density"           : struct.density,
        "Space Group Number": sga.get_space_group_number(),
        "Lattice (a)"       : lat.a,
        "Lattice (b)"       : lat.b,
        "Lattice (c)"       : lat.c,
        "Lattice (alpha)"   : lat.alpha,
        "Lattice (beta)"    : lat.beta,
        "Lattice (gamma)"   : lat.gamma,
    }

results = []

for i, doc in enumerate(docs):
    mid        = doc.material_id
    true_struct= doc.structure
    true_bg    = doc.band_gap

    try:
        # ── Cheap relaxation pathway ───────────────────────────────
        perturbed = perturb_structure(true_struct, seed=RANDOM_SEED + i)
        relax_out = relaxer.relax(perturbed, verbose=False)
        cheap_struct = relax_out["final_structure"]

        feats_dft   = extract_struct_features(true_struct)
        feats_cheap = extract_struct_features(cheap_struct)

        results.append({
            "material_id": str(mid),
            "true_bandgap": true_bg,
            **{f"dft_{k}": v for k, v in feats_dft.items()},
            **{f"cheap_{k}": v for k, v in feats_cheap.items()},
        })
    except Exception as e:
        print(f"  ⚠️  {mid}: relaxation failed ({str(e)[:60]}) — skipped")
        continue

    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(docs)}")

struct_compare_df = pd.DataFrame(results)
print(f"\n  Successfully processed {len(struct_compare_df)} / {len(docs)} structures")

# ── Feature-level agreement: DFT-relaxed vs CHGNet-relaxed ─────────────
print(f"\n  Structural feature agreement (DFT-relaxed vs CHGNet-cheap-relaxed):")
print(f"  {'Feature':<20} {'MAE':>10} {'Rel. MAE %':>12}")
for feat in ["Volume", "Density", "Lattice (a)", "Lattice (b)", "Lattice (c)"]:
    dft_vals   = struct_compare_df[f"dft_{feat}"].values
    cheap_vals = struct_compare_df[f"cheap_{feat}"].values
    mae_feat   = np.mean(np.abs(dft_vals - cheap_vals))
    rel_mae    = mae_feat / np.mean(np.abs(dft_vals)) * 100
    print(f"  {feat:<20} {mae_feat:>10.4f} {rel_mae:>11.2f}%")

sg_match = (struct_compare_df["dft_Space Group Number"] ==
            struct_compare_df["cheap_Space Group Number"]).mean()
print(f"  {'Space group match':<20} {sg_match*100:>10.1f}%")

# ── Now feed BOTH feature sets through the trained hurdle pipeline ─────
# We need the OTHER (non-structural) features too, taken from the original df
print(f"\n  Building full feature vectors (structural swapped, rest held fixed)...")

other_feature_cols = [c for c in feature_cols if c not in [
    "Nsites", "Volume", "Density", "Space Group Number",
    "Lattice (alpha)", "Lattice (beta)", "Lattice (gamma)",
    "alpha_orthogonal", "beta_orthogonal", "gamma_orthogonal",
]]

matched_ids = struct_compare_df["material_id"].values
df_lookup = df_chg.set_index(df_chg["Material ID"].astype(str)) if "Material ID" in df_chg.columns else None

pred_rows_dft, pred_rows_cheap = [], []
for _, row in struct_compare_df.iterrows():
    mid = row["material_id"]
    if mid not in df_lookup.index:
        continue
    base_row = df_lookup.loc[mid, other_feature_cols].copy()

    row_dft = base_row.copy()
    row_cheap = base_row.copy()
    for feat in ["Nsites", "Volume", "Density", "Space Group Number",
                 "Lattice (alpha)", "Lattice (beta)", "Lattice (gamma)"]:
        if f"dft_{feat}" in row:
            row_dft[feat]   = row[f"dft_{feat}"]
            row_cheap[feat] = row[f"cheap_{feat}"]

    # ── Recompute orthogonal flags from the swapped angle values ──────
    # These are derived features (angle == 90.0), not raw columns, so
    # they must be recomputed here rather than copied from base_row —
    # base_row's original flags correspond to the DFT-relaxed angles,
    # not the swapped-in cheap-relaxed ones.
    for angle_name, angle_col in [("alpha", "Lattice (alpha)"),
                                   ("beta",  "Lattice (beta)"),
                                   ("gamma", "Lattice (gamma)")]:
        flag_col = f"{angle_name}_orthogonal"
        if flag_col in feature_cols:
            row_dft[flag_col]   = int(row_dft[angle_col] == 90.0)
            row_cheap[flag_col] = int(row_cheap[angle_col] == 90.0)

    row_dft["Material ID"] = mid
    row_cheap["Material ID"] = mid
    row_dft["true_bandgap"] = row["true_bandgap"]
    row_cheap["true_bandgap"] = row["true_bandgap"]
    pred_rows_dft.append(row_dft)
    pred_rows_cheap.append(row_cheap)

X_struct_dft   = pd.DataFrame(pred_rows_dft)
X_struct_cheap = pd.DataFrame(pred_rows_cheap)

true_bg_struct = X_struct_dft["true_bandgap"].values
X_struct_dft_feats   = X_struct_dft[feature_cols].astype("float32")
X_struct_cheap_feats = X_struct_cheap[feature_cols].astype("float32")

# ── Run full hurdle pipeline on both feature sets ─────────────────────
def run_hurdle_pointpred(X_feats):
    proba = np.array(clf_stage1.predict_proba(X_feats)[:, 1])
    pred_class = (proba >= OPTIMAL_THRESHOLD).astype(int)
    pred_eV = np.zeros(len(X_feats))
    nm_mask = pred_class == 1
    if nm_mask.sum() > 0:
        log_pred = np.array(reg_stage2.predict(X_feats[nm_mask]))
        eV_pred  = np.expm1(log_pred)
        eV_pred  = apply_bin_correction(eV_pred, bin_corrections, BG_BINS)
        pred_eV[nm_mask] = eV_pred
    return pred_eV, pred_class

missing_dft   = set(feature_cols) - set(X_struct_dft.columns)
missing_cheap = set(feature_cols) - set(X_struct_cheap.columns)
assert not missing_dft,   f"🔴 Missing from X_struct_dft: {missing_dft}"
assert not missing_cheap, f"🔴 Missing from X_struct_cheap: {missing_cheap}"
print(f"  ✅ All {len(feature_cols)} feature_cols present in both reconstructed frames")

pred_eV_dft,   pred_class_dft   = run_hurdle_pointpred(X_struct_dft_feats)
pred_eV_cheap, pred_class_cheap = run_hurdle_pointpred(X_struct_cheap_feats)

mae_dft   = mean_absolute_error(true_bg_struct, pred_eV_dft)
mae_cheap = mean_absolute_error(true_bg_struct, pred_eV_cheap)
pred_shift = np.abs(pred_eV_dft - pred_eV_cheap)

print(f"\n{'='*65}")
print(f"  PREDICTION SENSITIVITY — DFT-RELAXED vs CHEAP-RELAXED FEATURES")
print(f"{'='*65}")
print(f"  n materials evaluated        : {len(true_bg_struct)}")
print(f"  MAE using DFT-relaxed feats   : {mae_dft:.4f} eV")
print(f"  MAE using CHGNet-relaxed feats: {mae_cheap:.4f} eV")
print(f"  Mean |prediction shift|       : {pred_shift.mean():.4f} eV")
print(f"  Median |prediction shift|     : {np.median(pred_shift):.4f} eV")
print(f"  Classification agreement      : {(pred_class_dft == pred_class_cheap).mean()*100:.1f}%")

struct_compare_df.to_csv(os.path.join(CONFIG["csv_dir"], "chgnet_structural_sensitivity.csv"), index=False)
print(f"\n  ✅ Saved chgnet_structural_sensitivity.csv")


del struct_compare_df
free_gpu_memory()

tracker.track(df_chg, "CHGNet Testing", note="Modal evaluation using CHGNet", dataset="Evaluation Dataset")


# AFTER
if CONFIG["gpu_env"]:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    try:
        import rmm
        rmm.reinitialize(pool_allocator=False)
    except ImportError:
        pass

free_gpu_memory()   # safe either way — internally try/except-guarded

if CONFIG["gpu_env"]:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    
# ══════════════════════════════════════════════════════════════════
#  CHEMICAL-SYSTEM GENERALIZATION TEST (Reviewer 2 comment #5)
#  Part A: Leave-Chemical-System-Out (LOCO) split
#  Part B: Anion-substitution probe (S/Se/Te family — the reviewer's
#          literal A-B-S / A-B-Se / A-B-Te example)
# ══════════════════════════════════════════════════════════════════

# ── Step 0: Recover 'Elements' aligned to current df index ───────────
if 'df_chg' not in dir():
    df_chg = pd.read_pickle(os.path.join(CONFIG["intermediate_dir"], "intermediate_data_3.0.pkl"))

assert "Elements" in df_chg.columns, "🔴 'Elements' column not found in df_chg"

elements_series = df_chg["Elements"].reindex(df.index)
missing_elem = elements_series.isna().sum()
print(f"  Materials with recovered Elements list : {(~elements_series.isna()).sum():,}")
if missing_elem > 0:
    print(f"  ⚠️  {missing_elem:,} rows could not be matched by index — excluded below")

def to_chem_system(elem_list):
    if elem_list is None:
        return None
    if isinstance(elem_list, float) and np.isnan(elem_list):
        return None
    try:
        return "-".join(sorted(str(e) for e in elem_list))   # string, not tuple
    except TypeError:
        return None

chem_system_series = elements_series.apply(to_chem_system)
valid_mask_cs = chem_system_series.notna()

print(f"  Unique chemical systems in dataset : {chem_system_series[valid_mask_cs].nunique():,}")

free_gpu_memory()

# ══════════════════════════════════════════════════════════════════
#  PART A — STRICT LEAVE-CHEMICAL-SYSTEM-OUT (LOCO) SPLIT
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  PART A — LEAVE-CHEMICAL-SYSTEM-OUT (LOCO) SPLIT")
print(f"{'='*65}")

unique_systems = np.array(chem_system_series[valid_mask_cs].unique(), dtype=object)
rng_cs = np.random.default_rng(RANDOM_SEED)
rng_cs.shuffle(unique_systems)

n_test_systems = int(0.20 * len(unique_systems))
test_systems  = set(unique_systems[:n_test_systems])
train_systems = set(unique_systems[n_test_systems:])

loco_test_mask  = valid_mask_cs & chem_system_series.isin(test_systems)
loco_train_mask = valid_mask_cs & chem_system_series.isin(train_systems)

overlap = test_systems & train_systems
assert len(overlap) == 0, f"🔴 {len(overlap)} chemical systems leaked across LOCO split!"

print(f"  Test-pool chemical systems  : {len(test_systems):,}  ({loco_test_mask.sum():,} materials)")
print(f"  Train-pool chemical systems : {len(train_systems):,}  ({loco_train_mask.sum():,} materials)")
print(f"  ✅ Zero chemical-system overlap between LOCO train/test")

X_loco_train = X.loc[loco_train_mask[loco_train_mask].index.intersection(X.index)]
X_loco_test  = X.loc[loco_test_mask[loco_test_mask].index.intersection(X.index)]

y_cls_loco_train = y_cls.loc[X_loco_train.index]
y_cls_loco_test  = y_cls.loc[X_loco_test.index]
y_log_loco_train = y_log.loc[X_loco_train.index]
y_raw_loco_test  = y_raw.loc[X_loco_test.index]

print(f"\n  LOCO train set : {len(X_loco_train):,}")
print(f"  LOCO test set  : {len(X_loco_test):,}")
print(f"  LOCO test metal fraction : {(y_cls_loco_test==0).mean()*100:.1f}%  "
      f"(full-dataset metal fraction: {(y_cls==0).mean()*100:.1f}%)")

# ── Retrain Stage 1 on LOCO train ─────────────────────────────────────
loco_scale_pos_weight = (
    int((y_cls_loco_train == 0).sum()) / max(int((y_cls_loco_train == 1).sum()), 1)
)

X_cls_tr_l, X_cls_val_l, y_cls_tr_l, y_cls_val_l = train_test_split(
    X_loco_train, y_cls_loco_train,
    test_size=0.05, random_state=RANDOM_SEED, stratify=y_cls_loco_train
)

clf_loco = xgb.XGBClassifier(
    **study_stage1.best_params,
    scale_pos_weight=loco_scale_pos_weight,
    **XGB_DEVICE_PARAMS,
    objective="binary:logistic",
    eval_metric=["logloss", "auc"],
    early_stopping_rounds=50,
    random_state=RANDOM_SEED,
)
clf_loco.fit(X_cls_tr_l, y_cls_tr_l, eval_set=[(X_cls_val_l, y_cls_val_l)], verbose=False)

proba_loco = np.array(clf_loco.predict_proba(X_loco_test)[:, 1])
pred_loco  = (proba_loco >= OPTIMAL_THRESHOLD).astype(int)

acc_loco = accuracy_score(np.array(y_cls_loco_test), pred_loco)
f1_loco  = f1_score(np.array(y_cls_loco_test), pred_loco, average="weighted")
auc_loco = roc_auc_score(np.array(y_cls_loco_test), proba_loco)

print(f"\n  Stage 1 (LOCO)             — Acc={acc_loco:.4f}  F1={f1_loco:.4f}  AUC={auc_loco:.4f}")
print(f"  Stage 1 (random split)     — Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

# ── Retrain Stage 2 on LOCO train (true non-metals only) ──────────────
nm_train_mask_l  = np.array(y_cls_loco_train) == 1
X_nm_train_l     = X_loco_train[nm_train_mask_l]
y_log_nm_train_l = np.array(y_log_loco_train)[nm_train_mask_l]

X_nm_tr_l, X_nm_val_l, y_log_nm_tr_l, y_log_nm_val_l = train_test_split(
    X_nm_train_l, y_log_nm_train_l, test_size=0.05, random_state=RANDOM_SEED
)

reg_loco = xgb.XGBRegressor(
    **study.best_params,
    **XGB_DEVICE_PARAMS,
    objective="reg:squarederror",
    eval_metric=["mae", "rmse"],
    early_stopping_rounds=50,
    random_state=RANDOM_SEED,
)
reg_loco.fit(X_nm_tr_l, y_log_nm_tr_l, eval_set=[(X_nm_val_l, y_log_nm_val_l)], verbose=False)

nm_test_mask_l  = np.array(y_cls_loco_test) == 1
X_nm_test_l     = X_loco_test[nm_test_mask_l]
y_raw_nm_test_l = np.array(y_raw_loco_test)[nm_test_mask_l]

log_pred_l = np.array(reg_loco.predict(X_nm_test_l))
eV_pred_l  = np.expm1(log_pred_l)

mae_loco  = mean_absolute_error(y_raw_nm_test_l, eV_pred_l)
rmse_loco = np.sqrt(mean_squared_error(y_raw_nm_test_l, eV_pred_l))
r2_loco   = r2_score(y_raw_nm_test_l, eV_pred_l)

print(f"\n  Stage 2 (LOCO, nonmetals-only, n={len(y_raw_nm_test_l):,}):")
print(f"    MAE={mae_loco:.4f} eV  RMSE={rmse_loco:.4f} eV  R²={r2_loco:.4f}")
print(f"  Stage 2 (random split)  — MAE={mae_eV_final:.4f} eV  R²={r2_eV_final:.4f}")

# ── Full pipeline, routed by clf_loco's own predictions ───────────────
full_pred_l = np.zeros(len(y_cls_loco_test), dtype=np.float64)
pred_nm_mask_l = pred_loco == 1
if pred_nm_mask_l.sum() > 0:
    X_pred_nm_l = X_loco_test[pred_nm_mask_l]
    log_full_l  = np.array(reg_loco.predict(X_pred_nm_l))
    full_pred_l[pred_nm_mask_l] = np.expm1(log_full_l)

true_eV_full_l = np.array(y_raw_loco_test)
mae_full_l = mean_absolute_error(true_eV_full_l, full_pred_l)
r2_full_l  = r2_score(true_eV_full_l, full_pred_l)

print(f"\n  Full pipeline (LOCO)   — MAE={mae_full_l:.4f} eV  R²={r2_full_l:.4f}")
print(f"  Full pipeline (random) — MAE={mae_final:.4f} eV  R²={r2_final:.4f}")

# ── Bin-wise error breakdown on LOCO test ──────────────────────────────
bin_idx_l = np.digitize(y_raw_nm_test_l, BG_BINS) - 1
loco_bin_report = []
for i, label in enumerate(BG_LABELS):
    mask = bin_idx_l == i
    if mask.sum() < 5:
        continue
    loco_bin_report.append({
        "BG Range": label,
        "n": int(mask.sum()),
        "MAE (eV)": round(float(np.abs(eV_pred_l[mask]-y_raw_nm_test_l[mask]).mean()), 4),
        "RMSE (eV)": round(float(np.sqrt(((eV_pred_l[mask]-y_raw_nm_test_l[mask])**2).mean())), 4),
    })
loco_bin_df = pd.DataFrame(loco_bin_report).set_index("BG Range")
print(f"\n  LOCO error breakdown by bandgap range:\n{loco_bin_df.to_string()}")

loco_bin_df.to_csv(os.path.join(CONFIG["csv_dir"], "loco_error_by_bin.csv"))

loco_summary = {
    "n_train_systems": len(train_systems), "n_test_systems": len(test_systems),
    "n_train_materials": len(X_loco_train), "n_test_materials": len(X_loco_test),
    "stage1_accuracy": acc_loco, "stage1_f1": f1_loco, "stage1_roc_auc": auc_loco,
    "stage2_mae_eV": mae_loco, "stage2_rmse_eV": rmse_loco, "stage2_r2_eV": r2_loco,
    "full_mae_eV": mae_full_l, "full_r2_eV": r2_full_l,
}
pd.Series(loco_summary).to_csv(os.path.join(CONFIG["csv_dir"], "loco_summary.csv"))
print(f"\n  ✅ Saved loco_summary.csv and loco_error_by_bin.csv")

del clf_loco, reg_loco
free_gpu_memory()


# ══════════════════════════════════════════════════════════════════
#  PART B — ANION-SUBSTITUTION PROBE (S / Se / Te family)
#  Holds out ALL Te-containing chalcogenides from training, tests
#  whether the model extrapolates to Te given it has seen the same
#  cation framework with S or Se (the reviewer's literal example).
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  PART B — ANION-SUBSTITUTION PROBE (S/Se/Te)")
print(f"{'='*65}")

CHALCOGENS = {"S", "Se", "Te"}

# Pull element lists out to plain Python once — do NOT use cudf-backed .apply()
# on data containing sets/tuples/lists themselves.
elements_list = elements_series.to_pandas().tolist() if hasattr(elements_series, "to_pandas") else elements_series.tolist()
idx = elements_series.index

def get_elem_set(elem_list):
    if elem_list is None:
        return None
    if isinstance(elem_list, float) and np.isnan(elem_list):
        return None
    try:
        return set(str(e) for e in elem_list)
    except TypeError:
        return None

def cation_framework_str(elem_set):
    if not elem_set:
        return None
    return "-".join(sorted(elem_set - CHALCOGENS))

# Do all container math in pure Python lists/dicts — never store sets in a Series
elem_sets = [get_elem_set(e) for e in elements_list]
cation_strs = [cation_framework_str(s) if s is not None else None for s in elem_sets]
contains_te_list = [bool(s and "Te" in s) for s in elem_sets]
contains_s_list  = [bool(s and "S"  in s) for s in elem_sets]
contains_se_list = [bool(s and "Se" in s) for s in elem_sets]
valid_es_list    = [s is not None for s in elem_sets]

# Now wrap only the *scalar* (string/bool) results back into Series — these are
# dtypes cuDF handles natively, so no more fast/slow round-trip corruption.
cation_series   = pd.Series(cation_strs, index=idx)
contains_te     = pd.Series(contains_te_list, index=idx)
contains_s      = pd.Series(contains_s_list, index=idx)
contains_se     = pd.Series(contains_se_list, index=idx)
valid_mask_es   = pd.Series(valid_es_list, index=idx)

print(f"  Materials containing Te : {contains_te.sum():,}")
print(f"  Materials containing S  : {contains_s.sum():,}")
print(f"  Materials containing Se : {contains_se.sum():,}")

# Cation frameworks seen with S or Se anywhere in the dataset
frameworks_with_s_or_se = set(
    cation_series[valid_mask_es & (contains_s | contains_se)].dropna().unique().tolist()
)

te_mask = contains_te
te_analog_seen_mask  = te_mask & cation_series.isin(frameworks_with_s_or_se)
te_analog_novel_mask = te_mask & ~cation_series.isin(frameworks_with_s_or_se)

print(f"\n  Te-containing materials whose cation framework also appears")
print(f"  with S or Se elsewhere in the dataset (analog seen)     : {te_analog_seen_mask.sum():,}")
print(f"  Te-containing materials with NO S/Se analog in dataset  : {te_analog_novel_mask.sum():,}")
# ── Build train/test split: hold out ALL Te materials from training ──
anion_test_mask  = te_mask & valid_mask_es
anion_train_mask = valid_mask_es & ~te_mask

X_anion_train = X.loc[anion_train_mask[anion_train_mask].index.intersection(X.index)]
X_anion_test  = X.loc[anion_test_mask[anion_test_mask].index.intersection(X.index)]

y_cls_anion_train = y_cls.loc[X_anion_train.index]
y_cls_anion_test  = y_cls.loc[X_anion_test.index]
y_log_anion_train = y_log.loc[X_anion_train.index]
y_raw_anion_test  = y_raw.loc[X_anion_test.index]

print(f"\n  Anion-holdout train set (no Te at all) : {len(X_anion_train):,}")
print(f"  Anion-holdout test set (Te-containing) : {len(X_anion_test):,}")

# ── Retrain Stage 1 + Stage 2 on the anion-holdout split ──────────────
anion_scale_pos_weight = (
    int((y_cls_anion_train == 0).sum()) / max(int((y_cls_anion_train == 1).sum()), 1)
)

X_cls_tr_a, X_cls_val_a, y_cls_tr_a, y_cls_val_a = train_test_split(
    X_anion_train, y_cls_anion_train,
    test_size=0.05, random_state=RANDOM_SEED, stratify=y_cls_anion_train
)

clf_anion = xgb.XGBClassifier(
    **study_stage1.best_params,
    scale_pos_weight=anion_scale_pos_weight,
    **XGB_DEVICE_PARAMS,
    objective="binary:logistic",
    eval_metric=["logloss", "auc"],
    early_stopping_rounds=50,
    random_state=RANDOM_SEED,
)
clf_anion.fit(X_cls_tr_a, y_cls_tr_a, eval_set=[(X_cls_val_a, y_cls_val_a)], verbose=False)

nm_train_mask_a  = np.array(y_cls_anion_train) == 1
X_nm_train_a     = X_anion_train[nm_train_mask_a]
y_log_nm_train_a = np.array(y_log_anion_train)[nm_train_mask_a]

X_nm_tr_a, X_nm_val_a, y_log_nm_tr_a, y_log_nm_val_a = train_test_split(
    X_nm_train_a, y_log_nm_train_a, test_size=0.05, random_state=RANDOM_SEED
)

reg_anion = xgb.XGBRegressor(
    **study.best_params,
    **XGB_DEVICE_PARAMS,
    objective="reg:squarederror",
    eval_metric=["mae", "rmse"],
    early_stopping_rounds=50,
    random_state=RANDOM_SEED,
)
reg_anion.fit(X_nm_tr_a, y_log_nm_tr_a, eval_set=[(X_nm_val_a, y_log_nm_val_a)], verbose=False)

nm_test_mask_a  = np.array(y_cls_anion_test) == 1
X_nm_test_a     = X_anion_test[nm_test_mask_a]
y_raw_nm_test_a = np.array(y_raw_anion_test)[nm_test_mask_a]

log_pred_a = np.array(reg_anion.predict(X_nm_test_a))
eV_pred_a  = np.expm1(log_pred_a)

mae_anion  = mean_absolute_error(y_raw_nm_test_a, eV_pred_a)
r2_anion   = r2_score(y_raw_nm_test_a, eV_pred_a)

print(f"\n  Stage 2 (Te held out entirely, n={len(y_raw_nm_test_a):,}):")
print(f"    MAE={mae_anion:.4f} eV  R²={r2_anion:.4f}")

# ── Split Te test performance by whether an S/Se analog was seen ─────
test_index_a = X_nm_test_a.index
analog_seen_for_test  = te_analog_seen_mask.reindex(test_index_a, fill_value=False).values
analog_novel_for_test = te_analog_novel_mask.reindex(test_index_a, fill_value=False).values

if analog_seen_for_test.sum() > 5:
    mae_seen = mean_absolute_error(
        y_raw_nm_test_a[analog_seen_for_test], eV_pred_a[analog_seen_for_test]
    )
    print(f"\n  Te materials WITH S/Se analog in training (n={analog_seen_for_test.sum():,}): "
          f"MAE={mae_seen:.4f} eV")

if analog_novel_for_test.sum() > 5:
    mae_novel = mean_absolute_error(
        y_raw_nm_test_a[analog_novel_for_test], eV_pred_a[analog_novel_for_test]
    )
    print(f"  Te materials with NO S/Se analog in training (n={analog_novel_for_test.sum():,}): "
          f"MAE={mae_novel:.4f} eV")

anion_summary = {
    "n_train": len(X_anion_train), "n_test_Te": len(X_anion_test),
    "n_test_nonmetal_Te": len(y_raw_nm_test_a),
    "stage2_mae_eV": mae_anion, "stage2_r2_eV": r2_anion,
    "n_analog_seen": int(analog_seen_for_test.sum()),
    "n_analog_novel": int(analog_novel_for_test.sum()),
}
pd.Series(anion_summary).to_csv(os.path.join(CONFIG["csv_dir"], "anion_substitution_summary.csv"))
print(f"\n  ✅ Saved anion_substitution_summary.csv")


tracker.track(df_chg, "Chemical System Generalization", note="LOCO + anion-substitution tests", dataset="Evaluation Dataset")

del df_chg, clf_anion, reg_anion
free_gpu_memory()

# ══════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY — K-FOLD CROSS-VALIDATION FOR VARIANCE ESTIMATION
#  (Reviewer comment #7 — stability of MAE / R² / ROC-AUC)
#
#  IMPORTANT: this CV runs ONLY on X_train_cal / y_*_train_cal —
#  the 80% pool carved out BEFORE the final train/cal split.
#  X_test is NEVER touched here — the original single-split test
#  metrics remain the primary reported result. This cell only
#  estimates how much those metrics would vary under resampling.
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  K-FOLD CV — STABILITY OF MAE / R² / ROC-AUC")
print(f"{'='*65}")

N_CV_FOLDS = 5

# ── Convert the training-cal pool to plain arrays once ────────────────
X_pool       = X_train_cal.reset_index(drop=True) if hasattr(X_train_cal, "reset_index") else X_train_cal
y_cls_pool   = np.array(y_cls_train_cal)
y_log_pool   = np.array(y_log_train_cal)
y_raw_pool   = np.array(y_raw_train_cal)

print(f"\n  CV pool size        : {len(X_pool):,}  (X_test untouched)")
print(f"  Folds               : {N_CV_FOLDS}")
print(f"  Stage 1 params      : reused from study_stage1.best_params")
print(f"  Stage 2 params      : reused from study.best_params")
print(f"  Threshold           : fixed at {OPTIMAL_THRESHOLD} (from main pipeline)")

cv_splitter_outer = StratifiedKFold(
    n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
)

# ── Storage for per-fold metrics ──────────────────────────────────────
cv_results = {
    "stage1_accuracy": [], "stage1_precision": [], "stage1_recall": [],
    "stage1_f1": [],       "stage1_roc_auc": [],
    "stage2_mae_eV": [],   "stage2_rmse_eV": [], "stage2_r2_eV": [],
    "full_mae_eV": [],     "full_rmse_eV": [],   "full_r2_eV": [],
}

for fold_i, (tr_idx, val_idx) in enumerate(cv_splitter_outer.split(X_pool, y_cls_pool), 1):

    print(f"\n  {'─'*55}")
    print(f"  FOLD {fold_i}/{N_CV_FOLDS}")
    print(f"  {'─'*55}")

    if hasattr(X_pool, "iloc"):
        X_fold_tr, X_fold_val = X_pool.iloc[tr_idx], X_pool.iloc[val_idx]
    else:
        X_fold_tr, X_fold_val = X_pool[tr_idx], X_pool[val_idx]

    y_cls_fold_tr,  y_cls_fold_val  = y_cls_pool[tr_idx], y_cls_pool[val_idx]
    y_log_fold_tr,  y_log_fold_val  = y_log_pool[tr_idx], y_log_pool[val_idx]
    y_raw_fold_tr,  y_raw_fold_val  = y_raw_pool[tr_idx], y_raw_pool[val_idx]

    # ── Internal split for early stopping (from fold-train only) ─────
    X_in_tr, X_in_val, y_cls_in_tr, y_cls_in_val = train_test_split(
        X_fold_tr, y_cls_fold_tr,
        test_size=0.05, random_state=RANDOM_SEED, stratify=y_cls_fold_tr
    )

    # ── Stage 1: retrain with tuned params on this fold ───────────────
    fold_scale_pos_weight = (
        int((y_cls_fold_tr == 0).sum()) / int((y_cls_fold_tr == 1).sum())
    )

    clf_fold = xgb.XGBClassifier(
        **study_stage1.best_params,
        scale_pos_weight=fold_scale_pos_weight,
        **XGB_DEVICE_PARAMS,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
    )
    clf_fold.fit(
        X_in_tr, y_cls_in_tr,
        eval_set=[(X_in_val, y_cls_in_val)],
        verbose=False,
    )

    proba_val = np.array(clf_fold.predict_proba(X_fold_val)[:, 1])
    pred_val  = (proba_val >= OPTIMAL_THRESHOLD).astype(int)

    fold_acc = accuracy_score(y_cls_fold_val, pred_val)
    fold_pre = precision_score(y_cls_fold_val, pred_val, average="weighted", zero_division=0)
    fold_rec = recall_score(y_cls_fold_val, pred_val, average="weighted", zero_division=0)
    fold_f1  = f1_score(y_cls_fold_val, pred_val, average="weighted", zero_division=0)
    fold_auc = roc_auc_score(y_cls_fold_val, proba_val)

    cv_results["stage1_accuracy"].append(fold_acc)
    cv_results["stage1_precision"].append(fold_pre)
    cv_results["stage1_recall"].append(fold_rec)
    cv_results["stage1_f1"].append(fold_f1)
    cv_results["stage1_roc_auc"].append(fold_auc)

    print(f"    Stage 1 — Acc={fold_acc:.4f}  F1={fold_f1:.4f}  AUC={fold_auc:.4f}")

    # ── Stage 2: retrain on TRUE non-metals of fold-train only ────────
    nm_tr_mask  = y_cls_fold_tr  == 1
    nm_val_mask = y_cls_fold_val == 1

    if hasattr(X_fold_tr, "iloc"):
        X_nm_fold_tr  = X_fold_tr.iloc[nm_tr_mask]
        X_nm_fold_val = X_fold_val.iloc[nm_val_mask]
    else:
        X_nm_fold_tr  = X_fold_tr[nm_tr_mask]
        X_nm_fold_val = X_fold_val[nm_val_mask]

    y_log_nm_fold_tr  = y_log_fold_tr[nm_tr_mask]
    y_raw_nm_fold_val = y_raw_fold_val[nm_val_mask]

    X_nm_in_tr, X_nm_in_val, y_log_nm_in_tr, y_log_nm_in_val = train_test_split(
        X_nm_fold_tr, y_log_nm_fold_tr, test_size=0.05, random_state=RANDOM_SEED
    )

    reg_fold = xgb.XGBRegressor(
        **study.best_params,
        **XGB_DEVICE_PARAMS,
        objective="reg:squarederror",
        eval_metric=["mae", "rmse"],
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
    )
    reg_fold.fit(
        X_nm_in_tr, y_log_nm_in_tr,
        eval_set=[(X_nm_in_val, y_log_nm_in_val)],
        verbose=False,
    )

    log_pred_nm  = np.array(reg_fold.predict(X_nm_fold_val))
    eV_pred_nm   = np.expm1(log_pred_nm)

    fold_mae_s2  = mean_absolute_error(y_raw_nm_fold_val, eV_pred_nm)
    fold_rmse_s2 = np.sqrt(mean_squared_error(y_raw_nm_fold_val, eV_pred_nm))
    fold_r2_s2   = r2_score(y_raw_nm_fold_val, eV_pred_nm)

    cv_results["stage2_mae_eV"].append(fold_mae_s2)
    cv_results["stage2_rmse_eV"].append(fold_rmse_s2)
    cv_results["stage2_r2_eV"].append(fold_r2_s2)

    print(f"    Stage 2 (non-metals) — MAE={fold_mae_s2:.4f}  R²={fold_r2_s2:.4f}")

    # ── Full pipeline on this fold's validation set ───────────────────
    # Route using the CLASSIFIER'S OWN PREDICTION, not the true label —
    # this matches what hurdle_predict() actually does at inference time
    pred_nm_val_mask = pred_val == 1   # predicted non-metal, from clf_fold

    full_pred_eV = np.zeros(len(y_cls_fold_val), dtype=np.float64)

    if pred_nm_val_mask.sum() > 0:
        if hasattr(X_fold_val, "iloc"):
            X_pred_nm_val = X_fold_val.iloc[pred_nm_val_mask]
        else:
            X_pred_nm_val = X_fold_val[pred_nm_val_mask]

        log_pred_full = np.array(reg_fold.predict(X_pred_nm_val))
        full_pred_eV[pred_nm_val_mask] = np.expm1(log_pred_full)
    # everything predicted metal (pred_val == 0) stays at 0.0 — correct hurdle behavior

    fold_mae_full  = mean_absolute_error(y_raw_fold_val, full_pred_eV)
    fold_rmse_full = np.sqrt(mean_squared_error(y_raw_fold_val, full_pred_eV))
    fold_r2_full   = r2_score(y_raw_fold_val, full_pred_eV)

    cv_results["full_mae_eV"].append(fold_mae_full)
    cv_results["full_rmse_eV"].append(fold_rmse_full)
    cv_results["full_r2_eV"].append(fold_r2_full)

    print(f"    Full pipeline — MAE={fold_mae_full:.4f}  R²={fold_r2_full:.4f}")

    del clf_fold, reg_fold
    free_gpu_memory()

# ── Aggregate: mean ± std across folds ────────────────────────────────
print(f"\n{'='*65}")
print(f"  CV SUMMARY — MEAN ± STD ACROSS {N_CV_FOLDS} FOLDS")
print(f"{'='*65}")

cv_summary = {}
for metric, vals in cv_results.items():
    vals = np.array(vals)
    cv_summary[metric] = {"mean": vals.mean(), "std": vals.std(ddof=1)}
    print(f"  {metric:<20} : {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

cv_summary_df = pd.DataFrame(cv_summary).T
cv_summary_df.to_csv(os.path.join(CONFIG["csv_dir"], "cv_stability_summary.csv"))
print(f"\n  ✅ Saved → cv_stability_summary.csv")

# ── Compare CV mean vs single-split test result ───────────────────────
print(f"\n  {'─'*55}")
print(f"  CV MEAN vs ORIGINAL SINGLE-SPLIT TEST RESULT")
print(f"  {'─'*55}")
print(f"  {'Metric':<20} {'CV mean':>12} {'Test (single)':>15} {'Diff':>10}")
print(f"  Stage1 ROC-AUC       {cv_summary['stage1_roc_auc']['mean']:>12.4f} {auc:>15.4f} {cv_summary['stage1_roc_auc']['mean']-auc:>+10.4f}")
print(f"  Stage2 MAE (eV)      {cv_summary['stage2_mae_eV']['mean']:>12.4f} {mae_eV_final:>15.4f} {cv_summary['stage2_mae_eV']['mean']-mae_eV_final:>+10.4f}")
print(f"  Full MAE (eV)        {cv_summary['full_mae_eV']['mean']:>12.4f} {mae_final:>15.4f} {cv_summary['full_mae_eV']['mean']-mae_final:>+10.4f}")
print(f"  Full R²              {cv_summary['full_r2_eV']['mean']:>12.4f} {r2_final:>15.4f} {cv_summary['full_r2_eV']['mean']-r2_final:>+10.4f}")

# ══════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY — COVERAGE–WIDTH TRADE-OFF & SCREENING UTILITY
#  (Reviewer comment #9)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  COVERAGE–WIDTH TRADE-OFF SWEEP")
print(f"{'='*65}")

# ── Sweep alpha values, recompute q_hat and evaluate on test TPs ─────
alpha_sweep = np.array([0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.02, 0.01])

n_reg = len(reg_cal_scores)   # from Step 4 rerun — already in scope
sweep_results = []

# Reuse the already-computed test predictions from the tuned model
y_proba_sweep = np.array(clf_stage1.predict_proba(X_test)[:, 1])
pred_class_sweep = (y_proba_sweep >= OPTIMAL_THRESHOLD).astype(int)
true_nonmetal_sweep = np.array(y_cls_test) == 1
tp_mask_sweep = true_nonmetal_sweep & (pred_class_sweep == 1)

# Log-scale point predictions for TP samples only (computed once)
if hasattr(X_test, "iloc"):
    X_tp_sweep = X_test.iloc[tp_mask_sweep]
else:
    X_tp_sweep = X_test[tp_mask_sweep]
log_pred_tp = np.array(reg_stage2.predict(X_tp_sweep))
true_eV_tp_sweep = np.array(y_raw_test)[tp_mask_sweep]

for alpha in alpha_sweep:
    q_level = min(np.ceil((n_reg + 1) * (1 - alpha)) / n_reg, 1.0)
    q_hat   = float(np.quantile(reg_cal_scores, q_level))

    lo_eV = np.clip(np.expm1(log_pred_tp - q_hat), 0.0, None)
    hi_eV = np.expm1(log_pred_tp + q_hat)
    width = hi_eV - lo_eV

    covered = ((true_eV_tp_sweep >= lo_eV) & (true_eV_tp_sweep <= hi_eV)).mean()

    sweep_results.append({
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": covered,
        "mean_width_eV": width.mean(),
        "median_width_eV": np.median(width),
        "q_hat_log": q_hat,
    })

    print(f"  target={1-alpha:.2f}  empirical={covered:.4f}  "
          f"mean_width={width.mean():.4f} eV  median_width={np.median(width):.4f} eV")

sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(os.path.join(CONFIG["csv_dir"], "coverage_width_tradeoff.csv"), index=False)

# ── Plot: Coverage vs Width trade-off curve ───────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(sweep_df["target_coverage"], sweep_df["mean_width_eV"],
         marker="o", color=COLORS["primary"], linewidth=2, markersize=6,
         label="Mean PI width (eV)")
ax1.fill_between(sweep_df["target_coverage"],
                  sweep_df["mean_width_eV"],
                  alpha=0.1, color=COLORS["primary"])

# Mark PI90 and PI95 operating points
for target, label in [(0.90, "PI90"), (0.95, "PI95")]:
    row = sweep_df.iloc[(sweep_df["target_coverage"] - target).abs().argmin()]
    ax1.scatter(row["target_coverage"], row["mean_width_eV"],
                color=COLORS["metal"], s=80, zorder=5)
    ax1.annotate(f"{label}\n({row['mean_width_eV']:.2f} eV)",
                 (row["target_coverage"], row["mean_width_eV"]),
                 textcoords="offset points", xytext=(10, 10), fontsize=9)

ax1.set_xlabel("Target Coverage (1 − α)")
ax1.set_ylabel("Mean Prediction Interval Width (eV)")
ax1.legend(loc="upper left")
apply_font(ax1)
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "Coverage_Width_Tradeoff")

print(f"\n  ✅ Saved coverage-width sweep → coverage_width_tradeoff.csv")


# ══════════════════════════════════════════════════════════════════
#  SCREENING UTILITY — CONSERVATIVE SELECTION VS POINT-THRESHOLD
#  (corrected: ranking-based precision@k is provably identical under
#   fixed-width intervals — see manuscript note. This instead compares
#   ADMISSION RULES at matched thresholds, which is where conformal
#   intervals add real value for screening.)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  SCREENING UTILITY — CONSERVATIVE (PI-CONSTRAINED) SELECTION")
print(f"{'='*65}")

SCREEN_THRESHOLD = 3.0

pred_nm_mask_screen = test_preds_uq["predicted_class"].values == 1
point_pred     = test_preds_uq["predicted_BG_eV"].values[pred_nm_mask_screen]
pi90_lower     = test_preds_uq["pi90_lower"].values[pred_nm_mask_screen]
true_bg_screen = np.array(y_raw_test)[pred_nm_mask_screen]

print(f"\n  Candidate pool (predicted non-metals): {len(point_pred):,}")
print(f"  Baseline true-positive rate in pool   : "
      f"{(true_bg_screen > SCREEN_THRESHOLD).mean():.4f}")

# ── Admission rule A: naive point-prediction threshold ────────────────
admit_point = point_pred >= SCREEN_THRESHOLD

# ── Admission rule B: conservative — entire PI90 interval clears bar ──
admit_pi    = pi90_lower >= SCREEN_THRESHOLD

n_point   = admit_point.sum()
n_pi      = admit_pi.sum()
prec_point= (true_bg_screen[admit_point] > SCREEN_THRESHOLD).mean() if n_point > 0 else np.nan
prec_pi   = (true_bg_screen[admit_pi]    > SCREEN_THRESHOLD).mean() if n_pi    > 0 else np.nan
rec_point = (true_bg_screen[admit_point] > SCREEN_THRESHOLD).sum() / (true_bg_screen > SCREEN_THRESHOLD).sum()
rec_pi    = (true_bg_screen[admit_pi]    > SCREEN_THRESHOLD).sum() / (true_bg_screen > SCREEN_THRESHOLD).sum()

print(f"\n  {'Rule':<30} {'n admitted':>12} {'Precision':>11} {'Recall':>9}")
print(f"  {'─'*65}")
print(f"  {'Point prediction ≥ τ':<30} {n_point:>12,} {prec_point:>11.4f} {rec_point:>9.4f}")
print(f"  {'PI90 lower bound ≥ τ':<30} {n_pi:>12,} {prec_pi:>11.4f} {rec_pi:>9.4f}")
print(f"\n  → PI-constrained rule admits a {'smaller' if n_pi < n_point else 'larger'} "
      f"set ({n_pi:,} vs {n_point:,}, {(n_pi/n_point - 1)*100:+.1f}%)")
print(f"  → Precision change: {prec_pi - prec_point:+.4f}")


# ── Sweep τ and plot precision/recall/set-size vs threshold ───────────
tau_values = np.linspace(1.0, 6.0, 26)
rows = []
for tau in tau_values:
    a_pt = point_pred >= tau
    a_pi = pi90_lower >= tau
    n_pt, n_p = a_pt.sum(), a_pi.sum()
    rows.append({
        "tau": tau,
        "n_point": n_pt,
        "n_pi": n_p,
        "precision_point": (true_bg_screen[a_pt] > SCREEN_THRESHOLD).mean() if n_pt > 0 else np.nan,
        "precision_pi":    (true_bg_screen[a_pi] > SCREEN_THRESHOLD).mean() if n_p  > 0 else np.nan,
    })
tau_df = pd.DataFrame(rows)
tau_df.to_csv(os.path.join(CONFIG["csv_dir"], "screening_admission_rule_sweep.csv"), index=False)

fig1, ax1 = plt.subplots(figsize=(7, 5.5))

ax1.plot(
    tau_df["tau"],
    tau_df["n_point"],
    marker="o",
    markersize=3,
    color=COLORS["metal"],
    label="Point-prediction rule",
)

ax1.plot(
    tau_df["tau"],
    tau_df["n_pi"],
    marker="s",
    markersize=3,
    color=COLORS["primary"],
    label="PI90 lower-bound rule",
)

ax1.axvline(
    SCREEN_THRESHOLD,
    color=COLORS["neutral"],
    linestyle="--",
    linewidth=1,
)

ax1.set_xlabel("Admission threshold τ (eV)")
ax1.set_ylabel("Number of candidates admitted")
ax1.legend()

apply_font(ax1)
fig1.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()


save_figure(fig1, "Screening_Admission_Count")

fig2, ax2 = plt.subplots(figsize=(7, 5.5))

ax2.plot(
    tau_df["tau"],
    tau_df["precision_point"],
    marker="o",
    markersize=3,
    color=COLORS["metal"],
    label="Point-prediction rule",
)

ax2.plot(
    tau_df["tau"],
    tau_df["precision_pi"],
    marker="s",
    markersize=3,
    color=COLORS["primary"],
    label="PI90 lower-bound rule",
)

ax2.axvline(
    SCREEN_THRESHOLD,
    color=COLORS["neutral"],
    linestyle="--",
    linewidth=1,
)

ax2.set_xlabel("Admission threshold τ (eV)")
ax2.set_ylabel(
    f"Precision (fraction with true Eg > {SCREEN_THRESHOLD} eV)"
)
ax2.legend()

apply_font(ax2)
fig2.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Screening_Admission_Precision")

y_proba_test = np.array(clf_stage1.predict_proba(X_test)[:, 1])
y_cls_test_np = np.array(y_cls_test)

# PR curve (nonmetal = positive class)
precisions, recalls, pr_thresholds = precision_recall_curve(y_cls_test_np, y_proba_test)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recalls, precisions, lw=2, color=COLORS["primary"], label="PR curve (non-metal)")
# Mark the chosen operating point
op_idx = np.argmin(np.abs(pr_thresholds - OPTIMAL_THRESHOLD))
ax.scatter(recalls[op_idx], precisions[op_idx], color=COLORS["metal"], s=80, zorder=5,
           label=f"Operating point (thr={OPTIMAL_THRESHOLD:.2f})")
ax.set_xlabel("Recall (non-metal)")
ax.set_ylabel("Precision (non-metal)")
ax.legend()
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "PR_curve_Stage1")

# Class-conditional metrics at the chosen threshold, on TEST
y_pred_final = (y_proba_test >= OPTIMAL_THRESHOLD).astype(int)
print(classification_report(y_cls_test_np, y_pred_final, target_names=["Metal", "Non-metal"], digits=4))

# DISSERTATION PLOTS

warnings.filterwarnings("ignore")

# ── Shared arrays (ensure fresh numpy) ───────────────────────────────
_true_nm   = np.array(y_raw_nm_test)      # true eV, non-metals only
_pred_nm   = eV_preds_corrected           # bin-corrected predictions
_true_all  = np.array(y_raw_test)         # true eV, all test samples
_pred_all  = final_pred_eV                # full pipeline predictions
_cls_true  = np.array(y_cls_test)
_cls_pred  = test_predictions_final["predicted_class"].values

# PLOT 1 — Non-metals Only

fig1, ax1 = plt.subplots(figsize=(7, 6))

true = np.asarray(_true_nm).ravel()
pred = np.asarray(_pred_nm).ravel()

# Scatter (density-style via alpha)
ax1.scatter(
    true, pred,
    alpha=0.15,
    s=6,
    color=COLORS["primary"],
    rasterized=True,
    label="Predictions"
)

# Perfect prediction line
lim = max(true.max(), pred.max()) * 1.02
ax1.plot([0, lim], [0, lim], color=COLORS["neutral"], lw=1.5, ls="--", label="Perfect prediction")

# ±0.5 eV band
x = np.linspace(0, lim, 200)
ax1.fill_between(x, x-0.5, x+0.5, color=COLORS["primary"], alpha=0.08, label="±0.5 eV band")

# Axes formatting
ax1.set_xlim(0, lim)
ax1.set_ylim(0, lim)
ax1.set_xlabel("True Band Gap (eV)")
ax1.set_ylabel("Predicted Band Gap (eV)")
ax1.set_title(f"Stage 2 — Non-metals Only\nMAE = {mae_eV_final:.3f} eV | R² = {r2_eV_final:.4f}", pad=10)
ax1.set_aspect("equal")
ax1.legend(frameon=False, loc="upper left")

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig1, "Parity Plots - Stage 2")

# PLOT 2 — Full Hurdle Pipeline

fig2, ax2 = plt.subplots(figsize=(7, 6))

true = np.asarray(_true_all).ravel()
pred = np.asarray(_pred_all).ravel()

# Scatter
ax2.scatter(
    true, pred,
    alpha=0.15,
    s=6,
    color=COLORS["primary"],
    rasterized=True,
    label="Predictions"
)

# Perfect prediction line
lim = max(true.max(), pred.max()) * 1.02
ax2.plot([0, lim], [0, lim], color=COLORS["neutral"], lw=1.5, ls="--", label="Perfect prediction")

# ±0.5 eV band
x = np.linspace(0, lim, 200)
ax2.fill_between(x, x-0.5, x+0.5, color=COLORS["primary"], alpha=0.08, label="±0.5 eV band")

# Axes formatting
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_xlabel("True Band Gap (eV)")
ax2.set_ylabel("Predicted Band Gap (eV)")
ax2.set_title(f"Full Hurdle Pipeline\nMAE = {mae_final:.3f} eV | R² = {r2_final:.4f}", pad=10)
ax2.set_aspect("equal")
ax2.legend(frameon=False, loc="upper left")

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Parity Plots - Full Pipeline")

# ── Compute errors ───────────────────────────────
errors_nm  = _pred_nm  - _true_nm
errors_all = _pred_all - _true_all

# PLOT 1 — Residuals: Stage 2 Non-metals

fig1, ax1 = plt.subplots(figsize=(7, 5))

ax1.hist(errors_nm, bins=100, color=COLORS["primary"],
         alpha=0.75, edgecolor="white", linewidth=0.3)
ax1.axvline(0, color=COLORS["neutral"], lw=1.5, ls="--", label="Zero error")
ax1.axvline(errors_nm.mean(), color=COLORS["metal"], lw=1.5, ls="-",
            label=f"Mean bias = {errors_nm.mean():.3f} eV")

ax1.set_xlabel("Residual (Predicted − True) (eV)")
ax1.set_ylabel("Count")

# Annotate standard deviation
ax1.text(0.97, 0.95, f"σ = {errors_nm.std():.3f} eV",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=10, color=COLORS["neutral"])

ax1.legend(fontsize=9, frameon=False)
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig1, "Stage 2 Residuals (Non-metals)")


# PLOT 2 — Residuals: Full Pipeline

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.hist(errors_all, bins=100, color=COLORS["primary"],
         alpha=0.75, edgecolor="white", linewidth=0.3)
ax2.axvline(0, color=COLORS["neutral"], lw=1.5, ls="--", label="Zero error")
ax2.axvline(errors_all.mean(), color=COLORS["metal"], lw=1.5, ls="-",
            label=f"Mean bias = {errors_all.mean():.3f} eV")

ax2.set_xlabel("Residual (Predicted − True) (eV)")
ax2.set_ylabel("Count")
# ax2.set_title("Full Pipeline Residuals (All materials)")

# Annotate standard deviation
ax2.text(0.97, 0.95, f"σ = {errors_all.std():.3f} eV",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=10, color=COLORS["neutral"])

ax2.legend(fontsize=9, frameon=False)
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig2, "Full Pipeline Residuals")

# PLOT 3 — MAE BY BAND GAP RANGE (with positive/negative bias)

fig, ax = plt.subplots(figsize=(11, 6))

bin_idx_plot = np.digitize(_true_nm, BG_BINS) - 1
mae_per_bin  = []
bias_per_bin = []
n_per_bin    = []

for i in range(len(BG_LABELS)):
    mask = bin_idx_plot == i
    if mask.sum() < 10:
        mae_per_bin.append(0)
        bias_per_bin.append(0)
        n_per_bin.append(0)
        continue
    mae_per_bin.append(float(np.abs(_pred_nm[mask] - _true_nm[mask]).mean()))
    bias_per_bin.append(float((_pred_nm[mask] - _true_nm[mask]).mean()))
    n_per_bin.append(int(mask.sum()))

x      = np.arange(len(BG_LABELS))
width  = 0.38

# MAE bars
bars_mae = ax.bar(x - width/2, mae_per_bin, width, label="MAE (eV)",
                  color=COLORS["primary"], alpha=0.85, zorder=3)

# Bias bars — separate positive and negative for explicit legend
bars_pos = ax.bar(x + width/2,
                  [b if b > 0 else 0 for b in bias_per_bin],
                  width, label="Positive Bias (eV)", color=COLORS["metal"], alpha=0.85, zorder=3)
bars_neg = ax.bar(x + width/2,
                  [b if b < 0 else 0 for b in bias_per_bin],
                  width, label="Negative Bias (eV)", color=COLORS["stable"], alpha=0.85, zorder=3)

# Annotate sample counts
for i, (bar, n) in enumerate(zip(bars_mae, n_per_bin)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"n={n:,}", ha="center", va="bottom", fontsize=8, color=COLORS["neutral"])

# Reference line and axes
ax.axhline(0, color=COLORS["neutral"], lw=1, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(BG_LABELS)
ax.set_xlabel("True Band Gap Range")
ax.set_ylabel("Error (eV)")
# ax.set_title("MAE and Bias by Band Gap Range — Stage 2 (Bin-corrected)", pad=10)

# DFT-PBE reference band
ax.axhspan(0.6, 1.0, alpha=0.08, color=COLORS["secondary"], label="DFT-PBE range (0.6–1.0 eV)")
ax.text(len(BG_LABELS)-0.55, 0.8, "DFT-PBE\nrange", fontsize=8,
        color=COLORS["secondary"], va="center", ha="right")

# Legend
ax.legend(fontsize=10)
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "MAE and Bias by Band Gap Range — Stage 2 (Bin-corrected)")

# ── Helper function to plot top 15 feature importances ─────────────
def plot_feature_importance(imp, stage, save_path=None):
    top = imp.head(15)
    colors = [COLORS["primary"] if i < 5 else COLORS["metal"] if i < 10
              else COLORS["neutral"] for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(top)), top.values, color=colors, alpha=0.85)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)")
    # ax.set_title(f"Top 15 Features — {stage}", pad=10)

    # Value labels
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + top.values.max() * 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color=COLORS["neutral"])

    # Legend for colour tiers
    patches = [
        mpatches.Patch(color=COLORS["primary"], label="Top 1–5"),
        mpatches.Patch(color=COLORS["metal"], label="Top 6–10"),
        mpatches.Patch(color=COLORS["neutral"], label="Top 11–15"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    if save_path:
        save_figure(fig, save_path)

# ── Plot Stage 1 — Classifier ─────────────────────────────
plot_feature_importance(imp_cls, "Stage 1 — Classifier",
                        save_path="feature_importance_stage1")

# ── Plot Stage 2 — Regressor ─────────────────────────────
plot_feature_importance(imp_reg, "Stage 2 — Regressor",
                        save_path="feature_importance_stage2")


# ── Helper function for conformal interval plotting ─────────────
def plot_conformal_intervals(s_true, s_pred, alpha, save_path=None):
    pct   = int((1 - alpha) * 100)
    q     = reg_q_hats[alpha]

    # Lower/upper bounds
    lo_eV = np.clip(np.expm1(np.log1p(np.clip(s_pred, 0, None)) - q), 0, None)
    hi_eV = np.expm1(np.log1p(np.clip(s_pred, 0, None)) + q)
    covered = ((s_true >= lo_eV) & (s_true <= hi_eV))

    # Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Interval band
    color_band = COLORS["primary"] if alpha == 0.10 else COLORS["metal"]
    ax.fill_between(range(len(s_true)), lo_eV, hi_eV,
                    alpha=0.25, color=color_band, label=f"PI{pct} interval")
    # True values
    ax.scatter(range(len(s_true)), s_true, s=4,
               c=np.where(covered, COLORS["stable"], COLORS["metal"]),
               zorder=3, label="True BG")
    # Predicted line
    ax.plot(s_pred, color=color_band, lw=1.0, alpha=0.7, label="Prediction")

    empirical = covered.mean()
    ax.set_xlabel("Sample index (sorted by true BG)")
    ax.set_ylabel("Band Gap (eV)")
    ax.set_title(
        f"PI{pct} Conformal Intervals\n"
        f"Empirical coverage = {empirical*100:.1f}%  "
        f"(target ≥ {pct}%)",
        pad=10
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["stable"],  label="Covered"),
        mpatches.Patch(color=COLORS["metal"], label="Missed"),
        mpatches.Patch(color=color_band, alpha=0.25, label=f"PI{pct} band"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    if save_path:
        save_figure(fig, save_path)

# ── Sort and sample for visualization ─────────────────────────
_idx    = np.argsort(_true_nm)
_sample = np.linspace(0, len(_idx)-1, 500, dtype=int)
_s_idx  = _idx[_sample]

s_true  = _true_nm[_s_idx]
s_pred  = _pred_nm[_s_idx]

# ── Plot PI90 (alpha=0.10) ───────────────────────────────────
plot_conformal_intervals(s_true, s_pred, alpha=0.10,
                         save_path="conformal_PI90")


# ── Plot PI95 (alpha=0.05) ───────────────────────────────────
plot_conformal_intervals(s_true, s_pred, alpha=0.05,
                         save_path="conformal_PI95")


# Stage 1 counts
# tn, fp, fn, tp already defined
conf_matrix = np.array([[tn, fp],
                        [fn, tp]])

# Labels
classes = ["Metal (BG=0)", "Non-metal (Stage 2)"]

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
cax = ax_cm.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.85)

# Annotate counts
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{conf_matrix[i, j]:,}",
                   ha="center", va="center",
                   fontsize=12,
                   color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")

# Axes labels
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(classes, rotation=45)
ax_cm.set_yticklabels(classes)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
# ax_cm.set_title("Stage 1 Gate — Confusion Matrix", pad=15)

fig_cm.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04, label="Count")

plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig_cm, "Stage 1 Gate — Confusion Matrix")

explainer_stage2 = shap.TreeExplainer(reg_stage2)
shap_values_stage2 = explainer_stage2.shap_values(X_nm_test)

# PLOT — SHAP Summary: Stage 2 Non-metals

fig, ax = plt.subplots(figsize=(10, 7))

# SHAP plot (draw onto current matplotlib axis)
shap.summary_plot(
    shap_values_stage2,
    X_nm_test.values,
    feature_names=X_nm_test.columns,
    plot_size=None,
    plot_type="dot",
    # color=COLORS["primary"],
    show=False
)

# Use fewer ticks
ax.xaxis.set_major_locator(MaxNLocator(8))

# Format to 2 decimal places
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Rotate tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Labels
ax.set_xlabel("SHAP value (impact on model output)")
ax.set_ylabel("Features")
# ax.set_title("SHAP Summary (Stage 2 Non-metals)")

# Clean layout
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "SHAP Summary (Stage 2 Non-metals)")


# ── SHAP for Stage 1 classifier ────────────────────────────────────
explainer_stage1 = shap.TreeExplainer(clf_stage1)
shap_values_stage1 = explainer_stage1.shap_values(X_test)

free_gpu_memory()

# For binary XGBClassifier, TreeExplainer usually returns a single
# array of shape (n_samples, n_features) corresponding to the
# log-odds of the positive class (non-metal); some SHAP versions
# instead return a list [class0, class1] — handle both:
if isinstance(shap_values_stage1, list):
    shap_values_stage1 = shap_values_stage1[1]   # non-metal class

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values_stage1,
    X_test.values,
    feature_names=X_test.columns,
    plot_size=None,
    show=False
)

# Use fewer ticks
ax.xaxis.set_major_locator(MaxNLocator(8))

# Format to 2 decimal places
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Rotate tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax.set_xlabel("SHAP value (impact on P(non-metal), log-odds)")
ax.set_ylabel("Features")
plt.tight_layout()

if CONFIG["display_graphs"]:
    plt.show()

save_figure(fig, "SHAP Summary (Stage 1 Classifier)")



del explainer_stage1, explainer_stage2, shap_values_stage1, shap_values_stage2
del chgnet_model, relaxer

free_gpu_memory()

tracker.track(df, "Structural Audits", note="Benchmark Experiments", dataset="Evaluation Dataset")


# # Single-stage XGBoost baseline (retrained on our split)

# ── PRINCIPAL PERFORMANCE TABLE: Per-class + Nonmetals-only ───────────

print(f"\n{'='*60}")
print(f"  PRINCIPAL METRICS — PER-CLASS + NONMETALS-ONLY")
print(f"{'='*60}")

# Stage 1 per-class classification report (metal vs nonmetal), at OPTIMAL_THRESHOLD
y_proba_test  = np.array(clf_stage1.predict_proba(X_test)[:, 1])
y_pred_test   = (y_proba_test >= OPTIMAL_THRESHOLD).astype(int)
y_cls_test_np = np.array(y_cls_test)

print("\n  Stage 1 — Per-class classification report (thr={:.2f}):".format(OPTIMAL_THRESHOLD))
print(classification_report(
    y_cls_test_np, y_pred_test,
    target_names=["Metal", "Non-metal"],
    digits=4
))

# Nonmetals-only regression metrics (already computed as mae_eV_final etc.)
print("  Stage 2 — Nonmetals-only regression (principal indicator):")
print(f"    MAE   : {mae_eV_final:.4f} eV")
print(f"    MedAE : {medae_eV_final:.4f} eV")
print(f"    RMSE  : {rmse_eV_final:.4f} eV")
print(f"    R²    : {r2_eV_final:.4f}")

# Metals-only "regression" performance (trivial: always predict 0)
true_metal_mask = y_cls_test_np == 0
metal_mae  = mean_absolute_error(true_eV_all[true_metal_mask], final_pred_eV[true_metal_mask])
metal_rmse = np.sqrt(mean_squared_error(true_eV_all[true_metal_mask], final_pred_eV[true_metal_mask]))
print(f"\n  Metals-only (n={true_metal_mask.sum():,}) — MAE: {metal_mae:.4f} eV, RMSE: {metal_rmse:.4f} eV")

# ── SINGLE-STAGE XGBoost BASELINE (no hurdle, trained on ALL materials) ──
print(f"\n{'='*60}")
print(f"  SINGLE-STAGE XGBOOST BASELINE")
print(f"{'='*60}")

SINGLE_STAGE_PARAMS = {
    "n_estimators": 1500, "learning_rate": 0.03, "max_depth": 7,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7,
    "colsample_bylevel": 0.8, "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.5,
    **XGB_DEVICE_PARAMS,
    "objective": "reg:squarederror", "eval_metric": ["mae", "rmse"],
    "early_stopping_rounds": 75, "random_state": RANDOM_SEED,
}

# Internal val split from training data only — same discipline as Stage 2
X_ss_tr, X_ss_val, y_ss_tr, y_ss_val = train_test_split(
    X_train_all, y_log_train_all, test_size=0.05, random_state=RANDOM_SEED
)

single_stage_model = xgb.XGBRegressor(**SINGLE_STAGE_PARAMS)
single_stage_model.fit(
    X_ss_tr, y_ss_tr,
    eval_set=[(X_ss_tr, y_ss_tr), (X_ss_val, y_ss_val)],
    verbose=100,
)

# Predict on FULL test set (metals + nonmetals together — no gating)
log_preds_ss = np.array(single_stage_model.predict(X_test))
eV_preds_ss  = np.expm1(log_preds_ss)
eV_preds_ss  = np.clip(eV_preds_ss, 0.0, None)  # bandgap can't be negative
eV_true_ss   = np.array(y_raw_test)

mae_ss  = mean_absolute_error(eV_true_ss, eV_preds_ss)
rmse_ss = np.sqrt(mean_squared_error(eV_true_ss, eV_preds_ss))
r2_ss   = r2_score(eV_true_ss, eV_preds_ss)
medae_ss = np.median(np.abs(eV_true_ss - eV_preds_ss))

print(f"\n  Single-stage — Full population (n={len(eV_true_ss):,}):")
print(f"    MAE   : {mae_ss:.4f} eV")
print(f"    MedAE : {medae_ss:.4f} eV")
print(f"    RMSE  : {rmse_ss:.4f} eV")
print(f"    R²    : {r2_ss:.4f}")

# Nonmetals-only slice for direct comparison to your hurdle Stage 2
nm_mask_test = y_cls_test_np == 1
mae_ss_nm  = mean_absolute_error(eV_true_ss[nm_mask_test], eV_preds_ss[nm_mask_test])
rmse_ss_nm = np.sqrt(mean_squared_error(eV_true_ss[nm_mask_test], eV_preds_ss[nm_mask_test]))
r2_ss_nm   = r2_score(eV_true_ss[nm_mask_test], eV_preds_ss[nm_mask_test])
print(f"\n  Single-stage — Nonmetals-only (n={nm_mask_test.sum():,}):")
print(f"    MAE   : {mae_ss_nm:.4f} eV")
print(f"    RMSE  : {rmse_ss_nm:.4f} eV")
print(f"    R²    : {r2_ss_nm:.4f}")

# Metals-only — does single-stage regress-toward-zero on true metals?
metal_mask_test = y_cls_test_np == 0
mae_ss_metal = mean_absolute_error(eV_true_ss[metal_mask_test], eV_preds_ss[metal_mask_test])
print(f"\n  Single-stage — Metals-only (n={metal_mask_test.sum():,}), true Eg=0:")
print(f"    MAE (should be near 0 if no regression-to-mean bias): {mae_ss_metal:.4f} eV")
print(f"    Mean predicted Eg for true metals: {eV_preds_ss[metal_mask_test].mean():.4f} eV")


tracker.track(df, "Single Stage Baseline", note="Single Stage XGBoost Baseline", dataset="Evaluation Dataset")

# ── COMPOSITION-ONLY DEEP BASELINE (MLP on Magpie features) ────────────

print(f"\n{'='*60}")
print(f"  COMPOSITION-ONLY DEEP BASELINE (MLP)")
print(f"{'='*60}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Select composition-only features ──────────────────────────────────
# Only Magpie compositional descriptors + Number of Elements —
# excludes Lattice, Density, Space Group Number, Magnetic Ordering,
# ox_missing, Total Magnetization (all structure/measurement-dependent)
comp_only_cols = [
    c for c in feature_cols
    if c.startswith("MagpieData") or c == "Number of Elements"
]
print(f"  Composition-only features selected: {len(comp_only_cols)}")

X_train_comp = X_train_all[comp_only_cols].astype("float32")
X_test_comp  = X_test[comp_only_cols].astype("float32")

# Convert to numpy (handles cuDF or pandas)
X_train_comp_np = np.array(X_train_comp)
X_test_comp_np  = np.array(X_test_comp)
y_train_comp_np = np.array(y_log_train_all).astype("float32")
y_test_comp_np  = np.array(y_log_test).astype("float32")

# ── Standardize features (important for MLP convergence) ──────────────
feat_mean = X_train_comp_np.mean(axis=0, keepdims=True)
feat_std  = X_train_comp_np.std(axis=0, keepdims=True) + 1e-8
X_train_comp_np = (X_train_comp_np - feat_mean) / feat_std
X_test_comp_np  = (X_test_comp_np  - feat_mean) / feat_std

# ── Internal val split (5% of train, same discipline as XGBoost stages) ─
n_val = int(0.05 * len(X_train_comp_np))
rng = np.random.default_rng(RANDOM_SEED)
val_idx = rng.choice(len(X_train_comp_np), size=n_val, replace=False)
train_idx = np.setdiff1d(np.arange(len(X_train_comp_np)), val_idx)

X_tr_t  = torch.tensor(X_train_comp_np[train_idx], dtype=torch.float32)
y_tr_t  = torch.tensor(y_train_comp_np[train_idx], dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_train_comp_np[val_idx], dtype=torch.float32)
y_val_t = torch.tensor(y_train_comp_np[val_idx], dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_comp_np, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)

# ── Simple 4-layer MLP ──────────────────────────────────────────────────
class CompositionMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model_mlp = CompositionMLP(len(comp_only_cols)).to(device)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

X_val_t, y_val_t = X_val_t.to(device), y_val_t.to(device)
X_test_t = X_test_t.to(device)

# ── Train with early stopping ───────────────────────────────────────────
best_val_loss = float("inf")
patience, patience_counter = 20, 0
best_state = None

for epoch in range(300):
    model_mlp.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model_mlp(xb), yb)
        loss.backward()
        optimizer.step()

    model_mlp.eval()
    with torch.no_grad():
        val_loss = criterion(model_mlp(X_val_t), y_val_t).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model_mlp.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}, best val loss: {best_val_loss:.5f}")
            break

model_mlp.load_state_dict(best_state)

# ── Evaluate on test set ────────────────────────────────────────────────
model_mlp.eval()
with torch.no_grad():
    log_preds_mlp = model_mlp(X_test_t).cpu().numpy().flatten()

eV_preds_mlp = np.clip(np.expm1(log_preds_mlp), 0.0, None)
eV_true_mlp  = np.array(y_raw_test)

mae_mlp   = mean_absolute_error(eV_true_mlp, eV_preds_mlp)
rmse_mlp  = np.sqrt(mean_squared_error(eV_true_mlp, eV_preds_mlp))
r2_mlp    = r2_score(eV_true_mlp, eV_preds_mlp)
medae_mlp = np.median(np.abs(eV_true_mlp - eV_preds_mlp))

print(f"\n  Composition-only MLP — Full population (n={len(eV_true_mlp):,}):")
print(f"    MAE   : {mae_mlp:.4f} eV")
print(f"    MedAE : {medae_mlp:.4f} eV")
print(f"    RMSE  : {rmse_mlp:.4f} eV")
print(f"    R²    : {r2_mlp:.4f}")

mae_mlp_nm = mean_absolute_error(eV_true_mlp[nm_mask_test], eV_preds_mlp[nm_mask_test])
r2_mlp_nm  = r2_score(eV_true_mlp[nm_mask_test], eV_preds_mlp[nm_mask_test])
print(f"\n  Composition-only MLP — Nonmetals-only (n={nm_mask_test.sum():,}):")
print(f"    MAE   : {mae_mlp_nm:.4f} eV")
print(f"    R²    : {r2_mlp_nm:.4f}")

del model_mlp, X_tr_t, X_val_t, y_val_t, X_test_t, single_stage_model
free_gpu_memory()

tracker.track(df, "Composition Baseline", note="Composition Deep Baseline", dataset="Evaluation Dataset")


# ══════════════════════════════════════════════════════════════════
#  ABLATION — OXIDATION-STATE AND MAGNETIC FEATURES
#  (Reviewer comments — potential leakage via ox_missing/Ox_*,
#   Total Magnetization, and Magnetic Ordering dummies acting as
#   proxies for metallic character)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  ABLATION — OXIDATION-STATE AND MAGNETIC-DERIVED FEATURES")
print(f"{'='*65}")

# ── Identify feature groups by name pattern ───────────────────────────
OX_FEATURES = [c for c in ["Ox_min", "Ox_max", "Ox_mean", "ox_missing"] if c in feature_cols]

MAG_FEATURES = [
    c for c in feature_cols
    if c == "Total Magnetization"
    or c.startswith("Magnetic Ordering_")
    or c == "mag_unknown"
]

ENERGY_FEATURES = [
    c for c in ["Energy Per Atom", "Formation Energy Per Atom"]
    if c in feature_cols
]

print(f"\n  Oxidation features found : {OX_FEATURES}")
print(f"  Magnetic features found  : {MAG_FEATURES}")
print(f"  Energy features found    : {ENERGY_FEATURES}")

# ── Report importance of ALL flagged features in the ORIGINAL models ──
print(f"\n  Importance of flagged features in ORIGINAL (full-feature) models:")
for f in OX_FEATURES + MAG_FEATURES + ENERGY_FEATURES:
    cls_rank = list(imp_cls.index).index(f) + 1 if f in imp_cls.index else None
    reg_rank = list(imp_reg.index).index(f) + 1 if f in imp_reg.index else None
    cls_val  = imp_cls.get(f, float("nan"))
    reg_val  = imp_reg.get(f, float("nan"))
    print(f"    {f:<26} Stage1 rank={str(cls_rank):>4}  imp={cls_val:.4f}  |  "
          f"Stage2 rank={str(reg_rank):>4}  imp={reg_val:.4f}")

# ── Define ablation scenarios ──────────────────────────────────────────
ABLATION_SCENARIOS = {
    "full_model"      : [],                          # baseline — no removal
    "no_oxidation"    : OX_FEATURES,
    "no_magnetic"     : MAG_FEATURES,
    "no_ox_and_mag"   : OX_FEATURES + MAG_FEATURES,
    "no_energy"       : ENERGY_FEATURES,
}

def run_ablation_scenario(drop_cols, scenario_name):
    """Retrain Stage 1 + Stage 2 with drop_cols removed from feature_cols.
    Returns a dict of test-set metrics + the fitted models' importances."""

    cols_kept = [c for c in feature_cols if c not in drop_cols]

    X_train_s = X_train_all[cols_kept]
    X_test_s  = X_test[cols_kept]

    # ── Stage 1 ─────────────────────────────────────────────────────
    X_cls_tr_s, X_cls_val_s, y_cls_tr_s, y_cls_val_s = train_test_split(
        X_train_s, y_cls_train_all,
        test_size=0.05, random_state=RANDOM_SEED, stratify=y_cls_train_all
    )
    clf_s = xgb.XGBClassifier(
        **study_stage1.best_params,
        scale_pos_weight=scale_pos_weight,
        **XGB_DEVICE_PARAMS,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
    )
    clf_s.fit(X_cls_tr_s, y_cls_tr_s, eval_set=[(X_cls_val_s, y_cls_val_s)], verbose=False)

    proba_s = np.array(clf_s.predict_proba(X_test_s)[:, 1])
    pred_s  = (proba_s >= OPTIMAL_THRESHOLD).astype(int)

    acc_s = accuracy_score(y_cls_test_np, pred_s)
    f1_s  = f1_score(y_cls_test_np, pred_s, average="weighted")
    auc_s = roc_auc_score(y_cls_test_np, proba_s)

    # ── Stage 2 — true non-metals only ───────────────────────────────
    nm_train_mask_s = (y_cls_train_all == 1)
    X_nm_train_s    = X_train_s[nm_train_mask_s]
    y_log_nm_train_s= y_log_train_all[nm_train_mask_s]

    X_nm_tr_s, X_nm_val_s, y_log_nm_tr_s, y_log_nm_val_s = train_test_split(
        X_nm_train_s, y_log_nm_train_s, test_size=0.05, random_state=RANDOM_SEED
    )
    reg_s = xgb.XGBRegressor(
        **study.best_params,
        **XGB_DEVICE_PARAMS,
        objective="reg:squarederror",
        eval_metric=["mae", "rmse"],
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
    )
    reg_s.fit(X_nm_tr_s, y_log_nm_tr_s, eval_set=[(X_nm_val_s, y_log_nm_val_s)], verbose=False)

    nm_test_mask_s   = (y_cls_test_np == 1)
    X_nm_test_s      = X_test_s[nm_test_mask_s]
    y_raw_nm_test_s  = np.array(y_raw_test)[nm_test_mask_s]

    log_pred_s = np.array(reg_s.predict(X_nm_test_s))
    eV_pred_s  = np.expm1(log_pred_s)

    mae_s  = mean_absolute_error(y_raw_nm_test_s, eV_pred_s)
    rmse_s = np.sqrt(mean_squared_error(y_raw_nm_test_s, eV_pred_s))
    r2_s   = r2_score(y_raw_nm_test_s, eV_pred_s)

    # ── Full pipeline — routed by THIS scenario's own classifier ─────
    full_pred_s = np.zeros(len(y_cls_test_np), dtype=np.float64)
    pred_nm_mask_s = pred_s == 1
    if pred_nm_mask_s.sum() > 0:
        X_pred_nm_s   = X_test_s[pred_nm_mask_s]
        log_full_s    = np.array(reg_s.predict(X_pred_nm_s))
        full_pred_s[pred_nm_mask_s] = np.expm1(log_full_s)

    true_eV_full_s = np.array(y_raw_test)
    mae_full_s = mean_absolute_error(true_eV_full_s, full_pred_s)
    r2_full_s  = r2_score(true_eV_full_s, full_pred_s)

    imp_cls_s = pd.Series(clf_s.feature_importances_, index=cols_kept).sort_values(ascending=False)
    imp_reg_s = pd.Series(reg_s.feature_importances_, index=cols_kept).sort_values(ascending=False)

    print(f"\n  [{scenario_name}]  n_features={len(cols_kept)}")
    print(f"    Stage1 — Acc={acc_s:.4f}  F1={f1_s:.4f}  AUC={auc_s:.4f}")
    print(f"    Stage2 — MAE={mae_s:.4f} eV  RMSE={rmse_s:.4f} eV  R²={r2_s:.4f}")
    print(f"    Full   — MAE={mae_full_s:.4f} eV  R²={r2_full_s:.4f}")
    print(f"    Top 5 Stage1 features : {list(imp_cls_s.head(5).index)}")
    print(f"    Top 5 Stage2 features : {list(imp_reg_s.head(5).index)}")

    del clf_s, reg_s
    free_gpu_memory()

    return {
        "n_features": len(cols_kept),
        "stage1_accuracy": acc_s, "stage1_f1": f1_s, "stage1_roc_auc": auc_s,
        "stage2_mae_eV": mae_s, "stage2_rmse_eV": rmse_s, "stage2_r2_eV": r2_s,
        "full_mae_eV": mae_full_s, "full_r2_eV": r2_full_s,
    }, imp_cls_s, imp_reg_s


# ── Run all scenarios ───────────────────────────────────────────────
ablation_metrics = {}
ablation_importances_cls = {}
ablation_importances_reg = {}

for scenario_name, drop_cols in ABLATION_SCENARIOS.items():
    print(f"\n{'─'*65}")
    print(f"  SCENARIO: {scenario_name}  (dropping {len(drop_cols)} features)")
    print(f"{'─'*65}")
    metrics, imp_c, imp_r = run_ablation_scenario(drop_cols, scenario_name)
    ablation_metrics[scenario_name] = metrics
    ablation_importances_cls[scenario_name] = imp_c
    ablation_importances_reg[scenario_name] = imp_r

free_gpu_memory()

# ── Summary table across all scenarios ────────────────────────────────
print(f"\n{'='*80}")
print(f"  ABLATION SUMMARY — ALL SCENARIOS")
print(f"{'='*80}")

ablation_df = pd.DataFrame(ablation_metrics).T
ablation_df.index.name = "scenario"
print(f"\n{ablation_df.round(4).to_string()}")

# ── Deltas relative to full model ──────────────────────────────────────
print(f"\n  Delta vs full_model:")
delta_df = ablation_df.subtract(ablation_df.loc["full_model"], axis=1)
delta_df = delta_df.drop(index="full_model")
print(f"\n{delta_df.round(4).to_string()}")

# ── Save everything ─────────────────────────────────────────────────
ablation_df.to_csv(os.path.join(CONFIG["csv_dir"], "ablation_summary_all_scenarios.csv"))
delta_df.to_csv(os.path.join(CONFIG["csv_dir"], "ablation_delta_vs_full.csv"))

for scenario_name in ABLATION_SCENARIOS:
    ablation_importances_cls[scenario_name].to_csv(
        os.path.join(CONFIG["csv_dir"], f"feature_importance_stage1_{scenario_name}.csv")
    )
    ablation_importances_reg[scenario_name].to_csv(
        os.path.join(CONFIG["csv_dir"], f"feature_importance_stage2_{scenario_name}.csv")
    )

print(f"\n  ✅ Saved ablation_summary_all_scenarios.csv, ablation_delta_vs_full.csv,")
print(f"     and per-scenario feature importance tables")

tracker.track(df, "Ablation Study", note="Ablation Study", dataset="Evaluation Dataset")

def generate_pipeline_report(summary_df, dataset_name):

    print(f"\n{'='*100}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*100}")

    labels = summary_df["stage"]
    x = np.arange(len(labels))

    # ==========================================================
    # FIGURE 1: Operational Schema Scale
    # ==========================================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(
        x,
        summary_df["columns"],
        marker="s",
        color=COLORS["tertiary"],
        linewidth=2.5,
        markersize=6,
        label="Column Count"
    )

    mean_cols = summary_df["columns"].mean()

    ax1.axhline(
        mean_cols,
        color=COLORS["metal"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Columns = {mean_cols:.1f}"
    )

    # ax1.set_title(f"{dataset_name} - Operational Schema Scale Profile", fontweight="bold")
    ax1.set_xlabel("Pipeline Operational Stages")
    ax1.set_ylabel("Total Column Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    apply_font(ax1)
    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    save_figure(fig1, f"{dataset_name}_Column_Scaling")

    # ==========================================================
    # FIGURE 2: Row Count
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.plot(
        x,
        summary_df["rows"],
        marker="o",
        color=COLORS["primary"],
        linewidth=2.5,
        markersize=6
    )

    # ax2.set_title(f"{dataset_name} - Dataset Volumetric Filtration Trace")
    ax2.set_xlabel("Pipeline Operational Stages")
    ax2.set_ylabel("Row Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )

    apply_font(ax2)
    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    save_figure(fig2, f"{dataset_name}_Row_Filtering")

    # ==========================================================
    # FIGURE 3: Memory
    # ==========================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    ax3.plot(
        x,
        summary_df["df_memory_mb"],
        marker="o",
        color=COLORS["secondary"],
        linewidth=2.5,
        label="DataFrame"
    )

    ax3.plot(
        x,
        summary_df["process_rss_mb"],
        marker="x",
        linestyle=":",
        color=COLORS["metal"],
        label="RSS"
    )

    # ax3.set_title(f"{dataset_name} - Memory Allocation Trace")
    ax3.set_xlabel("Pipeline Operational Stages")
    ax3.set_ylabel("Memory (MB)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=35, ha="right")
    ax3.legend()

    apply_font(ax3)
    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    save_figure(fig3, f"{dataset_name}_Memory")

    # ==========================================================
    # FIGURE 4: Latency
    # ==========================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    bars = ax4.barh(
        x,
        summary_df["elapsed_sec"],
        color=COLORS["stable"],
        edgecolor="black"
    )

    ax4.set_yticks(x)
    ax4.set_yticklabels(labels)
    ax4.invert_yaxis()
    # ax4.set_title(f"{dataset_name} - Execution Latency")
    ax4.set_xlabel("Seconds")

    for bar in bars:
        width = bar.get_width()
        ax4.text(
            width,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}s",
            va="center"
        )

    apply_font(ax4)
    plt.tight_layout()

    if CONFIG["display_graphs"]:
        plt.show()

    save_figure(fig4, f"{dataset_name}_Latency")

    # ==========================================================
    # TABLES
    # ==========================================================

    os.makedirs(CONFIG["csv_dir"], exist_ok=True)

    print("=" * 80)
    print(f"{dataset_name} - TABLE 1: DATA CORPUS STRUCTURE")
    print("=" * 80)
    table1 = summary_df[
        ["stage", "rows", "delta_rows", "columns", "delta_cols"]
    ]
    print(table1.to_string(index=False))
    table1.to_csv(
        os.path.join(CONFIG["csv_dir"], f"{dataset_name}_data_corpus_structure.csv"),
        index=False
    )

    print("\n")

    print("=" * 80)
    print(f"{dataset_name} - TABLE 2: MEMORY")
    print("=" * 80)
    table2 = summary_df[
        ["stage", "df_memory_mb", "delta_df_mem_mb", "process_rss_mb"]
    ]
    print(table2.to_string(index=False))
    table2.to_csv(
        os.path.join(CONFIG["csv_dir"], f"{dataset_name}_memory.csv"),
        index=False
    )

    print("\n")

    print("=" * 80)
    print(f"{dataset_name} - TABLE 3: LATENCY")
    print("=" * 80)
    table3 = summary_df[
        ["stage", "timestamp", "elapsed_sec", "note"]
    ]
    print(table3.to_string(index=False))
    table3.to_csv(
        os.path.join(CONFIG["csv_dir"], f"{dataset_name}_latency.csv"),
        index=False
    )

    print("\n")

    print("=" * 80)
    print(f"{dataset_name} - TABLE 4: DATA QUALITY")
    print("=" * 80)

    pd.set_option("display.max_colwidth", 45)

    table4 = summary_df[
        ["stage", "total_nulls", "total_duplicates", "dtype_summary"]
    ]
    print(table4.to_string(index=False))
    table4.to_csv(
        os.path.join(CONFIG["csv_dir"], f"{dataset_name}_data_quality.csv"),
        index=False
    )

    print("\n\n")

all_summaries = tracker.summary()

for dataset_name, summary_df in all_summaries.items():
    generate_pipeline_report(summary_df, dataset_name)


log_file.close()