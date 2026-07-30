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
    "raw_pkl"          : "materials_data_2026.pkl",
    "old_chunks"       : "../Final CPU Execution/data_chunks/",
    "num_features"     : 86,      # Number of Magpie features to use (max 86)
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

        # rmm_cupy_allocator moved location across RMM versions —
        # try the modern path first, fall back to the legacy one
        try:
            from rmm.allocators.cupy import rmm_cupy_allocator
        except ImportError:
            from rmm import rmm_cupy_allocator   # older RMM versions

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


# ISSUE: REQUIREMENTS lists mp_api, pymatgen AND matminer, but only mp_api is
# ever passed to install_if_needed below. pymatgen/matminer are imported
# directly further down with no version enforcement — inconsistent with the
# stated "check-before-install" contract for this cell.


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


print_heading("Step 2: Global Imports, Seeding & Version Logging", level=2)


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
from typing import List

# ── Numerical & Scientific ───────────────────────────────────────────
import numpy as np
import pandas as pd

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


print_heading("Step 3: MP API Authentication & Environment Freeze", level=2)

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

print_heading("Step 4: Data Ingestion & Partitioning", level=2)

# ── Download Raw Data ─────────────────────────────────────────────
if CONFIG["download_raw"]:
    print_heading("Raw Data Download", level=3)
    with MPRester(API, monty_decode=False, use_document_model=False) as mpr:
        docs = mpr.materials.summary.search()

        # Save the `docs` object to a file named 'materials_data.pkl'
        with open('materials_data_2026.pkl', 'wb') as f:
            pickle.dump(docs, f)
        print("✅ Data saved to materials_data_2026.pkl")
    del docs, mpr
    free_gpu_memory()

print_heading("Chunking Utility & Execution", level=3)


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

def pkl_to_ids(chunk_path: str) -> set[str]:
    """
    Load one chunk and return the set of material IDs.
    """
    with open(chunk_path, "rb") as f:
        data = pickle.load(f)

    ids = {str(entry["material_id"]) for entry in data}

    del data
    free_gpu_memory()

    return ids

chunk_files = sorted(
    [f for f in os.listdir(CONFIG["old_chunks"]) if f.endswith(".pkl")],
    key=lambda x: int(re.search(r"\d+", x).group())
)

old_ids = set()

for fname in chunk_files:
    path = os.path.join(CONFIG["old_chunks"], fname)

    ids = pkl_to_ids(path)
    old_ids.update(ids)

    print(
        f"✅ {fname}: {len(ids):,} IDs "
        f"(Total: {len(old_ids):,})"
    )

    del ids
    free_gpu_memory()

print(f"\nTotal OLD unique IDs: {len(old_ids):,}")

new_chunk_files = sorted(
    [f for f in os.listdir("data_chunks/") if f.endswith(".pkl")],
    key=lambda x: int(re.search(r"\d+", x).group())
)

new_ids = set()

for fname in new_chunk_files:
    path = os.path.join("data_chunks/", fname)

    ids = pkl_to_ids(path)
    new_ids.update(ids)

    print(
        f"✅ {fname}: {len(ids):,} IDs "
        f"(Total: {len(new_ids):,})"
    )

    del ids
    free_gpu_memory()

print(f"\nTotal NEW unique IDs: {len(new_ids):,}")

new_ids = list(new_ids)

CHUNK_SIZE = 10000

canonical_new_ids = set()
failed_batches = []

with MPRester(API, mute_progress_bars=True) as mpr:
    for i in range(0, len(new_ids), CHUNK_SIZE):
        chunk = new_ids[i:i+CHUNK_SIZE]
        try:
            docs = mpr.materials.summary.search(
                material_ids=chunk,
                fields=["material_id"]
            )
            canonical_new_ids.update(str(doc.material_id) for doc in docs)
            print(f"Chunk {i//CHUNK_SIZE}: requested {len(chunk)}, resolved {len(docs)}")
        except Exception as e:
            print(f"Chunk {i//CHUNK_SIZE} failed: {e}")
            failed_batches.append(chunk)

print(f"\nTotal canonical (numeric) new IDs resolved: {len(canonical_new_ids):,}")
print(f"Original alphabetic ID count: {len(new_ids):,}")
if len(canonical_new_ids) != len(new_ids):
    print(f"⚠️ Mismatch: {len(new_ids) - len(canonical_new_ids)} IDs did not resolve — "
          f"check failed_batches or deprecated/merged entries")

df_old = pd.read_pickle('../Final CPU Execution/Intermediate Pickles/intermediate_data_3.0.pkl')

old_id_set = set(df_old["Material ID"].astype(str))

overlap = old_id_set & canonical_new_ids

print(f"df_old unique IDs        : {len(old_id_set):,}")
print(f"canonical_new_ids        : {len(canonical_new_ids):,}")
print(f"Overlap (leakage)        : {len(overlap):,}")
print(f"% of new set overlapping : {100*len(overlap)/len(canonical_new_ids):.2f}%")

truly_new_materials = canonical_new_ids - old_id_set
print(f"Genuinely new materials in this batch: {len(truly_new_materials):,}")

with open("newMaterials.json", "w") as file:
    json.dump(list(truly_new_materials), file)

print_heading("Field Extraction Schema", level=3)

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


print_heading("Chunk-to-DataFrame Assembly", level=3)

def pkl_to_records(chunk_path: str, keep_ids: set[str]) -> list[dict]:
    """
    Load one chunk and extract records only for selected material IDs.
    """
    with open(chunk_path, "rb") as f:
        data = pickle.load(f)

    records = []

    for entry in data:
        if str(entry["material_id"]) not in keep_ids:
            continue

        records.append({
            field: extractor(entry)
            for field, extractor in FIELDS.items()
        })

    del data
    free_gpu_memory()

    return records

chunk_files = sorted(
    [f for f in os.listdir(CONFIG["chunks_dir"]) if f.endswith('.pkl')],
    key=lambda x: int(re.search(r'\d+', x).group())   # numeric sort
)

all_records = []

for fname in chunk_files:
    path = os.path.join(CONFIG["chunks_dir"], fname)

    records = pkl_to_records(path, truly_new_materials)
    all_records.extend(records)

    print(f"✅ {fname}: {len(records):,} new records loaded")

    del records
    free_gpu_memory()

df = pd.DataFrame(all_records)
del all_records
free_gpu_memory()

print(f"\nFinal DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.0.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 5: Null Audit & Missing-Value Resolution", level=2)


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

df = df.dropna(subset=['Energy Per Atom'])

# ── Null Audit — After ─────────────────────────────────────────────
null_after = df.isna().sum()
remaining_nulls = null_after[null_after > 0]

if remaining_nulls.empty:
    print("✅ No null values remaining")
else:
    print(remaining_nulls.to_string())

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.1.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")


print_heading("Step 6: Dtype Optimization & Range Validation", level=2)


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

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_1.2.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")


print_heading("Step 7: Composition Parsing & Magpie Featurization", level=2)

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

print("\nSanity check — first 3 valid Composition objects:")
samples = df["Composition_obj"].dropna().iloc[:3]
for comp in samples:
    print(
        f"  Type : {type(comp).__name__:<20} "
        f"| Value : {str(comp):<25} "
        f"| Valid : {isinstance(comp, Composition)}"
    )

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

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.0.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

del featurizer, n_partial_nulls, n_failed_rows, feature_nulls, n_features_got, feature_cols
free_gpu_memory()

print_heading("Step 8: Oxidation-State Feature Engineering", level=2)

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

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.1.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

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


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.2.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

# ── Add permanently — with clear documentation ────────────────────────
df["ox_missing"] = df["Oxidation States"].isna().astype("int8")

print(f"Feature 'ox_missing' added to df")
print(f"  Value counts:")
print(f"  0 (ox present) : {(df['ox_missing']==0).sum():>8,}   " f"({(df['ox_missing']==0).mean()*100:.1f}%)")
print(f"  1 (ox present) : {(df['ox_missing']==1).sum():>8,}   " f"({(df['ox_missing']==1).mean()*100:.1f}%)")
print(f"  dtype          : {df['ox_missing'].dtype}")

print_heading("Step 9: Categorical Encoding", level=2)

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

df[["Crystal System_Hexagonal", "Crystal System_Monoclinic", "Crystal System_Orthorhombic", "Crystal System_Tetragonal", "Crystal System_Triclinic"]] = np.nan


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

df[["Magnetic Ordering_AFM", "Magnetic Ordering_FM", "Magnetic Ordering_FiM"]] = np.nan

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.3.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

print_heading("Step 10: Numeric Transforms (log1p & Orthogonality Flags)", level=2)


# ── cuDF Safety Helpers ───────────────────────────────────────────────
def to_np(series, dtype=None) -> np.ndarray:
    """Convert a pandas/cuDF Series to numpy, optionally casting dtype."""
    arr = series.to_numpy() if hasattr(series, "to_numpy") else np.array(series)
    return arr.astype(dtype) if dtype else arr

def to_pd(obj):
    """Convert cuDF DataFrame or Series to pandas if needed."""
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


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


# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.4.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")

# Better: clip both tails symmetrically
p01 = float(df['Total Magnetization'].quantile(0.01))
p99 = float(df['Total Magnetization'].quantile(0.99))
df['Total Magnetization'] = df['Total Magnetization'].clip(lower=p01, upper=p99)

print(f"Total Magnetization clipped at 99th pct: {p99:.3f}")
print(f"New max: {float(df['Total Magnetization'].max()):.3f}")

print_heading("Step 11: Target Construction", level=2)


# Create classification target
# Is Metal (T) = 1 when Band Gap == 0, else 0
df['Is_Metal'] = (df['Band Gap (T)'] == 0).astype(int)

print(df['Is_Metal'].value_counts())
print(f"\nMetal (1)     : {int((df['Is_Metal']==1).sum()):,}")
print(f"Non-Metal (0) : {int((df['Is_Metal']==0).sum()):,}")

df['Log_BandGap'] = np.log1p(df['Band Gap (T)'])

# ── Save independently ───────────────────────────────────────
if CONFIG["save_pickle"]:
    out_path = os.path.join(CONFIG['intermediate_dir'], 'intermediate_data_2.5.pkl')
    df.to_pickle(out_path)
    print(f"Saved → '{out_path}'")
else:
    print("Saving disabled via CONFIG")


print_heading("Phase 2: Model Inference & Evaluation", level=1)
print_heading("Step 1: Load Targets & Final Feature Alignment", level=2)


TARGET_RAW = "Band Gap (T)"
TARGET_LOG = "Log_BandGap"

# ── Verify log target exists ──────────────────────────────────────────
if TARGET_LOG not in df.columns:
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])
    print(f"✅ '{TARGET_LOG}' created from log1p('{TARGET_RAW}')")

assert TARGET_RAW in df.columns
assert TARGET_LOG in df.columns

y_raw = df[TARGET_RAW]
y_log = df[TARGET_LOG]

with open("../label.json") as f:
    labels = json.load(f)

final_cols = list(labels['mp-775218'].keys())

print(f"Number of final columns: {len(final_cols)}")

safe_columns = [col for col in final_cols if col in df.columns]

leftout_cols = [item for item in final_cols if item not in safe_columns]
leftout_cols

df = df.reindex(
    columns=safe_columns, fill_value=0.0
)

df.shape

assert df.index.equals(y_raw.index), "df/y_raw index mismatch!"
assert df.index.equals(y_log.index), "df/y_log index mismatch!"
print(f"Rows aligned: {len(df)} rows, indices match ✓")

implied_log = np.log1p(y_raw)
discrepancy = (implied_log - y_log).abs()
print(f"log1p(y_raw) vs y_log — max diff: {discrepancy.max():.6f}, "
      f"rows >1e-3 off: {(discrepancy > 1e-3).sum()}")

print_heading("Step 2: Load Trained Models", level=2)

STAGE1_THRESHOLD = 0.28
INVERSE_FN = np.expm1


with open("../Models/stage1_classifier_final.pkl", "rb") as f:
    clf = pickle.load(f)

with open("../Models/stage2_regressor_final.pkl", "rb") as f:
    reg = pickle.load(f)

# Sanity check — confirms which column of predict_proba is "non-metal"
print("clf.classes_:", clf.classes_)

print_heading("Step 3: Build Inference Matrix", level=2)


X = df[safe_columns].values.astype(np.float32)
assert X.shape[1] == CONFIG["num_features"], f"Expected {CONFIG['num_features']} features, got {X.shape[1]}"

print_heading("Step 4: Two-Stage Hurdle Prediction", level=2)

probs = clf.predict_proba(X)
prob_non_metal = probs[:, 1]                 # re-verify against clf.classes_ above
is_metal = prob_non_metal < STAGE1_THRESHOLD
pred_class = (~is_metal).astype(int)

pred_log = np.zeros(len(df), dtype=np.float64)
nonmetal_idx = np.where(~is_metal)[0]

if len(nonmetal_idx):
    pred_log[nonmetal_idx] = reg.predict(X[nonmetal_idx])

pred_raw = INVERSE_FN(pred_log)
pred_raw[is_metal] = 0.0 

results = pd.DataFrame({
    "pred_class":     pred_class,
    "prob_non_metal": prob_non_metal,
    "pred_log":       pred_log,
    "pred_raw":       pred_raw,
    "true_raw":       y_raw.values,
    "true_log":       y_log.values,
}, index=df.index)

results["true_class"] = (y_raw.values > 1e-3).astype(int)

print_heading("Step 5: Evaluation Metrics", level=2)



# ── Stage 1: classifier metrics ─────────────────────────────────────────────
acc = (results["true_class"] == results["pred_class"]).mean()
auc = roc_auc_score(results["true_class"], results["prob_non_metal"])
cm  = confusion_matrix(results["true_class"], results["pred_class"])

print("\n── Stage 1: Classifier Gate ──")
print(f"Accuracy: {acc:.4f}   ROC-AUC: {auc:.4f}")
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

# ── Stage 2: regressor metrics, isolated (true non-metals only) ────────────
mask = results["true_class"] == 1

mae_log  = mean_absolute_error(results.loc[mask, "true_log"], results.loc[mask, "pred_log"])
rmse_log = np.sqrt(mean_squared_error(results.loc[mask, "true_log"], results.loc[mask, "pred_log"]))
r2_log   = r2_score(results.loc[mask, "true_log"], results.loc[mask, "pred_log"])

mae_raw  = mean_absolute_error(results.loc[mask, "true_raw"], results.loc[mask, "pred_raw"])
rmse_raw = np.sqrt(mean_squared_error(results.loc[mask, "true_raw"], results.loc[mask, "pred_raw"]))
r2_raw   = r2_score(results.loc[mask, "true_raw"], results.loc[mask, "pred_raw"])

print("\n── Stage 2: Regressor only (true non-metals, n={}) ──".format(mask.sum()))
print(f"[log scale] MAE: {mae_log:.4f}  RMSE: {rmse_log:.4f}  R²: {r2_log:.4f}")
print(f"[eV scale ] MAE: {mae_raw:.4f}  RMSE: {rmse_raw:.4f}  R²: {r2_raw:.4f}")

# ── End-to-end metrics (all rows — real deployment condition) ──────────────
mae_e2e  = mean_absolute_error(results["true_raw"], results["pred_raw"])
rmse_e2e = np.sqrt(mean_squared_error(results["true_raw"], results["pred_raw"]))
r2_e2e   = r2_score(results["true_raw"], results["pred_raw"])

misgated = results["true_class"] != results["pred_class"]

print("\n── End-to-End Pipeline (all rows) ──")
print(f"MAE: {mae_e2e:.4f} eV   RMSE: {rmse_e2e:.4f} eV   R²: {r2_e2e:.4f}")
print(f"Misgated rows: {misgated.sum()} / {len(results)} ({100*misgated.mean():.2f}%)")
if misgated.sum():
    mis_err = (results.loc[misgated, "true_raw"] - results.loc[misgated, "pred_raw"]).abs().mean()
    print(f"Mean |error| on misgated rows: {mis_err:.4f} eV")

print_heading("Bin-wise Error Breakdown", level=3)

# ── Bin-wise breakdown ───────────────────────────────────────────────────────
bins   = [-0.01, 0, 1, 2, 3, 5, np.inf]
bin_labels = ["metal(0)", "0-1eV", "1-2eV", "2-3eV", "3-5eV", ">5eV"]
results["true_bin"] = pd.cut(results["true_raw"], bins=bins, labels=bin_labels)

bin_report = results.groupby("true_bin", observed=True).apply(
    lambda g: pd.Series({
        "n":    len(g),
        "MAE":  mean_absolute_error(g["true_raw"], g["pred_raw"]),
        "RMSE": np.sqrt(mean_squared_error(g["true_raw"], g["pred_raw"])),
    })
)
print("\n── Bin-wise breakdown ──")
print(bin_report)

print_heading("Save Results", level=3)


# ── Save results for inspection / downstream use ────────────────────────────
results.to_csv("inference_results.csv")
print(f"\nSaved: inference_results.csv  ({len(results)} rows)")

# STEP 10 — FINAL SUMMARY (MAE / RMSE / R² only)

# ── Stage 1 — treat prob_non_metal vs true_class(0/1) as regression ────────
mae_cls  = mean_absolute_error(results["true_class"], results["prob_non_metal"])
rmse_cls = np.sqrt(mean_squared_error(results["true_class"], results["prob_non_metal"]))
r2_cls   = r2_score(results["true_class"], results["prob_non_metal"])

# ── Stage 2 — regressor only, true non-metals, eV scale ────────────────────
mae_eV_final  = mean_absolute_error(results.loc[mask, "true_raw"], results.loc[mask, "pred_raw"])
rmse_eV_final = np.sqrt(mean_squared_error(results.loc[mask, "true_raw"], results.loc[mask, "pred_raw"]))
r2_eV_final   = r2_score(results.loc[mask, "true_raw"], results.loc[mask, "pred_raw"])

# ── Full pipeline — all rows, eV scale ──────────────────────────────────────
mae_final  = mean_absolute_error(results["true_raw"], results["pred_raw"])
rmse_final = np.sqrt(mean_squared_error(results["true_raw"], results["pred_raw"]))
r2_final   = r2_score(results["true_raw"], results["pred_raw"])

print(f"\n{'='*60}")
print(f"  HURDLE FRAMEWORK — FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n  STAGE 1 — Metal / Non-metal Classifier")
print(f"  {'─'*40}")
print(f"    MAE  : {mae_cls:.4f}")
print(f"    RMSE : {rmse_cls:.4f}")
print(f"    R²   : {r2_cls:.4f}")

print(f"\n  STAGE 2 — Band Gap Regressor (Non-metals, n={mask.sum()})")
print(f"  {'─'*40}")
print(f"    MAE  (eV) : {mae_eV_final:.4f}")
print(f"    RMSE (eV) : {rmse_eV_final:.4f}")
print(f"    R²        : {r2_eV_final:.4f}")

print(f"\n  FULL PIPELINE (metals + non-metals, n={len(results)})")
print(f"  {'─'*40}")
print(f"    MAE  (eV) : {mae_final:.4f}")
print(f"    RMSE (eV) : {rmse_final:.4f}")
print(f"    R²        : {r2_final:.4f}")

print(f"\n  DFT-PBE benchmark : ~0.6–1.0 eV MAE")
print(f"  Pipeline vs benchmark : ", end="")
if mae_final < 0.6:
    print("✅ Better than DFT-PBE baseline")
elif mae_final < 1.0:
    print("⚠️  Within DFT-PBE range")
else:
    print("❌ Worse than DFT-PBE baseline")

print(f"\n{'='*60}\n")

print_heading("Step 7: Outlier & Residual Analysis", level=2)

results["abs_error"] = (results["true_raw"] - results["pred_raw"]).abs()
results_sorted = results.sort_values("abs_error", ascending=False)

print("── Top 15 worst residuals ──")
print(results_sorted[["true_class", "pred_class", "true_raw", "pred_raw", "abs_error"]].head(15))

print("\n── Error distribution stats ──")
print(results["abs_error"].describe())

# How much of the total error do the worst K predictions account for?
total_error = results["abs_error"].sum()
for k in [3, 5, 10, 20]:
    top_k_sum = results_sorted["abs_error"].head(k).sum()
    print(f"Top {k} worst predictions contribute {100*top_k_sum/total_error:.1f}% of total absolute error")

# Recompute MAE/R² with the worst K removed — if these numbers jump close to
# your original 0.2336/0.8945, that confirms a small-outlier story rather than
# broad degradation.
from sklearn.metrics import r2_score
print("\n── MAE / R² with worst-K outliers excluded ──")
for k in [0, 3, 5, 10]:
    trimmed = results_sorted.iloc[k:]
    mae_trim = trimmed["abs_error"].mean()
    r2_trim  = r2_score(trimmed["true_raw"], trimmed["pred_raw"])
    print(f"Excl. top {k:>2}: MAE={mae_trim:.4f}  R²={r2_trim:.4f}  (n={len(trimmed)})")

# Simple visual gut-check: how skewed is the error distribution?
print("\n── Percentile breakdown of abs_error ──")
for p in [50, 75, 90, 95, 99]:
    print(f"p{p}: {np.percentile(results['abs_error'], p):.4f} eV")


TIME_END = time.perf_counter()
TIME_TAKEN = TIME_END - TIME_START

hours, remainder = divmod(TIME_TAKEN, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Time taken: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
print(f"Time taken in Sec: {TIME_TAKEN}")

log_file.close()