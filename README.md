# Resolving Zero-Inflation in Bandgap Prediction for High-Throughput Materials Discovery using a Memory-Efficient, Consumer-Hardware Machine Learning Framework

This repository contains the source code and supporting analysis for the research study:

> **Resolving Zero-Inflation in Bandgap Prediction for High-Throughput Materials Discovery using a Memory-Efficient, Consumer-Hardware Machine Learning Framework**

The project implements a two-stage machine-learning hurdle framework for inorganic-material bandgap prediction using data from the **Materials Project**. The codebase includes the complete prediction pipeline, evaluation against a more recent Materials Project release, and CPU-versus-GPU performance analysis.

---

## Repository Contents

```text
bandgap-hurdle-framework/
│
├── bandgap_prediction_pipeline.py
├── latest_materials_project_evaluation.py
├── cpu_gpu_benchmark_analysis.py
│
├── benchmark_results/
│   ├── cpu/
│   │   ├── latency.csv
│   │   ├── memory.csv
│   │   └── data_corpus_structure.csv
│   │
│   └── gpu/
│       ├── latency.csv
│       ├── memory.csv
│       └── data_corpus_structure.csv
│
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```

### `bandgap_prediction_pipeline.py`

This is the main end-to-end implementation of the proposed framework.

The pipeline includes:

- Memory-optimized data preprocessing
- Materials Project data ingestion
- Chunked processing for large datasets
- Magpie compositional feature generation using `matminer`
- Feature filtering and collinearity reduction
- Two-stage hurdle modelling
- Stage 1 metal/non-metal classification using XGBoost
- Stage 2 non-metal bandgap regression using XGBoost
- Log-transformed regression targets
- Hyperparameter optimization using Optuna
- Model calibration and bias correction
- Conformal prediction intervals for uncertainty quantification
- SHAP-based model interpretation
- Structural sensitivity analysis using CHGNet
- Model and analysis artifact export
- Runtime, memory, and pipeline-stage tracking

The implementation supports both CPU and GPU execution. When GPU execution is enabled, the pipeline can use RAPIDS `cuDF`, CUDA-enabled XGBoost, CuPy, and RMM for accelerated dataframe and model operations.

### `latest_materials_project_evaluation.py`

This script performs an evaluation of the developed framework using a more recent Materials Project release.

It is provided separately from the main experimental pipeline so that the results reported in the study can remain associated with their original dataset/version while a newer Materials Project release can be evaluated independently.

### `cpu_gpu_benchmark_analysis.py`

This script performs the post-processing and visualization of CPU and GPU benchmark results.

The main pipeline is executed separately in CPU and GPU configurations. Each execution records stage-level measurements, including latency and memory usage. The resulting CSV files are then compared by this analysis script to generate CPU-versus-GPU performance figures.

---

## Framework Overview

The prediction architecture follows a two-stage hurdle formulation:

```text
                    Materials Project Data
                              │
                              ▼
                Preprocessing & Cleaning
                              │
                              ▼
                 Magpie Feature Engineering
                              │
                              ▼
                  Feature Selection / Pruning
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Stage 1: XGBoost      │
                  │ Metal / Non-metal     │
                  │ Classification        │
                  └───────────┬───────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  Metal              Non-metal
                    │                   │
                Eg = 0 eV               ▼
                              ┌───────────────────────┐
                              │ Stage 2: XGBoost      │
                              │ Bandgap Regression    │
                              └───────────┬───────────┘
                                          │
                                          ▼
                                Bias Calibration
                                          │
                                          ▼
                              Conformal Prediction
                                          │
                                          ▼
                              Final Bandgap Estimate
```

The hurdle formulation explicitly separates the zero-valued metallic regime from continuous bandgap prediction for non-metallic materials.

---

## Data

The pipeline uses data obtained from the **Materials Project** through its API and local cached data artifacts.

The repository does **not** include the complete raw Materials Project dataset.

A Materials Project API key is required for operations that query the API directly.

Set the API key as an environment variable:

```bash
export MP_API_KEY="YOUR_MATERIALS_PROJECT_API_KEY"
```

On Windows PowerShell:

```powershell
$env:MP_API_KEY="YOUR_MATERIALS_PROJECT_API_KEY"
```

The API key should never be placed directly in the source code or committed to the repository.

---

## Installation

A compatible Python environment is required.

Install the Python dependencies listed in:

```text
requirements-essential.txt
```

For example:

```bash
python -m pip install -r requirements-essential.txt
```

The GPU execution path additionally requires a compatible NVIDIA CUDA environment and RAPIDS/cuDF installation.

The code records versions of important scientific and machine-learning packages during execution to aid reproducibility.

Key dependencies include:

- NumPy
- pandas
- SciPy
- scikit-learn
- XGBoost
- Optuna
- SHAP
- pymatgen
- matminer
- mp-api
- CHGNet
- Matplotlib
- Seaborn
- psutil

GPU execution additionally uses the CUDA/RAPIDS ecosystem, including cuDF, CuPy, and RMM.

---

## Running the Main Pipeline

The primary entry point is:

```bash
python bandgap_prediction_pipeline.py
```

The pipeline contains configuration flags controlling data download, chunk creation, feature generation, GPU execution, caching, and output directories.

The main configuration includes options such as:

```python
CONFIG = {
    "download_raw": False,
    "create_chunks": False,
    "add_features": True,
    "display_graphs": False,
    "gpu_env": False,
}
```

### CPU execution

Set:

```python
"gpu_env": False
```

The pipeline then uses the CPU execution path.

### GPU execution

Set:

```python
"gpu_env": True
```

when a compatible CUDA/RAPIDS environment is available.

The GPU path activates the `cuDF` pandas accelerator and configures CUDA-enabled XGBoost operations.

---

## CPU vs GPU Benchmarking

The computational pipeline should be executed separately in CPU and GPU configurations.

The execution tracker records measurements for individual pipeline stages, including:

- Row count
- Column count
- DataFrame memory
- Process RSS memory
- Stage execution time
- Structural changes across stages

After both executions, the resulting benchmark CSV files are placed under:

```text
benchmark_results/
├── cpu/
└── gpu/
```

The benchmark analysis script can then be executed with:

```bash
python cpu_gpu_benchmark_analysis.py
```

This produces comparative analyses of:

- Stage-wise latency
- DataFrame memory footprint
- CPU versus GPU execution behavior
- Pipeline-stage performance relationships

The benchmark outputs are included in the repository so that the published performance figures can be regenerated without rerunning the complete computational pipeline.

---

## Latest Materials Project Evaluation

The newer-dataset evaluation can be run using:

```bash
python latest_materials_project_evaluation.py
```

This script is intended as a separate evaluation workflow and should not be interpreted as a replacement for the exact dataset/version used to produce the principal results of the study.

Keeping the workflows separate allows the original experimental configuration to remain reproducible while also allowing the framework to be evaluated against newer Materials Project data.

---

## Reproducibility

The experiments use a fixed random seed:

```python
RANDOM_SEED = 42
```

The main pipeline also records versions of important dependencies and saves model and calibration artifacts generated during execution.

The code uses separate training, calibration, and test partitions. The calibration partition is reserved for conformal uncertainty estimation rather than model fitting.

Model artifacts include the trained Stage 1 classifier, Stage 2 regressor, selected feature definitions, bias-correction information, and conformal calibration parameters.

---

## Outputs

Depending on the configuration, the pipeline creates several output directories:

```text
Output Images/
Output CSVs/
Models/
Intermediate Pickles/
data_chunks/
```

These may contain:

- Publication-quality figures
- Diagnostic plots
- Evaluation tables
- Execution metrics
- Trained model files
- Feature metadata
- Calibration parameters
- Intermediate processed datasets

Large intermediate files and raw datasets are intentionally excluded from the public repository unless explicitly required for reproduction.

---

## Computational Architecture

The GPU implementation is designed around a memory-conscious workflow for large materials datasets.

The main optimisation strategies include:

- Numeric downcasting
- Chunked feature engineering
- Explicit memory cleanup
- GPU dataframe acceleration through cuDF
- Shared GPU memory management through RMM
- CUDA-enabled XGBoost
- Runtime monitoring of dataframe and process memory

The pipeline is designed to operate on large Materials Project workloads while controlling system RAM and GPU memory consumption.

---

<!--
## Citation

If you use this software, please cite both the associated research article and the archived software release.

### Software citation

A persistent DOI for the archived version of this repository will be provided through Zenodo:

```text
https://doi.org/10.5281/zenodo.XXXXXXXX
```

Replace the placeholder above with the DOI assigned to the final Zenodo release.

Citation metadata is also provided in:

```text
CITATION.cff
```

### Research article

Add the final published article citation here once the article receives its DOI.
---

-->

## License

This software is distributed under the **MIT License**.

See:

```text
LICENSE
```

for the complete license text.

---

## Important Notes

This repository contains research code intended to reproduce the computational workflows described in the associated study.

Exact numerical results may depend on:

- Materials Project dataset/version
- Python version
- CUDA version
- RAPIDS/cuDF version
- XGBoost version
- Other dependency versions
- Available CPU/GPU hardware

The GPU execution path requires a compatible NVIDIA CUDA environment.

The Materials Project dataset itself is not redistributed with this repository.

---

## Version

The initial archival release corresponding to the research study is:

```text
v1.0.0
```

Future modifications should be released as new versions so that the exact software state associated with the published research remains identifiable and reproducible.
