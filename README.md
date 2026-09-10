# SEHH: A Search Economics-Based Hyper-Heuristic for Single-Objective Real-Parameter Optimization

> Accepted at *2026 IEEE Congress on Evolutionary Computation* (A leading international conference in evolutionary computation).
> 
> 📄 **[Paper](https://linklings.s3.amazonaws.com/organizations/WCCI/wcci2026/submissions/stype102/uehQ6-cec_pap410s2.pdf)**  ·  📏 Benchmark: **[CEC2025](https://github.com/P-N-Suganthan/2025-CEC)**, **[CEC2024](https://github.com/P-N-Suganthan/2024-CEC)**, **[CEC2022](https://github.com/P-N-Suganthan/2022-SO-BO)**, **[CEC2014](https://github.com/P-N-Suganthan/CEC2014)**

## 📖 Overview

**SEHH (Search Economics-Based Hyper-Heuristic)** is an advanced hyper-heuristic method designed to discover and construct effective sequences of Low-Level Heuristics (LLHs) for continuous numerical optimization. By using the exploration-exploitation balance of **Search Economics (SE)**, including its Vision Search, Marketing Survey, Vision Selection, and Transition mechanisms, SEHH navigates the heuristic sequence search space offline to synthesize robust optimization strategies that outperform across diverse fitness landscapes.

### 🌟 Key Highlights
- **Search Economics Strategy**: Employs Search Economics (SE) to search the combinatorial sequence space of metaheuristic operators.
- **Training & Testing Protocol**:
  - **Offline Training**: Mainly based on **CEC2014** single-objective optimization functions to learn  sequences.
  - **Online Testing**: Follows **CEC2024** and **CEC2025** (as well as CEC2022) to validate its performance.
- **Rich Low-Level Heuristics (LLHs)**:
  - **PSO**: Particle Swarm Optimization
  - **GA**: Genetic Algorithm
  - **DE Variants**: `DE/rand/1`, `DE/best/1`, `DE/rand/2`, `DE/best/2`
  - **State-of-the-Art Adaptive DE**: LSHADE, LSRTDE, RDE

---

## 🗂️ Project Layout

```text
SEHH/
├── pyproject.toml
├── uv.lock                       
├── requirements.txt              
├── Makefile                     
├── LICENSE                       
├── README.md    
│
├── benchmarks/                   # Official CEC competition suites & rotation matrices
│   ├── include/                  # Benchmark headers (CEC2014.h, CEC2020.h, CEC2022.h, CEC2024.h)
│   ├── src/                      # CEC benchmark C++ implementations
│   └── data/                     # Shift and shuffle matrix datasets
│
├── src/                          # Core algorithm implementations
│   ├── hyper/                    # SEHH hyper-heuristic engine (SEHH.h, hyperSequence.h, segment.h)
│   ├── metaheuristic/            # Low-level heuristics (PSO.h, GA.h, DE.h, LSHADE.h, LSRTDE.h, RDE.h)
│   ├── config/                   # Global parameter configurations (para_setting.h)
│   ├── utils/                    # Utility classes (instance.h, sample.h, search_state.h, tqdm.h)
│   └── apps/                     # Executable entry points
│       ├── main.cpp              # Online testing & baseline benchmarking entry point
│       ├── train_SEHH.cpp        # Offline SEHH training entry point
│       └── export_config.cpp     # Benchmark parameters and metadata exporter
│
├── scripts/                      # Automated experiment & evaluation pipelines
│   ├── run.py                    # Main test runner & plotting coordinator
│   ├── plot.py                   # Convergence curves, rankings, CSV & Markdown generators
│   ├── make_compare.py           # Statistical metric table generator
│   ├── plot_mutation_rate.py     # Mutation rate convergence visualizer
│   ├── offline_training.csh      # One-click offline training shell script
│   └── online_testing.csh        # One-click online testing shell script
│
├── bin/                          # Compiled binaries (generated during build)
│   ├── main
│   ├── train_SEHH
│   └── export_config
│
└── result/                       # Experimental outputs
    ├── hyper_sequences/          # Learned optimal hyper-heuristic sequence files
    ├── images/                   # Convergence and ranking plots (.png)
    ├── csv/                      # Evaluation metrics in CSV format
    └── md/                       # Evaluation summary tables in Markdown format
```

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements
- **Operating System**: Linux (recommended, e.g., Ubuntu 20.04+).
- **C++ Compiler**: `g++ >= 9.0` or `clang++ >= 11.0` (with C++17 support), `make`.
- **Python**: `Python >= 3.9`.

### 2. Environment Setup

Clone this repository and set up the Python environment using [**`uv`**](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone the repository
git clone https://github.com/jonathan0127/SEHH.git
cd SEHH

# Sync dependencies using uv 
uv sync
```

*(Alternatively, using traditional `pip`: `pip install -r requirements.txt`)*

---

## 🚀 Quick Start


### Step 1: Offline Training (Mainly based on CEC2014)

Run the offline training script to learn the optimal LLH execution sequences on the **CEC2014** benchmark functions:

```bash
# Using the C-Shell script (automatically sets FUNCTION_CHOICE=0, compiles, and trains)
csh scripts/offline_training.csh

# Or manually via Make:
make release
./bin/train_SEHH
```

The learned optimal operator sequence will be recorded and saved to:
`result/output/SEHH_output.txt` .

---

### Step 2: Online Testing & Benchmarking (Following CEC2024 & CEC2025)

Evaluate the learned SEHH sequences against standard baseline algorithms on the **CEC2024** and **CEC2025** (as well as CEC2022) benchmark suites.

> **Heuristic Sequence Files Requirement**:
> - When testing `SEHH-1`, `SEHH-2`, or `SEHH-3`, the program loads the corresponding learned operator sequences from text files located in **`result/hyper_sequences/`**:
> - File naming format: **`result/hyper_sequences/SEHH_sequences<ID>.txt`** (e.g., `SEHH_sequences1.txt` corresponds to `SEHH-1`, `SEHH_sequences2.txt` to `SEHH-2`, etc.).
> - Pre-trained optimal sequences for the benchmarks are provided in `result/hyper_sequences/`.
> - If you run custom offline training, ensure the generated sequence is copied or saved to `result/hyper_sequences/SEHH_sequences<ID>.txt` before starting online testing.

```bash
# Using the C-Shell script (automatically sets FUNCTION_CHOICE=1, compiles, and tests)
csh scripts/online_testing.csh

# Or directly via uv:
uv run python scripts/run.py
```

This will:
1. Compile `bin/main` and `bin/export_config`.
2. Load the heuristic sequences from `result/hyper_sequences/SEHH_sequences*.txt`.
3. Execute $N$ independent runs (default: 25 runs) per benchmark function across the configured benchmark year.
4. Automatically plot convergence curves and compute ranking statistics.
