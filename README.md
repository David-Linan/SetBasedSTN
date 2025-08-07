# Hexaly Benchmarking for STN Formulations

This repository provides a structured benchmarking framework for evaluating multiple mathematical formulations of the Short-Term Scheduling (STN) problem using the Hexaly optimization engine. It includes both model generation and performance analysis across varying time discretization levels.

---

## Repository Structure
hexaly_benchmark.py           # Core benchmarking script: builds models, solves them, and logs 
results hexaly_benchmark_plots.py     # Visualization script: generates plots from saved benchmarking results 
hexaly_benchmark_results/     # Folder containing output files (TXT, Excel) from benchmarking runs

---

## What It Does

### `hexaly_benchmark.py`

- Benchmarks five STN formulations:
  - MIP
  - MInP(1)
  - MInLiP(1)
  - MInP(2)
  - MInLiP(2)
- Compares each formulation in two modes:
  - **Unknown-n**: number of task executions is decided by the solver
  - **Known-n**: number of executions is fixed based on MIP output
- Evaluates performance across multiple discretization levels (`acc_level`)
- Extracts metrics:
  - Objective value
  - Bound and gap
  - Computation time
  - Solver status
- Saves results to:
  - `original.txt` and `known_n.txt` (tabular summaries)
  - `original.xlsx` and `known_n.xlsx` (logs)
  - `mip_n.xlsx` (execution counts from MIP)

---






