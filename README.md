# Experiment Results

This repository contains the **experimental results**, **plotting scripts**, and **generated figures** for the paper:  
**Building Stateless Serverless Vector DBs via Block-based Data Partitioning** – *Daniel Barcelona Pons et al., 2026*  

The goal is to provide transparent access to all outputs used in the paper's analysis and visualizations.

---

## Repository Structure

```
├── plots/          # Generated plots used in the paper
├── results/        # JSON files with raw experiment outputs
└── scripts/        # Scripts to process results and generate plots
```

---

## Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra
   ```

2. **Generate plots** from raw results:
   ```bash
   cd scripts/
   python SCRIPT_NAME.py
   ```

Plots will be saved in the `plots/` folder, in a different subdirectory depending on the executed script.

---

## Experiments Files Organization

Each JSON file in `results/` corresponds to a specific experiment described in the paper.
Directory structure:
```
results/
  └── dataset/
      └── implementation/
          └── phase/
```
JSON file naming:
```
results_DATASET_IMPLEMENTATION_<num-partitions>_<num-query-functions>_TIMESTAMP.json
```

---

## Citation

If you make use of this repository, the vector database implementation, or the original research paper, please cite:
```
@article{serverlessvectordb2026,
  title={Building Stateless Serverless Vector DBs via Block-based Data Partitioning},
  author={Daniel Barcelona Pons et al.},
  journal={SIGMOD},
  year={2026}
}
```