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

> Daniel Barcelona-Pons, Raúl Gracia-Tinedo, Albert Cañadilla-Domingo, Xavier Roca-Canals, and Pedro García-López. 2025. Building Stateless Serverless Vector DBs via Block-based Data Partitioning. Proc. ACM Manag. Data 3, 6 (SIGMOD), Article 304 (December 2025), 25 pages. https://doi.org/10.1145/3769769

```bibtex
@article{barcelonapons2025building,
    author = {Daniel Barcelona-Pons and Raúl Gracia-Tinedo and Albert Cañadilla-Domingo and Xavier Roca-Canals and Pedro García-López},
    title = {Building Stateless Serverless Vector DBs via Block-based Data Partitioning},
    year = {2025},
    issue_date = {December 2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {3},
    number = {6 (SIGMOD)},
    url = {https://doi.org/10.1145/3769769},
    doi = {10.1145/3769769},
    journal = {Proc. ACM Manag. Data},
    month = dec,
    articleno = {204},
    numpages = {25},
    keywords = {Vector Databases, Serverless Functions, Data Partitioning, Indexing}
}
```
