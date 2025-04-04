import os
from collections import defaultdict
import statistics
import json

# Initialize data structures
results = {
    "blocks": {
        "recall": defaultdict(list)
    },
    "clustering": {
        "recall": defaultdict(list)
    }
}

# Process data
directory = f"../results/deep100k/centroids/hnsw/querying/"
if os.path.exists(directory):
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            n_centroids = data["params"]["num_index"]
            num_centroids_search = data["params"]["num_centroids_search"]
            key = (n_centroids, num_centroids_search)
            results["clustering"]["recall"][key].append(data["recalls_mean"])

directory = f"../results/deep100k/blocks/hnsw/querying/"
if os.path.exists(directory):
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            n_centroids = data["params"]["num_index"]
            num_centroids_search = data["params"]["num_centroids_search"]
            key = (n_centroids, num_centroids_search)
            results["blocks"]["recall"][key].append(data["recalls_mean"])

# Get unique N values
all_configs = set(results["clustering"]["recall"].keys()).union(set(results["blocks"]["recall"].keys()))
unique_N = sorted(set(k1 for k1, k2 in all_configs))

# Define percentage relationships
percentages = [25, 50, 75, 100]
n_search_map = {n: {p: int(n * p / 100) for p in percentages} for n in unique_N}

# Calculate Blocks recall for each N (across all N_search values)
blocks_recall_dict = {}
for n in unique_N:
    blocks_values = [statistics.mean(results["blocks"]["recall"][(n, k)]) 
                    for k in set(k2 for (k1, k2) in results["blocks"]["recall"].keys() if k1 == n)
                    if (n, k) in results["blocks"]["recall"] and results["blocks"]["recall"][(n, k)]]
    blocks_recall_dict[n] = f"{statistics.mean(blocks_values):.2f}" if blocks_values else "---"

# Generate LaTeX table
latex_output = r"""\begin{table}[ht]
\centering
\caption{Recall Comparison on deep100k Blocks (Bl.) versus Clustering (Cl.)}
\label{tab:recall_comparison_clustering_blocks}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{c""" + "c" * (len(unique_N) * 2) + r"""}
\toprule
 & \multicolumn{""" + str(len(unique_N) * 2) + r""".}{c}{\textbf{Num. Partitions ($N$)}} \\
 & """ + " & ".join([fr"\multicolumn{{2}}{{c}}{{\textbf{{{n}}}}}" for n in unique_N]) + r""" \\
\cmidrule(lr){2-""" + str(len(unique_N) * 2 + 1) + r"""}
\textbf{$N_{\text{search}}$} & """ + " & ".join([r"\textbf{Bl.} & \textbf{Cl.}" for _ in unique_N]) + r""" \\
\midrule
"""

# Process each percentage
for i, percent in enumerate(percentages):
    row = [fr"\textbf{{{percent}\%}}"]
    for n in unique_N:
        n_search = n_search_map[n][percent]
        
        # Blocks recall (only on middle row, centered vertically)
        blocks_cell = ""
        if i == 1:  # Place in second row (50%) to center vertically among 4 rows
            blocks_cell = fr"\multirow{{2}}{{*}}{{{blocks_recall_dict[n]}}}"
        
        # Clustering recall
        clustering_recall = "---"
        if (n, n_search) in results["clustering"]["recall"] and results["clustering"]["recall"][(n, n_search)]:
            clustering_recall = f"{statistics.mean(results['clustering']['recall'][(n, n_search)]):.2f}"
        
        row.extend([blocks_cell, clustering_recall])
    
    latex_output += " & ".join(row) + r" \\" + "\n"

latex_output += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex_output)