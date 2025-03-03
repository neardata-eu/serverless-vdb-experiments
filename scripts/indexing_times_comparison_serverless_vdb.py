import json
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Compare total_indexing time between centroids and blocks implementations for different configs")
parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets (deep100M, deep, gist, sift)")
parser.add_argument("--plot_export_dest", required=False, help="Folder to save plots, if provided.")
args = parser.parse_args()

datasets = args.datasets
implementations = ["centroids", "blocks"]
comparison_data = {dataset: {impl: defaultdict(list) for impl in implementations} for dataset in datasets}

for dataset in datasets:
    for impl in implementations:
        directory = f"../results/{dataset}/{impl}/indexing/"
        
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                parts = filename.split("_")
                if len(parts) < 4:
                    continue  # Skip files that don't match the expected pattern
                
                config = f"{parts[-3]}"  # Extracts config (e.g., "16_4" from "results_deep_blocks_16_4_<timestamp>.json")
                
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    key = f"total_indexing_{impl}"
                    if key in data:
                        comparison_data[dataset][impl][config].append(data[key])

# Compute averages
stage_avg = {
    dataset: {
        impl: {config: statistics.mean(times) if times else 0 for config, times in config_data.items()}
        for impl, config_data in impl_data.items()
    }
    for dataset, impl_data in comparison_data.items()
}

# Plot comparison for each dataset
for dataset in datasets:
    configs = sorted(
        set().union(
            *[stage_avg[d]["centroids"].keys() for d in datasets],
            *[stage_avg[d]["blocks"].keys() for d in datasets]
        ),
        key=lambda x: int(x)  # Convert keys to integers for proper numerical sorting
    )

    configs = [str(x) for x in configs]
    x = np.arange(len(configs))
    width = 0.4

    centroids_avg = [stage_avg[dataset]["centroids"].get(config, 0) for config in configs]
    blocks_avg = [stage_avg[dataset]["blocks"].get(config, 0) for config in configs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, centroids_avg, width, label="Centroids")
    ax.bar(x + width/2, blocks_avg, width, label="Blocks")

    ax.set_xlabel("Configuration (Centroids_Parallelism)")
    ax.set_ylabel("Avg Indexing Time (s)")
    ax.set_title(f"Total Indexing Time Comparison: Centroids vs Blocks ({dataset})")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()

    plt.tight_layout()
    if args.plot_export_dest:
        plt.savefig(f"{args.plot_export_dest}/indexing_comparison_{dataset}_centroids_vs_blocks.png")
    else:
        plt.show()
