import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json

import statistics
import os
# import pandas as pd

from cycler import cycler
import pylab
# from matplotlib.collections import LineCollection

# import math
from pprint import pprint

pylab.switch_backend("Agg")

mpl.use("pgf")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pgf.texsystem": "pdflatex",
        "font.size": 9,  # footnote/caption size 9pt for paper
        # "font.size": 10,     # caption size 10pt on thesis
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{libertinus}",
                r"\usepackage{newtxmath}",
                # r"\usepackage{lmodern}",
            ]
        ),
        # "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.linestyle": "-",
        "axes.edgecolor": mpl.rcParams["grid.color"],
        # "ytick.color": mpl.rcParams["grid.color"],
        "ytick.direction": "in",
        # "xtick.color": mpl.rcParams["grid.color"],
        # "xtick.direction": "in",
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.titlesize": "medium",
        "axes.titlepad": 4,
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.spines.bottom": False,
        "axes.spines.left": True,
        "axes.axisbelow": True,  # grid below patches
        "axes.prop_cycle": cycler(
            # "color", ["#348ABD", "#7A68A6", "#A60628", "#467821", "#CF4457", "#188487", "#E24A33"]
            "color",
            ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"],
        ),
        "legend.labelspacing": 0.1,
        "legend.handlelength": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0.3,
    }
)

CAPSIZE = 3

results_dir = os.path.dirname(os.path.dirname(__file__)) + "/results"
plots_dir = os.path.dirname(os.path.dirname(__file__)) + "/plots/blocks_vs_clustering"
os.makedirs(plots_dir, exist_ok=True)

# dataset size, dataset name, implementation, pretty dataset name, pretty implementation name
INPUTS = [
    (10, "deep", "blocks", "DEEP10M", "Blocks"),
    (10, "deep", "centroids", "DEEP10M", "Clustering"),
    (1, "deep1M", "blocks", "DEEP1M", "Blocks"),
    (1, "deep1M", "centroids", "DEEP1M", "Clustering"),
    (0.1, "deep100k", "blocks", "DEEP100k", "Blocks"),
    (0.1, "deep100k", "centroids", "DEEP100k", "Clustering"),
]


def plot_comparison(data, impl, dst):
    fig, axs = pylab.subplots(3, 4, figsize=(15, 10))

    for i, (dataset, dataset_data) in enumerate(data.items()):
        impl_data = dataset_data[impl]
        configs = sorted(
            set().union(
                impl_data.keys(),
            ),
            key=lambda x: int(x),  # Convert keys to integers for proper numerical sorting
        )

        for j, config in enumerate(configs):
            config_data = impl_data[config]
            x = np.array([0, 1, 2, 3, 4, 5])
            y = np.array(
                [
                    config_data["load_dataset"]["mean"],
                    config_data["global_index"]["mean"],
                    config_data["distribute_vectors"]["mean"],
                    config_data["distribute_vectors_all"]["mean"],
                    config_data["generate_index"]["mean"],
                    config_data["generate_index_all"]["mean"],
                ]
            )
            yerr = np.array(
                [
                    config_data["load_dataset"]["stdev"],
                    config_data["global_index"]["stdev"],
                    config_data["distribute_vectors"]["stdev"],
                    config_data["distribute_vectors_all"]["stdev"],
                    config_data["generate_index"]["stdev"],
                    config_data["generate_index_all"]["stdev"],
                ]
            )
            parallelism = np.array(
                [
                    1,
                    1,
                    config_data["distribute_vectors_n"]["mean"],
                    config_data["distribute_vectors_n"]["mean"],
                    config_data["generate_index_n"]["mean"],
                    config_data["generate_index_n"]["mean"],
                ]
            )

            axs[i, j].bar(x, y, yerr=yerr, capsize=CAPSIZE)
            for k, txt in enumerate(parallelism):
                axs[i, j].annotate(txt, (x[k], y[k]), textcoords="offset points", xytext=(0, 5), ha="center")
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(["Load Dataset", "Global Index", "Distribute Vectors", "All", "Generate Index", "All"])
            axs[i, j].set_title(f"{dataset} - {config}")
            axs[i, j].sharey(axs[i, -1])
            axs[i, j].grid(True, axis="y")

    pylab.tight_layout()
    pylab.savefig(dst)


def main():
    data = {}
    for size, dataset, impl, pretty_dataset, pretty_impl in INPUTS:
        path = f"{results_dir}/{dataset}/{impl}/indexing"
        if not os.path.exists(path):
            continue

        for filename in os.listdir(path):
            parts = filename.split("_")
            if len(parts) < 4:
                continue  # Skip files that don't match the expected pattern

            config = f"{parts[-3]}"  # Extracts config (e.g., "16" from "results_deep_blocks_16_4_<timestamp>.json")

            with open(os.path.join(path, filename), "r") as file:
                data_dict = json.load(file)

                load_dataset = data_dict.get(f"load_dataset_{impl}", 0)
                global_index = data_dict.get(f"global_index_{impl}", 0)
                distribute_vectors = data_dict.get(f"distribute_vectors_{impl}", [0])
                generate_index = data_dict.get(f"generate_index_{impl}", [0])
                total_indexing = data_dict.get(f"total_indexing_{impl}", 0)

                if pretty_dataset not in data:
                    data[pretty_dataset] = {}

                if pretty_impl not in data[pretty_dataset]:
                    data[pretty_dataset][pretty_impl] = {}

                if config not in data[pretty_dataset][pretty_impl]:
                    data[pretty_dataset][pretty_impl][config] = {}
                    data[pretty_dataset][pretty_impl][config]["load_dataset"] = []
                    data[pretty_dataset][pretty_impl][config]["global_index"] = []
                    data[pretty_dataset][pretty_impl][config]["distribute_vectors"] = []
                    data[pretty_dataset][pretty_impl][config]["distribute_vectors_n"] = []
                    data[pretty_dataset][pretty_impl][config]["distribute_vectors_all"] = []
                    data[pretty_dataset][pretty_impl][config]["generate_index"] = []
                    data[pretty_dataset][pretty_impl][config]["generate_index_n"] = []
                    data[pretty_dataset][pretty_impl][config]["generate_index_all"] = []
                    data[pretty_dataset][pretty_impl][config]["total_indexing"] = []

                data[pretty_dataset][pretty_impl][config]["total_indexing"].append(total_indexing)
                data[pretty_dataset][pretty_impl][config]["load_dataset"].append(load_dataset)
                data[pretty_dataset][pretty_impl][config]["global_index"].append(global_index)
                data[pretty_dataset][pretty_impl][config]["distribute_vectors"].append(max(distribute_vectors))
                data[pretty_dataset][pretty_impl][config]["distribute_vectors_n"].append(len(distribute_vectors))
                data[pretty_dataset][pretty_impl][config]["distribute_vectors_all"].extend(distribute_vectors)
                data[pretty_dataset][pretty_impl][config]["generate_index"].append(max(generate_index))
                data[pretty_dataset][pretty_impl][config]["generate_index_n"].append(len(generate_index))
                data[pretty_dataset][pretty_impl][config]["generate_index_all"].extend(generate_index)

    # compute averages
    data = {
        dataset: {
            impl: {
                config: {
                    "total_indexing": {
                        "mean": statistics.mean(times["total_indexing"]) if times["total_indexing"] else 0,
                        "stdev": statistics.stdev(times["total_indexing"]) if times["total_indexing"] else 0,
                    },
                    "load_dataset": {
                        "mean": statistics.mean(times["load_dataset"]) if times["load_dataset"] else 0,
                        "stdev": statistics.stdev(times["load_dataset"]) if times["load_dataset"] else 0,
                    },
                    "global_index": {
                        "mean": statistics.mean(times["global_index"]) if times["global_index"] else 0,
                        "stdev": statistics.stdev(times["global_index"]) if times["global_index"] else 0,
                    },
                    "distribute_vectors": {
                        "mean": statistics.mean(times["distribute_vectors"]) if times["distribute_vectors"] else 0,
                        "stdev": statistics.stdev(times["distribute_vectors"]) if times["distribute_vectors"] else 0,
                    },
                    "generate_index": {
                        "mean": statistics.mean(times["generate_index"]) if times["generate_index"] else 0,
                        "stdev": statistics.stdev(times["generate_index"]) if times["generate_index"] else 0,
                    },
                    "distribute_vectors_n": {
                        "mean": statistics.mean(times["distribute_vectors_n"]) if times["distribute_vectors_n"] else 0,
                        "stdev": statistics.stdev(times["distribute_vectors_n"]) if times["distribute_vectors_n"] else 0,
                    },
                    "generate_index_n": {
                        "mean": statistics.mean(times["generate_index_n"]) if times["generate_index_n"] else 0,
                        "stdev": statistics.stdev(times["generate_index_n"]) if times["generate_index_n"] else 0,
                    },
                    "distribute_vectors_all": {
                        "mean": statistics.mean(times["distribute_vectors_all"]) if times["distribute_vectors_all"] else 0,
                        "stdev": statistics.stdev(times["distribute_vectors_all"]) if times["distribute_vectors_all"] else 0,
                    },
                    "generate_index_all": {
                        "mean": statistics.mean(times["generate_index_all"]) if times["generate_index_all"] else 0,
                        "stdev": statistics.stdev(times["generate_index_all"]) if times["generate_index_all"] else 0,
                    },
                }
                for config, times in config_data.items()
            }
            for impl, config_data in impl_data.items()
        }
        for dataset, impl_data in data.items()
    }

    pprint(data)

    configs = ["16", "32", "64", "128"]
    print("Clustering")
    for dataset in data:
        load_dataset = [data[dataset]["Clustering"][config]["load_dataset"]["mean"] for config in configs]
        global_index = [data[dataset]["Clustering"][config]["global_index"]["mean"] for config in configs]
        distribute_vectors = [data[dataset]["Clustering"][config]["distribute_vectors"]["mean"] for config in configs]
        generate_index = [data[dataset]["Clustering"][config]["generate_index"]["mean"] for config in configs]
        total_indexing = [data[dataset]["Clustering"][config]["total_indexing"]["mean"] for config in configs]

        print(f"{dataset} - Load Dataset: {load_dataset}")
        print(f"{dataset} - Global Index: {global_index}")
        print(f"{dataset} - Distribute Vectors: {distribute_vectors}")
        print(f"{dataset} - Generate Index: {generate_index}")
        print(f"{dataset} - Total Indexing: {total_indexing}")

    for config in configs:
        load_dataset = [data[dataset]["Clustering"][config]["load_dataset"]["mean"] for dataset in data]
        global_index = [data[dataset]["Clustering"][config]["global_index"]["mean"] for dataset in data]
        distribute_vectors = [data[dataset]["Clustering"][config]["distribute_vectors"]["mean"] for dataset in data]
        generate_index = [data[dataset]["Clustering"][config]["generate_index"]["mean"] for dataset in data]
        total_indexing = [data[dataset]["Clustering"][config]["total_indexing"]["mean"] for dataset in data]

        print(f"{config} - Load Dataset: {load_dataset}")
        print(f"{config} - Global Index: {global_index}")
        print(f"{config} - Distribute Vectors: {distribute_vectors}")
        print(f"{config} - Generate Index: {generate_index}")
        print(f"{config} - Total Indexing: {total_indexing}")

    plot_comparison(data, "Clustering", f"{plots_dir}/indexing_breakdown_clustering.pdf")
    plot_comparison(data, "Blocks", f"{plots_dir}/indexing_breakdown_blocks.pdf")


if __name__ == "__main__":
    main()
