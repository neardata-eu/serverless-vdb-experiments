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
        "font.size": 8,  # footnote/caption size 9pt for paper
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
        "legend.borderpad": 0,
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


def create_indexing_dataset_plot(data, dataset, dst):
    fig, ax = pylab.subplots(figsize=(3.33, 2))

    impl_data = data[dataset]

    configs = sorted(
        set().union(
            *[impl_data[impl].keys() for impl in impl_data],
        ),
        key=lambda x: int(x),  # Convert keys to integers for proper numerical sorting
    )

    configs = [str(x) for x in configs]
    x = np.arange(len(configs))
    width = 0.4

    blocks_avg = [impl_data["Blocks"].get(config, {"mean": 0})["mean"] for config in configs]
    blocks_stdevs = [impl_data["Blocks"].get(config, {"stdev": 0})["stdev"] for config in configs]
    clustering_avg = [impl_data["Clustering"].get(config, {"mean": 0})["mean"] for config in configs]
    clustering_stdevs = [impl_data["Clustering"].get(config, {"stdev": 0})["stdev"] for config in configs]

    # ax.plot(x, blocks_avg, label="Blocks", marker="o")
    # ax.plot(x, clustering_avg, label="Clustering", marker="o")

    ax.bar(x - width / 2, blocks_avg, width, label="Blocks", yerr=blocks_stdevs, capsize=CAPSIZE)
    ax.bar(x + width / 2, clustering_avg, width, label="Clustering", yerr=clustering_stdevs, capsize=CAPSIZE)

    ax.set_xlabel("Num. Partitions ($N$)")
    ax.set_ylabel("Partitioning Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.grid(True)
    ax.legend()

    pylab.tight_layout()
    pylab.savefig(dst)


def create_indexing_config_plot(data, config, datasets, dst):
    fig, ax = pylab.subplots(figsize=(3.33, 2))

    # configs = [str(x) for x in configs]
    x = np.arange(len(datasets))
    width = 0.4

    blocks_avg = [data[dataset]["Blocks"].get(config, {"mean": 0})["mean"] for dataset in datasets]
    blocks_stdevs = [data[dataset]["Blocks"].get(config, {"stdev": 0})["stdev"] for dataset in datasets]
    clustering_avg = [data[dataset]["Clustering"].get(config, {"mean": 0})["mean"] for dataset in datasets]
    clustering_stdevs = [data[dataset]["Clustering"].get(config, {"stdev": 0})["stdev"] for dataset in datasets]
    # pprint(blocks_avg)

    # ax.plot(x, blocks_avg, label="Blocks", marker="o")
    # ax.plot(x, clustering_avg, label="Clustering", marker="o")

    ax.bar(x - width / 2, blocks_avg, width, label="Blocks", yerr=blocks_stdevs, capsize=CAPSIZE)
    ax.bar(x + width / 2, clustering_avg, width, label="Clustering", yerr=clustering_stdevs, capsize=CAPSIZE)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Indexing time (s)")
    # ax.set_title("Total Indexing Time Comparison: Blocks vs Clustering")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(True)
    ax.legend()

    pylab.tight_layout()
    pylab.savefig(dst)


def side_by_side_bar_plot(data, impl, dataset, config, dst):
    # fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.4))
    fig, axs = plt.subplots(1, 2, figsize=(4, 1.4))
    left = axs[0]
    right = axs[1]

    # left
    datasets = list(data.keys())
    datasets.reverse()
    x = np.arange(len(datasets))
    width = 0.4

    impl_avg = [data[dataset][impl].get(config, {"mean": 0})["mean"] for dataset in datasets]
    impl_stdevs = [data[dataset][impl].get(config, {"stdev": 0})["stdev"] for dataset in datasets]

    left.bar(x, impl_avg, width, label=impl, yerr=impl_stdevs, capsize=CAPSIZE)

    ticks = map(lambda x: x.replace("DEEP", ""), datasets)

    left.set_xlabel("Dataset Size (DEEP)")
    left.set_ylabel("Partitioning Time (s)")
    left.set_xticks(x)
    left.set_xticklabels(ticks)
    left.grid(True)

    # right
    config_data = data[dataset][impl]

    configs = sorted(
        config_data.keys(),
        key=lambda x: int(x),  # Convert keys to integers for proper numerical sorting
    )

    configs = [str(x) for x in configs]
    x = np.arange(len(configs))
    width = 0.4

    impl_avg = [config_data.get(config, {"mean": 0})["mean"] for config in configs]
    impl_stdevs = [config_data.get(config, {"stdev": 0})["stdev"] for config in configs]

    right.bar(x, impl_avg, width, label=impl, yerr=impl_stdevs, capsize=CAPSIZE)

    # right.sharey(left)
    right.set_xlabel("Num. Partitions ($N$)")
    # right.set_ylabel("Indexing time (s)")
    # right.set_title("Total Indexing Time Comparison: Blocks vs Clustering")
    right.set_xticks(x)
    right.set_xticklabels(configs)
    right.grid(True)

    pylab.tight_layout()
    pylab.savefig(dst)


def side_by_side_bar_compare(data, dataset_set, config, dst):
    fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.5))
    left = axs[0]
    right = axs[1]

    # left
    datasets = list(data.keys())
    datasets.reverse()
    x = np.arange(len(datasets))
    width = 0.4

    blocks_avg = [data[dataset]["Blocks"].get(config, {"mean": 0})["mean"] for dataset in datasets]
    blocks_stdevs = [data[dataset]["Blocks"].get(config, {"stdev": 0})["stdev"] for dataset in datasets]
    clustering_avg = [data[dataset]["Clustering"].get(config, {"mean": 0})["mean"] for dataset in datasets]
    clustering_stdevs = [data[dataset]["Clustering"].get(config, {"stdev": 0})["stdev"] for dataset in datasets]
    dif = np.array(clustering_avg) / np.array(blocks_avg)
    print(f"{config}: blocks is {dif}x faster than clustering")

    left.bar(
        x - width / 2,
        blocks_avg,
        width,
        label="Blocks",
        yerr=blocks_stdevs,
        capsize=CAPSIZE,
    )
    left.bar(
        x + width / 2,
        clustering_avg,
        width,
        label="Clustering",
        yerr=clustering_stdevs,
        capsize=CAPSIZE,
    )

    ticks = map(lambda x: x.replace("DEEP", ""), datasets)

    left.set_xlabel("Dataset Size (DEEP)")
    left.set_ylabel("Partitioning Time (s)")
    left.set_title("$N=" + str(config) + "$")
    left.set_xticks(x)
    left.set_xticklabels(ticks)
    left.grid(True)

    fig.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="upper center", ncol=2)
    # left.legend()

    # right
    config_data_blocks = data[dataset_set]["Blocks"]
    config_data_clustering = data[dataset_set]["Clustering"]

    configs = sorted(
        config_data_blocks.keys(),
        key=lambda x: int(x),  # Convert keys to integers for proper numerical sorting
    )

    configs = [str(x) for x in configs]
    x = np.arange(len(configs))
    width = 0.4

    blocks_avg = [config_data_blocks.get(config, {"mean": 0})["mean"] for config in configs]
    blocks_stdevs = [config_data_blocks.get(config, {"stdev": 0})["stdev"] for config in configs]
    clustering_avg = [config_data_clustering.get(config, {"mean": 0})["mean"] for config in configs]
    clustering_stdevs = [config_data_clustering.get(config, {"stdev": 0})["stdev"] for config in configs]

    dif = np.array(clustering_avg) / np.array(blocks_avg)
    print(f"{dataset_set}: blocks is {dif}x faster than clustering")

    right.bar(
        x - width / 2,
        blocks_avg,
        width,
        label="Blocks",
        yerr=blocks_stdevs,
        capsize=CAPSIZE,
    )
    right.bar(
        x + width / 2,
        clustering_avg,
        width,
        label="Clustering",
        yerr=clustering_stdevs,
        capsize=CAPSIZE,
    )

    # right.sharey(left)
    right.set_xlabel("Num. Partitions ($N$)")
    # right.set_ylabel("Indexing time (s)")
    right.set_title(dataset_set)
    right.set_xticks(x)
    right.set_xticklabels(configs)
    right.grid(True)

    # fig.legend(
    #             bbox_to_anchor=(1, 0.55, 0.6, 1),
    #             loc="lower right",
    #             # mode="expand",
    #             borderaxespad=0,
    #             ncol=2,
    #             frameon=False,
    #         )

    pylab.tight_layout()
    pylab.subplots_adjust(top=0.75, bottom=0.23)
    pylab.savefig(dst)


def main():
    data = {}
    for size, dataset, impl, pretty_dataset, pretty_impl in INPUTS:
        print(impl)
        if dataset == "deep100k" and impl == "centroids":
            path = f"{results_dir}/{dataset}/{impl}/0/indexing/unbalanced"
        elif impl == "blocks":
            path = f"{results_dir}/{dataset}/{impl}/indexing/"
        else:
            path = f"{results_dir}/{dataset}/{impl}/0/indexing"
        if not os.path.exists(path):
            print(path)
            continue

        for filename in os.listdir(path):
            parts = filename.split("_")
            if len(parts) < 4:
                continue  # Skip files that don't match the expected pattern

            config = f"{parts[-3]}"  # Extracts config (e.g., "16" from "results_deep_blocks_16_4_<timestamp>.json")

            with open(os.path.join(path, filename), "r") as file:
                data_dict = json.load(file)
                key = f"total_indexing_{impl}"
                if key in data_dict:
                    if pretty_dataset not in data:
                        data[pretty_dataset] = {"Blocks": {}, "Clustering": {}}

                    if pretty_impl not in data[pretty_dataset]:
                        data[pretty_dataset][pretty_impl] = {}

                    if config not in data[pretty_dataset][pretty_impl]:
                        data[pretty_dataset][pretty_impl][config] = []

                    data[pretty_dataset][pretty_impl][config].append(data_dict[key])

    # compute averages
    data = {
        dataset: {
            impl: {
                config: {
                    "mean": statistics.mean(times) if times else 0,
                    "stdev": statistics.stdev(times) if times else 0,
                }
                for config, times in config_data.items()
            }
            for impl, config_data in impl_data.items()
        }
        for dataset, impl_data in data.items()
    }

    pprint(data)
    # for dataset in data:
    #     create_indexing_dataset_plot(data, dataset, f"{plots_dir}/indexing_comparison_{dataset}.pdf")

    # impl_data = data[dataset]

    # configs = sorted(
    #     set().union(
    #         *[impl_data[impl].keys() for impl in impl_data],
    #     ),
    #     key=lambda x: int(x),  # Convert keys to integers for proper numerical sorting
    # )

    # configs = [str(x) for x in configs]
    # datasets = list(data.keys())
    # datasets.reverse()
    # for config in configs:
    #     create_indexing_config_plot(data, config, datasets, f"{plots_dir}/indexing_comparison_{config}.pdf")

    side_by_side_bar_plot(
        data, "Clustering", "DEEP100k", "16", f"{plots_dir}/indexing_scaling_side_by_side_clustering.pdf"
    )
    side_by_side_bar_plot(data, "Blocks", "DEEP1M", "16", f"{plots_dir}/indexing_scaling_side_by_side_blocks.pdf")

    side_by_side_bar_compare(data, "DEEP1M", "16", f"{plots_dir}/indexing_scaling_side_by_side_compare.pdf")


if __name__ == "__main__":
    main()
