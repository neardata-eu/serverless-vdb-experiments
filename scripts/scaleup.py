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
from collections import defaultdict

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
plots_dir = os.path.dirname(os.path.dirname(__file__)) + "/plots/scaleup"
os.makedirs(plots_dir, exist_ok=True)

# dataset size, dataset name, pretty dataset name
INPUTS = [
    (10, "deep", "DEEP10M"),
    (100, "deep100M", "DEEP100M"),
]


def plot_scaleup(data, dst):
    fig, ax = pylab.subplots(figsize=(3.33, 1.5))
    x = np.arange(len(data))
    width = 0.6
    bottom = np.zeros(len(data))
    datasets = list(data.keys())
    stages = list(data[datasets[0]].keys())

    for stage in stages:
        values = np.array([data[dataset][stage] for dataset in datasets])
        ax.bar(x, values, width, label=stage, bottom=bottom)
        bottom += values

    # ax.set_xlabel("Dataset")
    ax.set_ylabel("Avg Time (s)")
    # ax.set_title("Stacked Query Time Breakdown: deep100M 320_80 vs deep 32_8")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    # ax.set_ylim(0, 20)
    ax.legend(bbox_to_anchor=(1, 0.5, 0.25, 1), loc="lower right", mode="expand", borderaxespad=0, ncol=1, frameon=False)
    ax.grid(True)

    pylab.tight_layout()
    pylab.savefig(dst)


def main():
    comparison_data = {}
    time_sum = defaultdict(list)
    for size, dataset, pretty_dataset in INPUTS:
        path = f"{results_dir}/{dataset}/blocks/querying"
        if not os.path.exists(path):
            continue
        comparison_data[pretty_dataset] = defaultdict(list)
        for filename in os.listdir(path):
            if dataset == "deep" and not filename.startswith("results_deep_blocks_32_8_"):
                continue  # Skip files that don't use 8 functions
            with open(os.path.join(path, filename), "r") as file:
                data = json.load(file)
                time_sum[pretty_dataset].append(data["total_querying_times_mean"])
                # comparison_data[pretty_dataset]["shuffle"].append(0)
                comparison_data[pretty_dataset]["map_iterdata"].append(data["map_iterdata_times"][0])
                for function in data["map_queries_times"][0]:
                    comparison_data[pretty_dataset]["download_queries"].append(function[1])
                    comparison_data[pretty_dataset]["download_index"].append(sum(function[2]))
                    comparison_data[pretty_dataset]["load_index"].append(sum(function[3]))
                    comparison_data[pretty_dataset]["query_time"].append(sum(function[4]))
                    comparison_data[pretty_dataset]["reduce_time"].append(function[5])
                comparison_data[pretty_dataset]["reduce_iterdata"].append(data["reduce_iterdata_times"][0])
                comparison_data[pretty_dataset]["final_reduce"].append(data["reduce_queries_times"][0][0])

    # Compute averages
    stage_avg = {
        dataset: {stage: statistics.mean(values) for stage, values in stages.items()}
        for dataset, stages in comparison_data.items()
    }
    for dataset in comparison_data:
        total_querying_avg = statistics.mean(time_sum[dataset])
        stage_avg[dataset]["Invoke"] = total_querying_avg - sum(stage_avg[dataset].values())

    plot_data = {
        dataset: {
            "Invoke": stage_avg[dataset]["Invoke"],
            "Load Index": stage_avg[dataset]["download_index"] + stage_avg[dataset]["load_index"],
            "Query": stage_avg[dataset]["download_queries"]
            + stage_avg[dataset]["query_time"]
            + stage_avg[dataset]["reduce_time"],
            "Reduce": stage_avg[dataset]["reduce_iterdata"] + stage_avg[dataset]["final_reduce"],
        }
        for dataset in stage_avg
    }
    pprint(plot_data)
    plot_scaleup(plot_data, f"{plots_dir}/scaleup_querying_100M_vs_10M.pdf")


if __name__ == "__main__":
    main()
