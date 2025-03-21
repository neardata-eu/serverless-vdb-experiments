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
import re
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
plots_dir = os.path.dirname(os.path.dirname(__file__)) + "/plots/milvus"
os.makedirs(plots_dir, exist_ok=True)

# dataset name, implementation, pretty dataset name, pretty implementation name
INPUTS = [
    ("deep", "blocks", "DEEP10M", "Blocks"),
    ("deep", "milvus", "DEEP10M", "Milvus"),
    ("sift", "blocks", "SIFT10M", "Blocks"),
    ("sift", "milvus", "SIFT10M", "Milvus"),
    ("gist", "blocks", "GIST1M", "Blocks"),
    ("gist", "milvus", "GIST1M", "Milvus"),
]


def parse_milvus_results(dataset):
    milvus_indexing = defaultdict(list)
    milvus_querying = defaultdict(list)

    for query_functions in [4, 8]:
        directory = f"{results_dir}/{dataset}/milvus{query_functions}qf/"
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.split(".")[1] != "txt":
                    continue
                config = filename.split(".")[0]
                with open(os.path.join(directory, filename), "r") as file:
                    data = file.read()
                    load_time_pattern = re.findall(r"load_duration\(insert \+ optimize\) = ([\d.]+)", data)
                    query_time_pattern = re.findall(r"cost=([\d.]+)s", data)

                    milvus_indexing[f"{config}"].extend([float(q) for q in load_time_pattern])
                    milvus_querying[f"{config}_{query_functions}"].extend([float(q) for q in query_time_pattern])

    return milvus_indexing, milvus_querying


def parse_blocks_results(dataset, version):
    blocks_indexing = defaultdict(list)
    blocks_querying = defaultdict(lambda: defaultdict(list))

    directory_index = f"{results_dir}/{dataset}/blocks/indexing/"
    directory_query = f"{results_dir}/{dataset}/blocks/querying/"

    if os.path.exists(directory_index):
        for filename in os.listdir(directory_index):
            config = filename.split("_")[-3]
            with open(os.path.join(directory_index, filename), "r") as file:
                data = json.load(file)
                blocks_indexing[config].append(data["total_indexing_blocks"])

    if os.path.exists(directory_query):
        for filename in os.listdir(directory_query):
            config = filename.split("_")[-3] + "_" + filename.split("_")[-2]  # Npartitions_Nfunctions
            with open(os.path.join(directory_query, filename), "r") as file:
                data = json.load(file)
                total = data["total_querying_times_mean"]
                stages = defaultdict(list)
                stages["map_iterdata"].append(data["map_iterdata_times"][0])
                for function in data["map_queries_times"][0]:
                    stages["download_queries"].append(function[1])
                    stages["download_index"].append(sum(function[2]))
                    stages["load_index"].append(sum(function[3]))
                    stages["query_time"].append(sum(function[4]))
                    stages["reduce_time"].append(function[5])
                stages["reduce_iterdata"].append(data["reduce_iterdata_times"][0])
                stages["final_reduce"].append(data["reduce_queries_times"][0][0])

                invoke = total - sum(statistics.mean(stage) for stage in stages.values())
                load_index = sum(statistics.mean(stage) for stage in [stages["download_index"], stages["load_index"]])

                if version == 1:
                    total = data["total_querying_times_mean"]
                    query = total - invoke - load_index
                    startup = total - query
                    blocks_querying[config]["total"].append(total)
                    blocks_querying[config]["query"].append(query)
                    blocks_querying[config]["startup"].append(startup)
                    blocks_querying[config]["invoke"].append(invoke)
                    blocks_querying[config]["load"].append(load_index)

                elif version == 2:
                    blocks_querying[config].append(
                        data["map_iterdata_times"][0]
                        + statistics.mean((function[0] for function in data["map_queries_times"][0]))
                        + data["reduce_iterdata_times"][0]
                        + data["reduce_queries_times"][0][0]
                    )
                elif version == 3:
                    blocks_querying[config].append(
                        data["map_iterdata_times"][0]
                        + statistics.mean(
                            (function[1] + sum(function[4]) + function[5] for function in data["map_queries_times"][0])
                        )
                        + data["reduce_iterdata_times"][0]
                        + data["reduce_queries_times"][0][0]
                    )
                elif version == 4:
                    total = data["total_querying_times_mean"]

                    blocks_querying[config].append(
                        total
                        - statistics.mean(
                            (sum(function[2]) + sum(function[3]) for function in data["map_queries_times"][0])
                        )
                    )

    return blocks_indexing, blocks_querying


def plot_comparison(data, dst):
    fig, axs = pylab.subplots(2, 3, figsize=(6.6, 3.6))
    for i, dataset in enumerate(data):
        configs = sorted(
            set().union(data[dataset]["Blocks"]["indexing"].keys(), data[dataset]["Milvus"]["indexing"].keys()),
            key=lambda x: int(x),
        )

        milvus_means = np.array([statistics.mean(data[dataset]["Milvus"]["indexing"][c]) for c in configs]) / 60
        milvus_sd = np.array([statistics.stdev(data[dataset]["Milvus"]["indexing"][c]) for c in configs]) / 60
        blocks_means = np.array([statistics.mean(data[dataset]["Blocks"]["indexing"][c]) for c in configs]) / 60
        blocks_sd = np.array([statistics.stdev(data[dataset]["Blocks"]["indexing"][c]) for c in configs]) / 60

        x = np.arange(len(configs))
        width = 0.4
        axs[0][i].bar(x - width / 2, milvus_means, width, label="Milvus")
        axs[0][i].errorbar(
            x - width / 2,
            milvus_means,
            yerr=milvus_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )
        axs[0][i].bar(x + width / 2, blocks_means, width, label="Blocks")
        axs[0][i].errorbar(
            x + width / 2,
            blocks_means,
            yerr=blocks_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )

        axs[0][i].set_title(f"{dataset}")
        axs[0][i].grid(True)
        # if i == 1:
        # axs[0][i].set_xlabel("Number of Partitions")

        axs[0][i].set_xticks(x)
        axs[0][i].set_xticklabels(configs)
        if i == 0:
            axs[0][i].set_ylabel("Indexing Time (min)")
        if i == 2:
            axs[0][i].legend(
                bbox_to_anchor=(1, 0.75, 0.25, 1),
                loc="lower right",
                mode="expand",
                borderaxespad=0,
                ncol=1,
                frameon=False,
            )

    for i, dataset in enumerate(data):
        configs = sorted(
            set().union(data[dataset]["Blocks"]["querying"].keys(), data[dataset]["Milvus"]["querying"].keys()),
            key=lambda x: tuple(map(int, x.split("_"))),
        )

        milvus_means = np.array([statistics.mean(data[dataset]["Milvus"]["querying"][c]) for c in configs])
        milvus_sd = np.array([statistics.stdev(data[dataset]["Milvus"]["querying"][c]) for c in configs])
        blocks_means = np.array([statistics.mean(data[dataset]["Blocks"]["querying"][c]["query"]) for c in configs])
        blocks_sd = np.array([statistics.stdev(data[dataset]["Blocks"]["querying"][c]["query"]) for c in configs])
        # blocks_tot_means = np.array([statistics.mean(data[dataset]["Blocks"]["querying"][c]["total"]) for c in configs])
        blocks_tot_sd = np.array([statistics.stdev(data[dataset]["Blocks"]["querying"][c]["total"]) for c in configs])
        blocks_inv_means = np.array(
            [statistics.mean(data[dataset]["Blocks"]["querying"][c]["invoke"]) for c in configs]
        )
        # blocks_inv_sd = np.array([statistics.stdev(data[dataset]["Blocks"]["querying"][c]["invoke"]) for c in configs])
        blocks_load_means = np.array([statistics.mean(data[dataset]["Blocks"]["querying"][c]["load"]) for c in configs])
        # blocks_load_sd = np.array([statistics.stdev(data[dataset]["Blocks"]["querying"][c]["load"]) for c in configs])

        x = np.arange(len(configs))
        width = 0.4
        axs[1][i].bar(x - width / 2, milvus_means, width, label="Milvus")
        axs[1][i].errorbar(
            x - width / 2,
            milvus_means,
            yerr=milvus_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )
        axs[1][i].bar(x + width / 2, blocks_means, width, label="Blocks: Query")
        axs[1][i].bar(x + width / 2, blocks_inv_means, width, label="Blocks: Invoke", bottom=blocks_means)
        axs[1][i].bar(
            x + width / 2, blocks_load_means, width, label="Blocks: Load", bottom=blocks_means + blocks_inv_means
        )
        axs[1][i].errorbar(
            x + width / 2,
            blocks_load_means + blocks_means + blocks_inv_means,
            yerr=blocks_tot_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )

        axs[1][i].grid(True)
        if i == 1:
            axs[1][i].set_xlabel("Number of Partitions and Query CFs")

        axs[1][i].set_xticks(x)
        axs[1][i].set_xticklabels(configs, rotation=45)
        if i == 0:
            axs[1][i].set_ylabel("Querying Time (s)")
        if i == 2:
            axs[1][i].legend(
                bbox_to_anchor=(1, 0.55, 0.6, 1),
                loc="lower right",
                mode="expand",
                borderaxespad=0,
                ncol=1,
                frameon=False,
            )

    pylab.tight_layout()
    pylab.savefig(dst)


def main():
    data = {}
    for dataset, impl, pretty_dataset, pretty_impl in INPUTS:
        if impl == "milvus":
            indexing, querying = parse_milvus_results(dataset)
        elif impl == "blocks":
            indexing, querying = parse_blocks_results(dataset, 1)
        if pretty_dataset not in data:
            data[pretty_dataset] = {}
        data[pretty_dataset][pretty_impl] = {
            "indexing": indexing,
            "querying": querying,
        }

    pprint(data)
    plot_comparison(data, f"{plots_dir}/milvus_comparison.pdf")


if __name__ == "__main__":
    main()
