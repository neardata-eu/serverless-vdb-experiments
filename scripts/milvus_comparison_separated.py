import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import statistics
import os
from cycler import cycler
import pylab
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
        "font.size": 9,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{libertinus}",
                r"\usepackage{newtxmath}",
            ]
        ),
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.linestyle": "-",
        "axes.edgecolor": mpl.rcParams["grid.color"],
        "ytick.direction": "in",
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.titlesize": "medium",
        "axes.titlepad": 4,
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.spines.bottom": False,
        "axes.spines.left": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(
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

INPUTS = [
    ("deep", "blocks", "DEEP10M", "SVDB"),
    ("deep", "milvus", "DEEP10M", "Milvus"),
    ("sift", "blocks", "SIFT10M", "SVDB"),
    ("sift", "milvus", "SIFT10M", "Milvus"),
    ("gist", "blocks", "GIST1M", "SVDB"),
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
            config = filename.split("_")[-3] + "_" + filename.split("_")[-2]
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

    return blocks_indexing, blocks_querying

def plot_indexing(data, dst):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.0))
    handles, labels = None, None
    
    for i, dataset in enumerate(data):
        configs = sorted(
            set().union(data[dataset]["SVDB"]["indexing"].keys(), data[dataset]["Milvus"]["indexing"].keys()),
            key=lambda x: int(x),
        )

        milvus_means = np.array([statistics.mean(data[dataset]["Milvus"]["indexing"][c]) for c in configs]) / 60
        milvus_sd = np.array([statistics.stdev(data[dataset]["Milvus"]["indexing"][c]) for c in configs]) / 60
        blocks_means = np.array([statistics.mean(data[dataset]["SVDB"]["indexing"][c]) for c in configs]) / 60
        blocks_sd = np.array([statistics.stdev(data[dataset]["SVDB"]["indexing"][c]) for c in configs]) / 60

        x = np.arange(len(configs))
        width = 0.4
        milvus_bars = axs[i].bar(x - width / 2, milvus_means, width, label="Milvus")
        axs[i].errorbar(
            x - width / 2,
            milvus_means,
            yerr=milvus_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )
        svdb_bars = axs[i].bar(x + width / 2, blocks_means, width, label="SVDB")
        axs[i].errorbar(
            x + width / 2,
            blocks_means,
            yerr=blocks_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )

        axs[i].set_title(f"{dataset}")
        axs[i].grid(True)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(configs)
        if i == 0:
            axs[i].set_ylabel("Indexing Time (min)")
        if i == 1:
            axs[i].set_xlabel("Number of Partitions")

        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()

    fig.legend(handles, labels, 
               bbox_to_anchor=(0.5, 1.05),  # Moved higher from 0.98 to 1.05
               loc="upper center",
               ncol=2,
               frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{dst}/milvus_indexing.pdf")
    plt.close()

def plot_querying(data, dst):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.0))
    handles, labels = None, None
    
    for i, dataset in enumerate(data):
        configs = sorted(
            set().union(data[dataset]["SVDB"]["querying"].keys(), data[dataset]["Milvus"]["querying"].keys()),
            key=lambda x: tuple(map(int, x.split("_"))),
        )

        milvus_means = np.array([statistics.mean(data[dataset]["Milvus"]["querying"][c]) for c in configs])
        milvus_sd = np.array([statistics.stdev(data[dataset]["Milvus"]["querying"][c]) for c in configs])
        blocks_means = np.array([statistics.mean(data[dataset]["SVDB"]["querying"][c]["query"]) for c in configs])
        blocks_sd = np.array([statistics.stdev(data[dataset]["SVDB"]["querying"][c]["query"]) for c in configs])
        blocks_tot_sd = np.array([statistics.stdev(data[dataset]["SVDB"]["querying"][c]["total"]) for c in configs])
        blocks_inv_means = np.array(
            [statistics.mean(data[dataset]["SVDB"]["querying"][c]["invoke"]) for c in configs]
        )
        blocks_load_means = np.array([statistics.mean(data[dataset]["SVDB"]["querying"][c]["load"]) for c in configs])

        x = np.arange(len(configs))
        width = 0.4
        milvus_bars = axs[i].bar(x - width / 2, milvus_means, width, label="Milvus")
        axs[i].errorbar(
            x - width / 2,
            milvus_means,
            yerr=milvus_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )
        query_bars = axs[i].bar(x + width / 2, blocks_means, width, label="SVDB: Query")
        invoke_bars = axs[i].bar(x + width / 2, blocks_inv_means, width, label="SVDB: Invoke", bottom=blocks_means)
        load_bars = axs[i].bar(
            x + width / 2, blocks_load_means, width, label="SVDB: Load", bottom=blocks_means + blocks_inv_means
        )
        axs[i].errorbar(
            x + width / 2,
            blocks_load_means + blocks_means + blocks_inv_means,
            yerr=blocks_tot_sd,
            capsize=CAPSIZE,
            capthick=0.5,
            elinewidth=0.5,
            fmt="none",
            ecolor="black",
        )

        axs[i].grid(True)
        axs[i].set_title(f"{dataset}")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(configs, rotation=45)
        if i == 0:
            axs[i].set_ylabel("Querying Time (s)")
        if i == 1:
            axs[i].set_xlabel("Number of Partitions and Query CFs")

        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()

    fig.legend(handles, labels,
               bbox_to_anchor=(0.5, 1.05),  # Moved higher from 0.98 to 1.05
               loc="upper center",
               ncol=4,
               frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{dst}/milvus_querying.pdf")
    plt.close()

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
    plot_indexing(data, plots_dir)
    plot_querying(data, plots_dir)

if __name__ == "__main__":
    main()