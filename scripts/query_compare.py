import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json

import statistics
import os
import pandas as pd

from cycler import cycler
import pylab
# from matplotlib.collections import LineCollection

# import math
from pprint import pprint
import seaborn as sns
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
        "legend.borderpad": 0,
    }
)


CAPSIZE = 0.3

results_dir = os.path.dirname(os.path.dirname(__file__)) + "/results"
plots_dir = os.path.dirname(os.path.dirname(__file__)) + "/plots/blocks_vs_clustering"
os.makedirs(plots_dir, exist_ok=True)

# dataset name, implementation, pretty dataset name, pretty implementation name
INPUTS = [
    ("deep", "blocks", "DEEP10M", "Blocks"),
    ("deep", "centroids", "DEEP10M", "Clustering"),
    # ("deep1M", "blocks", "DEEP1M", "Blocks"),
    # ("deep1M", "centroids", "DEEP1M", "Clustering"),
    # ("deep100k", "blocks", "DEEP100k", "Blocks"),
    # ("deep100k", "centroids", "DEEP100k", "Clustering"),
]


def parse_data():
    df = None
    for dataset, impl, pretty_dataset, pretty_impl in INPUTS:
        if dataset == "deep100k" and impl == "centroids":
            path = f"{results_dir}/{dataset}/{impl}/0/querying/unbalanced"
        elif impl == "blocks":
            path = f"{results_dir}/{dataset}/{impl}/querying/"
        else:
            path = f"{results_dir}/{dataset}/{impl}/0/querying"
        if not os.path.exists(path):
            print(path)
            continue

        for filename in os.listdir(path):
            nparts = filename.split("_")[-3]
            nfuncs = filename.split("_")[-2]
            with open(os.path.join(path, filename), "r") as file:
                d = json.load(file)
                nsearch = d["params"]["num_centroids_search"] if impl == "centroids" else int(nparts)
                nsearchp = nsearch / int(nparts) * 100
                data = {
                    "Dataset": pretty_dataset,
                    "dataset": dataset,
                    "Implementation": pretty_impl,
                    "implementation": impl,
                    "$N$": int(nparts),
                    "Num. Functions": int(nfuncs),
                    "N search": nsearch,
                    "N search %": nsearchp,
                    "Name": pretty_impl if impl == "blocks" else f"{pretty_impl} {int(nsearchp)}%"
                }
                total = d["total_querying_times_mean"]
                stages = defaultdict(list)
                stages["map_iterdata"].append(d["map_iterdata_times"][0])
                for function in d["map_queries_times"][0]:
                    stages["download_queries"].append(function[1])
                    stages["download_index"].append(sum(function[2]))
                    stages["load_index"].append(sum(function[3]))
                    stages["query_time"].append(sum(function[4]))
                    stages["reduce_time"].append(function[5])
                stages["reduce_iterdata"].append(d["reduce_iterdata_times"][0])
                stages["final_reduce"].append(d["reduce_queries_times"][0][0])

                invoke = total - sum(statistics.mean(stage) for stage in stages.values())
                load_index = sum(statistics.mean(stage) for stage in [stages["download_index"], stages["load_index"]])

                total = d["total_querying_times_mean"]
                query = total - invoke - load_index
                # startup = total - query
                for name, time in [
                    ("Invoke", invoke),
                    ("Data Load", load_index),
                    ("Query", query),
                    ("Total", total),
                ]:
                    data["Type"] = name
                    data["Time"] = time
                    # data["Total Time"] = total
                    # data["Query Time"] = query
                    # data["Startup Time"] = startup
                    # data["Invoke Time"] = invoke
                    # data["Load Index Time"] = load_index
                    df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
    return df


def plot_query_times(df, dst):
    fig, axs = pylab.subplots(2, 2, figsize=(3.33, 2.6))

    ax = axs
    data = df
    impls = df["Implementation"].unique().tolist()
    types = df["Type"].unique().tolist()
    # datasets.reverse()

    # for impl, ax in zip(impls, axs.flatten()):
    #     data = df[df["Implementation"] == impl]
    #     sns.barplot(
    #         data,
    #         x="$N$",
    #         y="Time",
    #         hue="Type",
    #         errorbar="sd",
    #         err_kws={
    #             "linewidth": 0.5,
    #         },
    #         capsize=CAPSIZE,
    #         ax=ax,
    #     )

    #     ax.grid(True)
    #     ax.set_title(impl)
    #     ax.set_ylabel("Query Time (s)")
    #     ax.set_xlabel("Num. Partitions ($N$)")

    for type, ax in zip(types, axs.flatten()):
        data = df[df["Type"] == type]
        sns.barplot(
            data,
            x="$N$",
            y="Time",
            hue="Name",
            errorbar="sd",
            err_kws={
                "linewidth": 0.5,
            },
            capsize=CAPSIZE,
            ax=ax,
        )
        ax.grid(True)
        ax.set_title(type)
        if type in ["Invoke", "Query"]:
            ax.set_ylabel("Time (s)")
        else:
            ax.set_ylabel(None)
        if type in ["Total", "Query"]:
            ax.set_xlabel("Num. Partitions ($N$)")
        else:
            ax.set_xlabel(None)

    h, la = ax.get_legend_handles_labels()
    fig.legend(
        h,
        la,
        bbox_to_anchor=(0.5, 0.9),
        loc="lower center",
        title=None,
        # mode="expand",
        borderaxespad=0,
        ncol=3,
        frameon=False,

    )
    for ax in axs.flatten():
        ax.get_legend().remove()
    # axs[0].get_legend().remove()
    # axs[1].get_legend().remove()
    # axs[0].sharey(axs[1])

    pylab.tight_layout()
    pylab.subplots_adjust(
        top=0.83,
        bottom=0.15,
        # left=0.15,
    )
    pylab.savefig(dst)


def main():
    df = parse_data()
    df.drop(df[df["Num. Functions"] == "8"].index, inplace=True)
    df.drop(df[df["N search"] == 1].index, inplace=True)
    # df["N search %"] = (df['N search'] / df['$N$'] * 100).astype(int).astype(str) + "%"
    sorter = ["Blocks", "Clustering 100%", "Clustering 75%", "Clustering 50%", "Clustering 25%"]
    df.Name = df.Name.astype("category")
    df.Name = df.Name.cat.set_categories(sorter)

    df.sort_values(["Name"])
    # df["Name"] = df["Implementation"] if df["Implementation"] == "Blocks" else "Clustering"
    # df = df[df["N search %"] == 100]

    # df["Name"] = df["Implementation"] + " " + df["N search %"].astype(int).astype(str)
    print(df)
    # return
    plot_query_times(df, f"{plots_dir}/querying.pdf")


if __name__ == "__main__":
    main()
