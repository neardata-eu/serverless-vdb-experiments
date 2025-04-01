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


CAPSIZE = 3

results_dir = os.path.dirname(os.path.dirname(__file__)) + "/results"
plots_dir = os.path.dirname(os.path.dirname(__file__)) + "/plots/clustering"
os.makedirs(plots_dir, exist_ok=True)

# redundancy, pretty redundancy
INPUTS = [
    (0, "0%"),
    (1, "1%"),
    (2, "2%"),
    (5, "5%"),
]


def parse_dataframe(redundancy):
    df = None
    directory = f"{results_dir}/deep/centroids/{redundancy}/indexing/"
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), "r") as file:
                config = filename.split("_")[-3]
                file_data = json.load(file)
                data = {
                    "redundancy": redundancy,
                    "Partitions": int(config),
                    "Total Vectors": file_data["total_generated_vectors"],
                    "Indexing Time": file_data["total_indexing_centroids"],
                    "Load Dataset Time": file_data["load_dataset_centroids"][0]
                    if file_data["load_dataset_centroids"] != 0
                    else 0,
                }
                df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
    return df


def plot_percentages(df: pd.DataFrame, dst):
    fig, ax = pylab.subplots(1, 2, figsize=(3.33, 1.5))
    # ax2 = ax.twinx()

    sns.barplot(
        data=df[df["redundancy"] != 0],
        x="Partitions",
        y="Indexing Time (%)",
        hue="Redundancy",
        errorbar="sd",
        err_kws={
            "linewidth": 0.5,
        },
        # palette="colorblind",
        ax=ax[0],
        legend=None,
    )
    ax[0].grid(True)
    ax[0].set_ylabel("Time Inc. (%)")
    ax[0].set_xlabel("Number of Partitions")
    ax[0].set_ylim(0, 25)
    # h, l = ax[0].get_legend_handles_labels() # Extracting handles and labels
    # ph = [plt.plot([],marker="", ls="")[0]] # Canvas
    # handles = ph + h
    # labels = ["Redundancy:"] + l  # Merging labels
    # ax[0].legend(
    #     handles, labels,
    #     # title="Redundancy",
    #     alignment="left",
    #     bbox_to_anchor=(0, 0.15, 1, 1),
    #     loc="upper left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=4,
    #     frameon=False,
    # )

    sns.barplot(
        data=df[df["redundancy"] != 0],
        x="Partitions",
        y="Vectors (%)",
        hue="Redundancy",
        errorbar=None,
        # palette="colorblind",
        ax=ax[1],
    )
    ax[1].grid(True)
    ax[1].set_ylabel("Size Inc. (%)")
    ax[1].set_xlabel("Number of Partitions")
    # ax[1].legend(
    #     title="Redundancy",
    #     # alignment="left",
    #     bbox_to_anchor=(1, 0.5),
    #     loc="center left",
    #     # mode="expand",
    #     borderaxespad=0.2,
    #     ncol=1,
    #     frameon=False,
    # )

    h, la = ax[1].get_legend_handles_labels()
    ph = [plt.plot([], marker="", ls="")[0]]  # Canvas
    handles = ph + h
    labels = ["Redundancy:"] + la  # Merging labels
    fig.legend(
        handles,
        labels,
        # title="Redundancy",
        bbox_to_anchor=(0.5, 0.95),
        loc="upper center",
        # mode="expand",
        borderaxespad=0,
        ncol=4,
        frameon=False,
    )
    ax[1].get_legend().remove()

    # sns.pointplot(
    #     data=df[df["redundancy"] != 0],
    #     x="Partitions",
    #     y="Vectors (%)",
    #     hue="Redundancy",
    #     # errorbar="sd",
    #     # palette="colorblind",
    #     ax=ax[1],
    #     legend=None,
    #     dodge=True,
    #     markers=["o", "|", "^"],
    #     markersize=3,
    #     linewidth=0.8,
    # )
    # ax2.set_ylim(0, 100)
    # ax.sharey(ax2)

    pylab.tight_layout()
    pylab.subplots_adjust(top=0.85, bottom=0.25)
    pylab.savefig(dst)


def main():
    all_data = []
    for redundancy, pretty_redundancy in INPUTS:
        data = parse_dataframe(redundancy)
        if data.empty:
            print(f"No data found for redundancy {redundancy}")
            continue
        data["Redundancy"] = pretty_redundancy
        all_data.append(data)

    df = pd.concat(all_data, ignore_index=True)

    pprint(df)

    # Add load dataset time to indexing time
    r0 = df[df["redundancy"] == 0]
    configs = sorted(
        df["Partitions"].unique(),
        key=lambda x: int(x),
    )
    load_dataset_means = {
        config: statistics.mean(r0["Load Dataset Time"][r0["Partitions"] == config]) for config in configs
    }
    print(load_dataset_means)
    for config, load_time in load_dataset_means.items():
        df.loc[(df["redundancy"] != 0) & (df["Partitions"] == config), "Indexing Time"] = (
            df.loc[(df["redundancy"] != 0) & (df["Partitions"] == config), "Indexing Time"] + load_time
        )

    # calculate percentage increase
    baseline = df[df["redundancy"] == 0]
    baseline_means = {
        config: statistics.mean(baseline["Indexing Time"][baseline["Partitions"] == config]) for config in configs
    }
    for config, baseline in baseline_means.items():
        df.loc[(df["redundancy"] != 0) & (df["Partitions"] == config), "Indexing Time (%)"] = (
            (df.loc[(df["redundancy"] != 0) & (df["Partitions"] == config), "Indexing Time"] - baseline)
            / baseline
            * 100
        )
    baseline = df[df["redundancy"] == 0]
    baseline_means = {
        config: statistics.mean(baseline["Total Vectors"][baseline["Partitions"] == config]) for config in configs
    }
    for config, baseline in baseline_means.items():
        df.loc[(df["redundancy"] != 0) & (df["Partitions"] == config), "Vectors (%)"] = (
            (df.loc[(df["redundancy"] != 0) & (df["Partitions"] == config), "Total Vectors"] - baseline)
            / baseline
            * 100
        )

    df.drop(columns=["Load Dataset Time"], inplace=True)
    df.sort_values(by=["Partitions", "redundancy"], inplace=True)
    # print(df)

    for redundancy, data in df.groupby("Redundancy"):
        print(f"Redundancy {redundancy}")
        print(data)

    plot_percentages(df, f"{plots_dir}/redundancy_overhead.pdf")


if __name__ == "__main__":
    main()
