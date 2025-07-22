import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import statistics
import os
import pandas as pd

from cycler import cycler
import pylab
import seaborn as sns
from collections import defaultdict

pylab.switch_backend("Agg")
mpl.use("pgf")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "font.size": 9,
    "pgf.preamble": "\n".join([
        r"\usepackage{libertinus}",
        r"\usepackage{newtxmath}",
    ]),
    "lines.markersize": 3,
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
        ["#b2182b", "#67a9cf"],
    ),
    "legend.labelspacing": 0.1,
    "legend.handlelength": 1,
    "legend.handletextpad": 0.2,
    "legend.columnspacing": 1,
    "legend.borderpad": 0,
})

CAPSIZE = 0.3

base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results/deep/query_batch")
plots_dir = os.path.join(base_dir, "plots/blocks_vs_clustering")
os.makedirs(plots_dir, exist_ok=True)

SIZES = ["5k"]

INPUTS = []
for size in SIZES:
    INPUTS.append((size, "blocks", f"DEEP{size.upper()}", "Blocks"))
    INPUTS.append((size, "centroids", f"DEEP{size.upper()}", "Clustering"))

def parse_data():
    df = None
    invoke_stats = []

    for dataset, impl, pretty_dataset, pretty_impl in INPUTS:
        path = os.path.join(results_dir, dataset, impl)
        if not os.path.exists(path):
            continue
        for filename in os.listdir(path):
            if filename == "25":
                continue
            try:
                nparts = int(filename.split("_")[-3])
                nfuncs = int(filename.split("_")[-2])
            except:
                continue

            if nparts != 32:
                continue

            with open(os.path.join(path, filename), "r") as file:
                d = json.load(file)
                nsearch = d["params"]["num_centroids_search"] if impl == "centroids" else int(nparts)
                nsearchp = nsearch / int(nparts) * 100
                if nsearchp != 25:
                    continue
                name = pretty_impl if impl == "blocks" else f"{pretty_impl} {int(nsearchp)}%"
                data = {
                    "Size": dataset,
                    "Implementation": pretty_impl,
                    "implementation": impl,
                    "dataset": dataset,
                    "$N$": int(nparts),
                    "Num. Functions": int(nfuncs),
                    "N search": nsearch,
                    "N search %": nsearchp,
                    "Name": name
                }

                # Collect invokes
                map_invokes = [t for sublist in d.get("map_invoke", [[]]) for t in sublist]
                reduce_invokes = [t for sublist in d.get("reduce_invoke", [[]]) for t in sublist]

                

                # Timing sections
                stages = defaultdict(list)
                for function in d["map_queries_times"][0]:
                    stages["total"].append(function[0])
                    stages["download_queries"].append(function[1])
                    stages["download_index"].append(sum(function[2]))
                    stages["load_index"].append(sum(function[3]))
                    stages["query_time"].append(sum(function[4]))
                    stages["reduce_time"].append(function[5])


                invoke_stats.append({
                    "Name": name,
                    "Size": dataset,
                    "map_invoke": map_invokes,
                    "reduce_invoke": reduce_invokes#stages["reduce_time"],
                })



                if impl == "centroids":
                    data_preparation = d["shuffle_centroids_times"][0] + d["map_iterdata_times"][0] + d["create_map_data"][0]
                elif impl == "blocks":
                    data_preparation = d["create_map_data"][0]

                #total = total = d["total_querying_times_mean"]
                total = statistics.mean(stages["total"])
                #total = d["reduce_queries_times"][0][0]
                load_data = sum(statistics.mean(stage) for stage in [stages["download_queries"], stages["download_index"], stages["load_index"]])
                query = sum(statistics.mean(stage) for stage in [stages["query_time"]]) + statistics.mean(stages["reduce_time"])

                for name_section, time in [
                    ("Query Preparation", data_preparation),
                    ("Data Load", load_data),
                    ("Search", query),
                    ("Total", total),
                    ("Reduce", statistics.mean(stages["reduce_time"]))
                ]:
                    data["Type"] = name_section
                    data["Time"] = time
                    df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

    return df, invoke_stats

def print_average_invoke_stats(invoke_stats):
    grouped = defaultdict(lambda: {"map": [], "reduce": []})
    all_map = []
    all_reduce = []

    for entry in invoke_stats:
        key = (entry["Name"], entry["Size"])
        grouped[key]["map"].extend(entry["map_invoke"])
        grouped[key]["reduce"].extend(entry["reduce_invoke"])

        all_map.extend(entry["map_invoke"])
        all_reduce.extend(entry["reduce_invoke"])

    print("\n=== Aggregated map_invoke and reduce_invoke Stats per (Name, Size) ===")
    for (name, size), times in grouped.items():
        def fmt_stats(label, values):
            return (f"{label}: avg={statistics.mean(values):.4f}, std={statistics.stdev(values) if len(values) > 1 else 0:.4f}")

        print(f"{name} | {size}")
        print("  " + fmt_stats("map_invoke", times["map"]))
        print("  " + fmt_stats("reduce_invoke", times["reduce"]))

    print("\n=== Overall Averages ===")
    print(f"Overall map_invoke avg: {statistics.mean(all_map):.4f}")
    print(f"Overall reduce_invoke avg: {statistics.mean(all_reduce):.4f}")

def plot_query_times(df, dst):
    fig, axs = pylab.subplots(2, 2, figsize=(3.33, 2.66))

    types = df["Type"].unique().tolist()

    for type, ax in zip(types, axs.flatten()):
        data = df[df["Type"] == type]
        sns.barplot(
            data,
            x="Size",
            y="Time",
            hue="Name",
            errorbar="sd",
            err_kws={"linewidth": 0.5},
            capsize=CAPSIZE,
            ax=ax,
        )
        ax.grid(True)
        ax.set_title(type)
        ax.set_ylabel("Time (s)" if type in ["Query Preparation", "Search"] else None)
        ax.set_xlabel("Query Batch Size" if type in ["Total", "Search"] else None)

    h, la = ax.get_legend_handles_labels()
    fig.legend(
        h,
        la,
        bbox_to_anchor=(0.5, 0.9),
        loc="lower center",
        borderaxespad=0,
        ncol=5,
        frameon=False,
    )
    for ax in axs.flatten():
        ax.get_legend().remove()

    pylab.tight_layout()
    pylab.subplots_adjust(top=0.83, bottom=0.15)
    pylab.savefig(dst)

def size_to_number(size_str):
    size_str = size_str.lower().replace(" ", "")
    if "k" in size_str:
        try:
            return float(size_str.replace("k", "")) * 1000
        except ValueError:
            pass
    try:
        return float(size_str)
    except ValueError:
        return 0  # fallback



if __name__ == "__main__":

    df, invoke_stats = parse_data()

    try:
        df.drop(df[df["Num. Functions"] == 8].index, inplace=True)
    except:
        pass
    try:
        df.drop(df[df["N search"] == 1].index, inplace=True)
    except:
        pass
    sorter = ["Blocks", "Clustering 25%"]
    df.Name = df.Name.astype("category")
    df.Name = df.Name.cat.set_categories(sorter)
    df["Size_num"] = df["Size"].apply(size_to_number)
    df.sort_values(["Name", "Size_num"], inplace=True)

    print_average_invoke_stats(invoke_stats)

    plot_query_times(df, f"{plots_dir}/query_batch_all_sizes.pdf")