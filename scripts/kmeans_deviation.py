import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
import pylab
import os, re

# Use Agg backend for LaTeX PGF export
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

def extract_partition_data(directory):
    pattern = re.compile(r"\s(\d+(?:\.\d+)?)(MiB)\s+STANDARD\s+.*?\.ann")
    balanced_data = {}
    unbalanced_data = {}

    for filename in os.listdir(directory):
        n_partitions = filename.split("_")[-1].split(".")[0]
        with open(os.path.join(directory, filename), "r") as file:
            data = file.read()
        if filename.split("_")[0] == "small":
            if n_partitions not in unbalanced_data:
                unbalanced_data[n_partitions] = []

            for match in pattern.finditer(data):
                size = float(match.group(1))
                unbalanced_data[n_partitions].append(size)
        else:
            if n_partitions not in balanced_data:
                balanced_data[n_partitions] = []

            for match in pattern.finditer(data):
                size = float(match.group(1))
                balanced_data[n_partitions].append(size)

    return balanced_data, unbalanced_data

def summarize(label, data_dict):
    print(f"\n--- {label} Partition Stats ---")
    for k in sorted(data_dict.keys(), key=lambda x: int(x)):
        values = np.array(data_dict[k])
        mean = np.mean(values)
        std = np.std(values)
        cv = std / mean if mean else 0
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"{k} clusters: "
              f"min={min_val:.2f}, "
              f"max={max_val:.2f}, "
              f"mean={mean:.2f} MiB, "
              f"std={std:.2f}, "
              f"cv={cv:.2f}, ")

# Load both datasets
results_dir = "../results/kmeans_unbalancing"

balanced_data, unbalanced_data = extract_partition_data(results_dir)

summarize("Balanced", balanced_data)
summarize("Unbalanced", unbalanced_data)

# Prepare for plotting
num_index_values = sorted(set(balanced_data.keys()) & set(unbalanced_data.keys()), key=lambda x: int(x))

balanced_means = [np.mean(balanced_data[k]) for k in num_index_values]
balanced_stds = [np.std(balanced_data[k]) for k in num_index_values]

unbalanced_means = [np.mean(unbalanced_data[k]) for k in num_index_values]
unbalanced_stds = [np.std(unbalanced_data[k]) for k in num_index_values]

# Plotting
fig, ax = plt.subplots(figsize=(3, 2))
positions = np.arange(len(num_index_values))
bar_width = 0.35

ax.bar(positions - bar_width / 2, balanced_means, bar_width, yerr=balanced_stds,
       capsize=3, label="Whole", color="#ef8a62")
ax.bar(positions + bar_width / 2, unbalanced_means, bar_width, yerr=unbalanced_stds,
       capsize=3, label="Limited", color="#b2182b")

ax.set_xticks(positions)
ax.set_xticklabels([str(k) for k in num_index_values])
ax.set_xlabel("Num. Partitions")
ax.set_ylabel("Avg. Partition Size (MiB)")
ax.set_title("Partition Size vs Cluster Count")
ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.75)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("../plots/partition_size_comparison.pdf", dpi=300)
