import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pylab
from pprint import pprint
import pandas as pd
import seaborn as sns

# Set up LaTeX formatting
pylab.switch_backend("Agg")  # Use Agg backend for testing
mpl.use("pgf")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pgf.texsystem": "pdflatex",
        "font.size": 8,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{libertinus}",
                r"\usepackage{newtxmath}",
            ]
        ),
        "lines.markersize": 3,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.linestyle": "-",
        "axes.edgecolor": mpl.rcParams["grid.color"],
        "ytick.direction": "out",  # Ensure ticks are drawn outward
        "xtick.direction": "out",
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
            ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"],
        ),
        "legend.labelspacing": 0.5,
        "legend.handlelength": 1,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 2,
        "legend.borderpad": 0,
    }
)

# Load and process data
results_indexing = {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]}
results_vectors = {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]}

for v in ["balanced", "unbalanced"]:
    directory = f"../results/deep100k/centroids/0/indexing/{v}"
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), "r") as file:
                data = json.load(file)
                num_index = data["params"]["num_index"]
                results_indexing[v][num_index].append(data["total_indexing_centroids"])
                results_vectors[v][num_index].append(data["generated_vectors"])

# Calculate means and stds
means_indexing = {k: {x: np.mean(v) for x, v in d.items()} for k, d in results_indexing.items()}
stds_indexing = {k: {x: np.std(v) for x, v in d.items()} for k, d in results_indexing.items()}

# Plot setup
x_labels = [16, 32, 64]
x = np.arange(len(x_labels))
width = 0.35
# fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.4))
fig, axs = plt.subplots(1, 2, figsize=(4, 1.4))
ax = axs[0]

# Set logarithmic scale
ax.set_yscale("log", base=10)

# Plot bars with error bars
ax.bar(
    x - width / 2,
    [means_indexing["unbalanced"][x_val] for x_val in x_labels],
    width,
    yerr=[stds_indexing["unbalanced"][x_val] for x_val in x_labels],
    label="Unbalanced",
    error_kw={"elinewidth": 0.5, "capthick": 0.5, "capsize": 3},
)

ax.bar(
    x + width / 2,
    [means_indexing["balanced"][x_val] for x_val in x_labels],
    width,
    yerr=[stds_indexing["balanced"][x_val] for x_val in x_labels],
    label="Balanced",
    error_kw={"elinewidth": 0.5, "capthick": 0.5, "capsize": 3},
)


# Configure axes
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Num. Partitions ($N$)")
ax.set_ylabel("Partitioning Time (s)")

# Determine the range of the data to set appropriate y-axis limits
y_min = min(min(means_indexing["balanced"].values()), min(means_indexing["unbalanced"].values()))
y_max = max(max(means_indexing["balanced"].values()), max(means_indexing["unbalanced"].values()))
ax.set_ylim(y_min / 2, y_max * 2)

major_ticks = [10, 100, 1000]
minor_ticks = [i * 10**j for j in range(1, 3) for i in range(1, 10)] + [1000, 2000]
ax.set_yticks(major_ticks, minor=False)
ax.set_yticks(minor_ticks, minor=True)

# Use a proper logarithmic formatter for the y-axis
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

# Customize tick mark appearance to ensure visibility
ax.tick_params(axis="y", which="major", length=6, width=1, direction="out", color="black")
ax.tick_params(axis="y", which="minor", length=3, width=0.8, direction="out", color="black")

ax.grid(True, which="both", linewidth=0.2, linestyle="-", alpha=0.75)

# Legend
fig.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="upper center", ncol=2)


time_df = None
for v, impl_data in means_indexing.items():
    for npart, npart_data in impl_data.items():
        data = {
            "K-means": v,
            "Partitions": npart,
            "Indexing Time": npart_data,
        }
        time_df = pd.concat([time_df, pd.DataFrame(data, index=[0])], ignore_index=True)

print(time_df)
balanced = time_df[time_df["K-means"] == "balanced"]["Indexing Time"].to_numpy()
unbalanc = time_df[time_df["K-means"] == "unbalanced"]["Indexing Time"].to_numpy()
speeedup = balanced / unbalanc
print(speeedup)


# pprint(results_vectors)
df = None
for v, impl_data in results_vectors.items():
    for npart, npart_data in impl_data.items():
        for rep in npart_data:
            for i, part in enumerate(rep):
                data = {
                    "K-means": v,
                    "Partitions": npart,
                    "Func. Num.": i,
                    "Vectors per\nPartition ($\\times10^3$)": part / 1000,
                }
                df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
print(df)

df = df.groupby(["K-means", "Partitions", "Func. Num."]).mean()

means = df.groupby(["K-means", "Partitions"]).mean()
sds = df.groupby(["K-means", "Partitions"]).std()
print(means)
print(sds)

# df.sort_values("K-means", ascending=False)

sns.stripplot(
    data=df,
    x="Partitions",
    y="Vectors per\nPartition ($\\times10^3$)",
    hue="K-means",
    hue_order=["unbalanced", "balanced"],
    # errorbar="sd",
    # err_kws={
    #     "linewidth": 0.5,
    # },
    # palette="colorblind",
    size=1,
    dodge=True,
    ax=axs[1],
    legend=None,
)
axs[1].grid(True)
axs[1].set_ylim(0, 10)
axs[1].set_xlabel("Num. Partitions ($N$)")


plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.25)
plt.savefig("../plots/balanced_vs_vanilla_kmeans/balanced_vs_vanilla_indexing_time.pdf", dpi=600)
plt.close()
