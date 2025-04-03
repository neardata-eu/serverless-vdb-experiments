import os
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Add this import for tick customization
import json
import numpy as np
from cycler import cycler

# Set up LaTeX style formatting
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
            ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"],
        ),
        "legend.labelspacing": 0.1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0,
    }
)

LAMBDA_GB_SECOND = 0.0000166667
EC2_CENTROIDS_PER_SECOND = 0.00059444444
results = {
    p: {impl: {"indexing_cost": [], "querying_cost": []} for impl in ["centroids", "blocks"]}
    for p in [16, 32, 64, 128]
}

directory = f"../results/deep/centroids/0/indexing/"
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as file:
        data = json.load(file)
        n_centroids = data["params"]["num_index"]
        num_centroids_search = data["params"]["num_centroids_search"]
        results[n_centroids]["centroids"]["indexing_cost"].append(
            EC2_CENTROIDS_PER_SECOND * (data["load_dataset_centroids"][0] + data["global_index_centroids"])
            + sum(map(lambda x: x * LAMBDA_GB_SECOND * 10, data["distribute_vectors_centroids"]))
            + sum(map(lambda x: x * LAMBDA_GB_SECOND * 10, data["generate_index_centroids"]))
        )

directory = f"../results/deep/centroids/0/querying/"
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as file:
        data = json.load(file)
        n_centroids = data["params"]["num_index"]
        num_centroids_search = data["params"]["num_centroids_search"]
        if n_centroids != num_centroids_search:
            continue
        map_times = [query_execution[0] for batch in data["map_queries_times"] for query_execution in batch]
        results[n_centroids]["centroids"]["querying_cost"].append(
            sum(10 * x * LAMBDA_GB_SECOND * 8 for x in map_times)
            + sum(10 * x * LAMBDA_GB_SECOND * 2 for x in data["reduce_queries_times"][0])
        )

directory = f"../results/deep/blocks/indexing/"
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as file:
        data = json.load(file)
        n_centroids = data["params"]["num_index"]
        results[n_centroids]["blocks"]["indexing_cost"].append(
            sum(map(lambda x: x * LAMBDA_GB_SECOND * 10, data["generate_index_blocks"]))
        )

directory = f"../results/deep/blocks/querying/"
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as file:
        data = json.load(file)
        n_centroids = data["params"]["num_index"]
        if int(n_centroids / data["params"]["query_batch_size"]) != 4:
            continue
        map_times = [query_execution[0] for batch in data["map_queries_times"] for query_execution in batch]
        results[n_centroids]["blocks"]["querying_cost"].append(
            sum(10 * x * LAMBDA_GB_SECOND * 8 for x in map_times)
            + sum(10 * x * LAMBDA_GB_SECOND * 2 for x in data["reduce_queries_times"][0])
        )


def prepare_cost_data(results):
    cost_data = {
        "blocks": {"indexing": {}, "querying": {}},
        "centroids": {"indexing": {}, "querying": {}},
        "n_partitions": [16, 32, 64, 128]
    }
    
    for p in cost_data["n_partitions"]:
        for impl in ["blocks", "centroids"]:
            # Indexing costs
            indexing_costs = results[p][impl]["indexing_cost"]
            indexing_mean = np.mean(indexing_costs) if indexing_costs else 0
            cost_data[impl]["indexing"][p] = indexing_mean
            
            # Querying costs
            querying_costs = results[p][impl]["querying_cost"]
            querying_mean = np.mean(querying_costs) if querying_costs else 0
            cost_data[impl]["querying"][p] = querying_mean
    
    return cost_data

# Prepare the data
cost_data = prepare_cost_data(results)

# Create a single plot
fig, ax = plt.subplots(figsize=(3.33, 2.5))

# Set up the X-axis (n_partitions)
n_partitions = cost_data["n_partitions"]
x = np.arange(len(n_partitions))
bar_width = 0.35  # Width of each bar, adjusted to fit two bars side by side

# Plot indexing costs as bars
centroids_indexing = ax.bar(
    x - bar_width/2,
    [cost_data["centroids"]["indexing"][p] for p in n_partitions],
    bar_width,
    label="Partitioning (Centroids)",
    alpha=0.9,
    color="#ef8a62"
)
blocks_indexing = ax.bar(
    x + bar_width/2,
    [cost_data["blocks"]["indexing"][p] for p in n_partitions],
    bar_width,
    label="Partitioning (Blocks)",
    alpha=0.9,
    color="#b2182b"
)

# Plot querying costs as lines with markers
centroids_querying = ax.plot(
    x,
    [cost_data["centroids"]["querying"][p] for p in n_partitions],
    markersize=3,
    linestyle="-",
    label="Querying (Centroids)",
    linewidth=1,
    color="#67a9cf"
)
blocks_querying = ax.plot(
    x,
    [cost_data["blocks"]["querying"][p] for p in n_partitions],
    markersize=3,
    linestyle="--",
    label="Querying (Blocks)",
    linewidth=1,
    color="#2166ac"
)

# Customize the plot
ax.set_xlabel("Number of Partitions ($N$)")
ax.set_ylabel("Cost (USD)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"{p}" for p in n_partitions])

# Add major and minor ticks to the y-axis
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

# Add grid
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.75)
ax.grid(True, which="minor", axis="y", linestyle="--", alpha=0.25)

# Adjust y-axis limits dynamically
valid_costs = (
    [cost_data["centroids"]["indexing"][p] for p in n_partitions] +
    [cost_data["blocks"]["indexing"][p] for p in n_partitions] +
    [cost_data["centroids"]["querying"][p] for p in n_partitions] +
    [cost_data["blocks"]["querying"][p] for p in n_partitions]
)
valid_costs = [cost for cost in valid_costs if cost > 0]
min_cost = min(valid_costs) if valid_costs else 1e-5
max_cost = max(valid_costs) if valid_costs else 1
ax.set_ylim(min_cost * 0.5, max_cost * 2)

# Reorder legend: Indexing Blocks, Querying Blocks, Indexing Centroids, Querying Centroids
handles = [blocks_indexing, centroids_indexing, blocks_querying[0], centroids_querying[0]]
labels = ["Indexing (Blocks)", "Indexing (Centroids)", "Querying (Blocks)", "Querying (Centroids)"]
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    ncol=2,
    frameon=False,
    labelspacing=0.1,
    handletextpad=0.2,
    columnspacing=1,
    borderpad=0
)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.2)
plt.savefig("../plots/blocks_vs_clustering/cost_breakdown.pdf")
plt.close()