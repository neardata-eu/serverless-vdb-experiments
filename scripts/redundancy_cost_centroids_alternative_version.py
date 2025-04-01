import os
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
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
        "font.size": 9,  # footnote/caption size 9pt for paper
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
            "color", ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
        ),
        "legend.labelspacing": 0.1,
        "legend.handlelength": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0.3,
    }
)

LAMBDA_GB_SECOND=0.0000166667

# Initialize results structure
results = {
    p: {
        r: {
            "indexing_cost": defaultdict(list),
            "querying_cost": defaultdict(list)
        }
        for r in [0, 1, 2, 5]
    } for p in [16, 32, 64, 128]
}

# Process querying results
for r in [0, 1, 2, 5]:
    directory = f"../results/deep/centroids/{r}/indexing/"
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            n_centroids = data["params"]["num_index"]
            num_centroids_search = data["params"]["num_centroids_search"]
            # Extract all first query times across batches and executions
            results[n_centroids][r]["indexing_cost"][num_centroids_search].append(
                sum(map(lambda x: x * LAMBDA_GB_SECOND * 10, data["distribute_vectors_centroids"])) + 
                sum(map(lambda x: x * LAMBDA_GB_SECOND * 10, data["generate_index_centroids"]))
            )

    directory = f"../results/deep/centroids/{r}/querying/"
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            n_centroids = data["params"]["num_index"]
            num_centroids_search = data["params"]["num_centroids_search"]
            map_times = [query_execution[0] for batch in data["map_queries_times"] for query_execution in batch]
            results[n_centroids][r]["querying_cost"][num_centroids_search].append(
                sum(10 * x * LAMBDA_GB_SECOND * 8 for x in map_times) +
                sum(10 * x * LAMBDA_GB_SECOND * 2 for x in data["reduce_execution_times"])
            )

# Prepare data for plotting with NaN handling
def prepare_cost_data(results):
    cost_data = {}
    for p in [16, 32, 64, 128]:
        # Get all unique num_centroids_search values for this partition
        all_num_centroids = set()
        for r in [0, 1, 2, 5]:
            all_num_centroids.update(results[p][r]["indexing_cost"].keys())
            all_num_centroids.update(results[p][r]["querying_cost"].keys())
        num_centroids_search_values = sorted(all_num_centroids)

        cost_data[p] = {
            'indexing': {r: {} for r in [0, 1, 2, 5]},  # Nested dict: {replication: {num_centroids_search: cost}}
            'querying': {r: {} for r in [0, 1, 2, 5]},
            'num_centroids_search_values': num_centroids_search_values,
            'replication_labels': [f"{r}%" for r in [0, 1, 2, 5]]
        }

        for r in [0, 1, 2, 5]:
            for ncs in num_centroids_search_values:
                # Indexing cost
                indexing_costs = results[p][r]["indexing_cost"].get(ncs, [])
                indexing_mean = np.mean(indexing_costs) if indexing_costs else 0
                cost_data[p]['indexing'][r][ncs] = indexing_mean

                # Querying cost
                querying_costs = results[p][r]["querying_cost"].get(ncs, [])
                querying_mean = np.mean(querying_costs) if querying_costs else 0
                cost_data[p]['querying'][r][ncs] = querying_mean

    return cost_data

cost_data = prepare_cost_data(results)

# Create grouped bar plots for each partition count
for p in [16, 32, 64, 128]:
    plt.figure(figsize=(3.5, 2.5))  # Increase figure width to accommodate more bars
    plt.title(f"Cost Breakdown ({p} partitions)")
    plt.xlabel("Replication Factor")
    plt.ylabel("Cost")

    # Get the colors from your prop_cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of replication factors and num_centroids_search values
    num_replications = len(cost_data[p]['replication_labels'])
    num_centroids_search_values = len(cost_data[p]['num_centroids_search_values'])
    bar_width = 0.15  # Width of each sub-bar
    group_width = bar_width * num_centroids_search_values  # Total width of each group

    # X positions for each group (one group per replication factor)
    x = np.arange(num_replications)

    # Plot bars for each num_centroids_search value
    for idx, ncs in enumerate(cost_data[p]['num_centroids_search_values']):
        # Calculate the offset for this num_centroids_search value
        offset = (idx - (num_centroids_search_values - 1) / 2) * bar_width

        # Indexing costs
        indexing_costs = [cost_data[p]['indexing'][r][1] for r in [0, 1, 2, 5]]
        bars_indexing = plt.bar(x + offset, 
                                indexing_costs, 
                                bar_width,
                                label=f'Indexing (ncs={ncs})' if idx == 0 else "",
                                color=colors[idx % len(colors)],
                                edgecolor='none')

        # Querying costs (stacked on top of indexing)
        querying_costs = [cost_data[p]['querying'][r][ncs] for r in [0, 1, 2, 5]]
        bars_querying = plt.bar(x + offset, 
                                querying_costs, 
                                bar_width,
                                bottom=indexing_costs,
                                label=f'Querying (ncs={ncs})' if idx == 0 else "",
                                color=colors[(idx + 1) % len(colors)],
                                edgecolor='none')

        # Add value labels on top of each bar
        for i in range(len(x)):
            total_height = indexing_costs[i] + querying_costs[i]
            if not np.isnan(total_height):  # Only add label if value is valid
                plt.text(x[i] + offset, total_height, 
                         f"{total_height:.4f}", 
                         ha='center', 
                         va='bottom',
                         fontsize=7,
                         rotation=90)  # Rotate labels to fit

    # Set x-axis labels and ticks
    plt.xticks(x, cost_data[p]['replication_labels'])

    # Add grid and legend
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    # Adjust y-axis limits with some padding, ignoring NaN values
    valid_heights = []
    for ncs in cost_data[p]['num_centroids_search_values']:
        for r in [0, 1, 2, 5]:
            total = cost_data[p]['indexing'][r][1] + cost_data[p]['querying'][r][ncs]
            if not np.isnan(total):
                valid_heights.append(total)
    if valid_heights:  # Only set limits if we have valid data
        max_value = max(valid_heights)
        plt.ylim(0, max_value * 1.3)  # Increased padding to accommodate rotated labels

    plt.tight_layout(pad=0.5)
    plt.savefig(f'../plots/replication_analysis/cost_breakdown_{p}_partitions.pdf', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()