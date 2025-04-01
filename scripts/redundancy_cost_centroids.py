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
            "color", ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
        ),
        "legend.labelspacing": 0.1,
        "legend.handlelength": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0.3,
    }
)

LAMBDA_GB_SECOND = 0.0000166667

results = {
    p: {
        r: {
            "indexing_cost": defaultdict(list),
            "querying_cost": defaultdict(list)
        }
        for r in [0, 1, 2, 5]
    } for p in [16, 32, 64, 128]
}

for r in [0, 1, 2, 5]:
    directory = f"../results/deep/centroids/{r}/indexing/"
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            n_centroids = data["params"]["num_index"]
            num_centroids_search = data["params"]["num_centroids_search"]
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

def prepare_cost_data(results):
    cost_data = {}
    for p in [16, 32, 64, 128]:
        all_num_centroids = set()
        for r in [0, 1, 2, 5]:
            all_num_centroids.update(results[p][r]["indexing_cost"].keys())
            all_num_centroids.update(results[p][r]["querying_cost"].keys())
        num_centroids_search_values = sorted(all_num_centroids)

        cost_data[p] = {
            'indexing': {r: {} for r in [0, 1, 2, 5]},
            'querying': {r: {} for r in [0, 1, 2, 5]},
            'num_centroids_search_values': num_centroids_search_values,
            'replication_labels': [f"{r}%" for r in [0, 1, 2, 5]]
        }

        for r in [0, 1, 2, 5]:
            for ncs in num_centroids_search_values:
                indexing_costs = results[p][r]["indexing_cost"].get(ncs, [])
                indexing_mean = np.mean(indexing_costs) if indexing_costs else 0
                cost_data[p]['indexing'][r][ncs] = indexing_mean

                querying_costs = results[p][r]["querying_cost"].get(ncs, [])
                querying_mean = np.mean(querying_costs) if querying_costs else 0
                cost_data[p]['querying'][r][ncs] = querying_mean

    return cost_data

cost_data = prepare_cost_data(results)

# Modified plotting section with more y-axis ticks
for p in [16, 32, 64, 128]:
    fig, ax = plt.subplots(figsize=(3.33, 2.5))
    ax.set_title(f"Cost Breakdown ({p} partitions)")
    ax.set_xlabel("Replication Factor")
    ax.set_ylabel("Cost (in USD)")
    ax.set_yscale('log')  # Single logarithmic y-axis

    # Add major and minor ticks to the y-axis
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_replications = len(cost_data[p]['replication_labels'])
    bar_width = 0.5  # Width of the indexing cost bars
    x = np.arange(num_replications)  # X positions for each replication factor

    # Plot indexing costs as bars
    indexing_costs = [cost_data[p]['indexing'][r][1] for r in [0, 1, 2, 5]]
    bars = ax.bar(x, indexing_costs, bar_width, label='Indexing', 
                  color=colors[0], alpha=0.75)
    # Plot querying costs as lines with markers
    for idx, ncs in enumerate(cost_data[p]['num_centroids_search_values'], start=1):
        querying_costs = [cost_data[p]['querying'][r][ncs] for r in [0, 1, 2, 5]]
        ax.plot(x, querying_costs, marker='o', markersize=4, 
                label=f'Querying {ncs} centroids', 
                color=colors[idx % len(colors)], linewidth=1.5)
    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(cost_data[p]['replication_labels'])
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.75)
    ax.grid(True, which='minor', axis='y', linestyle='--', alpha=0.75)
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust y-axis limits dynamically
    valid_costs = ([cost for cost in indexing_costs] + 
                   [cost_data[p]['querying'][r][ncs] 
                    for r in [0, 1, 2, 5] 
                    for ncs in cost_data[p]['num_centroids_search_values']])
    valid_costs = [cost for cost in valid_costs if cost > 0]
    min_cost = min(valid_costs)
    max_cost = max(valid_costs)
    ax.set_ylim(min_cost * 0.75, max_cost * 2)

    plt.tight_layout()
    plt.savefig(f'../plots/replication_analysis/cost_breakdown_{p}_partitions.pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()