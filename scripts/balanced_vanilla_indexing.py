import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pylab

# Set up LaTeX formatting
pylab.switch_backend("Agg")  # Use Agg backend for testing
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
        "ytick.direction": "out",  # Ensure ticks are drawn outward
        "xtick.direction": "out",
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.titlesize": "medium",
        "axes.titlepad": 4,
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(
            "color",
            ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"],
        ),
        "legend.labelspacing": 0.1,
        "legend.handlelength": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0.3,
    }
)

# Load and process data
results_indexing = {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]}

for v in ["balanced", "unbalanced"]:
    directory = f"../results/deep100k/centroids/0/indexing/{v}"
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                num_index = data["params"]["num_index"]
                results_indexing[v][num_index].append(data["total_indexing_centroids"])

# Calculate means and stds
means_indexing = {k: {x: np.mean(v) for x, v in d.items()} for k, d in results_indexing.items()}
stds_indexing = {k: {x: np.std(v) for x, v in d.items()} for k, d in results_indexing.items()}

# Plot setup
x_labels = [16, 32, 64]
x = np.arange(len(x_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(3.33, 2.0))
plt.subplots_adjust(top=0.85)

# Get colors
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
balanced_color = colors[0]
unbalanced_color = colors[1]

# Set logarithmic scale
ax.set_yscale('log', base=10)

# Plot bars with error bars
ax.bar(x - width/2, 
       [means_indexing["balanced"][x_val] for x_val in x_labels], 
       width,
       yerr=[stds_indexing["balanced"][x_val] for x_val in x_labels],
       color=balanced_color,
       label='Balanced',
       error_kw={
           'elinewidth': 0.5,
           'capthick': 0.5,
           'capsize': 3
       })

ax.bar(x + width/2,
       [means_indexing["unbalanced"][x_val] for x_val in x_labels],
       width,
       yerr=[stds_indexing["unbalanced"][x_val] for x_val in x_labels],
       color=unbalanced_color,
       label='Unbalanced',
       error_kw={
           'elinewidth': 0.5,
           'capthick': 0.5,
           'capsize': 3
       })

# Configure axes
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Number of Indexes', fontsize=8)
ax.set_ylabel('Indexing Time (s)', fontsize=8)

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
ax.tick_params(axis='y', which='major', length=6, width=1, direction='out', color='black')
ax.tick_params(axis='y', which='minor', length=3, width=0.8, direction='out', color='black')

ax.grid(True, which='both', linewidth=0.2, linestyle='-', alpha=0.75)

# Legend
ax.legend(frameon=False, fontsize=7, handlelength=0.8,
          bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig("../plots/balanced_vs_vanilla_kmeans/balanced_vs_vanilla_indexing_time.pdf", dpi=600, bbox_inches='tight')
plt.close()