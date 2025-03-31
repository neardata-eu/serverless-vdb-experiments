import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pylab

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
        "legend.handlelength": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0.3,
    }
)

CAPSIZE = 3

results_indexing = {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]}

for v in ["balanced", "unbalanced"]:
    directory = f"../results/deep100k/centroids/0/indexing/{v}"
    
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            parts = filename.split("_")
            if len(parts) < 4:
                continue 
            
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                num_index = data["params"]["num_index"]
                
                results_indexing[v][num_index].append(data["total_indexing_centroids"])
                

means_indexing = {k: {x: np.mean(v) for x, v in d.items()} for k, d in results_indexing.items()}
stds_indexing = {k: {x: np.std(v) for x, v in d.items()} for k, d in results_indexing.items()}

x_labels = [16, 32, 64]
x = np.arange(len(x_labels))
width = 0.35

# Create plots with smaller size (approximately half)
fig1, ax1 = plt.subplots(figsize=(3.33, 2.0))  # Half of original 6.66x4.0
plt.subplots_adjust(top=0.85)  # Slightly more top margin for legend

ax2 = ax1.twinx()

# Get colors from rcParams
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
balanced_color = colors[0]  # '#2166ac'
unbalanced_color = colors[1]  # '#b2182b'

# Adjust bar width for smaller plot
width = 0.25  # Reduced from 0.35

# Plot data with smaller elements
ax1.bar(x - width/2, 
        [means_indexing["balanced"][x_val] for x_val in x_labels], 
        width, 
        yerr=[stds_indexing["balanced"][x_val] for x_val in x_labels],
        color=balanced_color, 
        label='Balanced',
        error_kw={
            'elinewidth': 0.5,  # Thinner error bars
            'capthick': 0.5,
            'capsize': 3  # Smaller caps
        })

ax2.bar(x + width/2, 
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

# Set axis scales and ticks (unchanged)
balanced_ticks = [500, 1000, 1500]
unbalanced_ticks = [5, 10, 15]
ax1.set_yticks(balanced_ticks)
ax2.set_yticks(unbalanced_ticks)
ax1.set_ylim(0, balanced_ticks[-1]*1.30)
ax2.set_ylim(0, unbalanced_ticks[-1]*1.30)

# Axis labels with adjusted font sizes
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('Number of Indexes', color='black', fontsize=8)  # Smaller font
ax1.set_ylabel('Balanced Time (s)', color='black', fontsize=8)
ax2.set_ylabel('Unbalanced Time (s)', color='black', fontsize=8)

# Tick formatting
ax1.tick_params(axis='both', labelsize=7)  # Smaller tick labels
ax2.tick_params(axis='y', labelsize=7)
ax1.tick_params(axis='x', colors='black')
ax1.tick_params(axis='y', colors='black')
ax2.tick_params(axis='y', colors='black')

# Grid and legend
ax1.grid(True, linewidth=0.2, linestyle='-')  # Thinner grid lines
ax2.grid(False)

# Compact legend
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, 
           frameon=False,
           fontsize=7,  # Smaller legend text
           handlelength=0.8,  # Shorter handles
           bbox_to_anchor=(0.5, 1.15),  # Higher position
           loc='upper center',
           ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjusted layout
plt.savefig("../plots/balanced_vs_vanilla_kmeans/balanced_vs_vanilla_indexing_time.pdf", dpi=600, bbox_inches='tight')