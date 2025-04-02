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
        "axes.prop_cycle": cycler("color", ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]),
        "legend.labelspacing": 0.1,
        # "legend.handlelength": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 1,
        "legend.borderpad": 0,
    }
)

# Initialize results structure
results = {p: {r: {"recalls": defaultdict(list)} for r in [0, 1, 2, 5]} for p in [16, 32, 64, 128]}

# Process querying results
for r in [0, 1, 2, 5]:
    directory = f"../results/deep/centroids/{r}/querying/"
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            data = json.load(file)
            n_centroids = data["params"]["num_index"]
            num_centroids_search = data["params"]["num_centroids_search"]
            results[n_centroids][r]["recalls"][num_centroids_search].append(data["recalls_mean"])

# Define plot style parameters
line_styles = ["-", "--", "-.", ":"]

fig, axs = plt.subplots(1, 4, figsize=(7, 1.8))
# Create a separate plot for each p value
for p, ax in zip([16, 32, 64, 128], axs.flatten()):
    # Create a figure

    # Get sorted keys for x-axis
    num_centroids_searches = sorted(results[p][0]["recalls"].keys())
    x = np.array(num_centroids_searches)

    # Plot the main plot (full range, 60-100)
    for i, r in enumerate([0, 1, 2, 5]):
        # Calculate mean recalls
        mean_recalls = [np.mean(results[p][r]["recalls"][n]) for n in num_centroids_searches]

        ax.errorbar(
            x,
            mean_recalls,
            label=f"{r}%",
            linestyle=line_styles[i % len(line_styles)],
            markersize=3,
            linewidth=0.8,
            capsize=2,
            elinewidth=0.5,
            capthick=0.5,
        )

    # Set main plot properties
    ax.set_ylim(60, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_xticks(x)
    # if p in [64, 128]:
    ax.set_xlabel("$N_{\\text{search}}$")
    if p in [16]:
        ax.set_ylabel("Recall (%)")
    ax.set_title("$N=" + str(p) + "$")
    ax.grid(True, linestyle="-", alpha=0.3)

    # Highlight the zoomed region in the main plot
    ax.axhspan(94, 98, facecolor="gray", alpha=0.1)  # Shaded region for 94-98

    # Add an inset plot to zoom in on 94-98
    ax_inset = ax.inset_axes([0.5, 0.15, 0.45, 0.4])  # [x, y, width, height] in axes coordinates
    for i, r in enumerate([0, 1, 2, 5]):
        mean_recalls = [np.mean(results[p][r]["recalls"][n]) for n in num_centroids_searches]
        ax_inset.errorbar(
            x,
            mean_recalls,
            linestyle=line_styles[i % len(line_styles)],
            markersize=3,
            linewidth=0.8,
            capsize=2,
            elinewidth=0.5,
            capthick=0.5,
        )

    # Set inset plot properties to zoom in on 94-98
    ax_inset.set_ylim(95, 99)
    ax_inset.set_yticks([95, 96, 97, 98, 99])
    ax_inset.set_xticks(x)
    ax_inset.grid(True, linestyle="-", alpha=0.3)
    ax_inset.set_title("Zoom: 95-99", fontsize=8, pad=2)  # Add a title to the inset

    # Adjust font sizes for the inset plot and remove axis labels
    ax_inset.tick_params(axis="both", labelsize=7)  # Smaller tick label font size
    ax_inset.set_xlabel("")  # Remove x-axis label
    ax_inset.set_ylabel("")  # Remove y-axis label

    # Add a box around the inset plot
    ax_inset.spines["top"].set_visible(True)
    ax_inset.spines["right"].set_visible(True)


h, la = ax.get_legend_handles_labels()
# Add legend to the main plot
# fig.legend(h, la, title="Redundancy", bbox_to_anchor=(1.05, 0.5), loc="center right", frameon=False, handlelength=2)

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
    ncol=5,
    frameon=False,
)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.78, bottom=0.2)
plt.savefig("../plots/replication_analysis/replication_analysis_recall.pdf")
plt.close()  # Close the figure to free memory
