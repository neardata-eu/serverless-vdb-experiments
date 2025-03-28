import os
import json
import numpy as np
import matplotlib.pyplot as plt

results_indexing = {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]}
results_generated = {
    str(r): {
        'average': {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]},
        'std_dev': {v: {impl: [] for impl in [16, 32, 64]} for v in ["balanced", "unbalanced"]}
    } for r in [0, 1]
}

for r in [0, 1]:
    for v in ["balanced", "unbalanced"]:
        directory = f"../results/deep100k/centroids/{r}/indexing/{v}"
        
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                parts = filename.split("_")
                if len(parts) < 4:
                    continue 
                
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    num_index = data["params"]["num_index"]
                    
                    if r == 0:
                        results_indexing[v][num_index].append(data["total_indexing_centroids"])
                    
                    results_generated[str(r)]['average'][v][num_index].append(data["avg_generated_vectors"])
                    results_generated[str(r)]['std_dev'][v][num_index].append(data["std_dev_generated_vectors"])

means_indexing = {k: {x: np.mean(v) for x, v in d.items()} for k, d in results_indexing.items()}
stds_indexing = {k: {x: np.std(v) for x, v in d.items()} for k, d in results_indexing.items()}

means_generated = {
    r: {
        'average': {k: {x: np.mean(v) for x, v in d.items()} for k, d in data['average'].items()},
        'std_dev': {k: {x: np.mean(v) for x, v in d.items()} for k, d in data['std_dev'].items()}
    } for r, data in results_generated.items()
}

x_labels = [16, 32, 64]
x = np.arange(len(x_labels))
width = 0.35

fig1, ax1 = plt.subplots(figsize=(10, 6))
for i, (category, color) in enumerate(zip(['balanced', 'unbalanced'], ['blue', 'red'])):
    y = [means_indexing[category][x_val] for x_val in x_labels]
    y_err = [stds_indexing[category][x_val] for x_val in x_labels]
    offset = width * i
    ax1.bar(x + offset, y, width, label=category, color=color, 
            yerr=y_err, capsize=5, alpha=0.7)

ax1.set_xlabel('Number of Indexes')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Indexing Deep100k Time')
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(x_labels)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3)

fig2, ax2 = plt.subplots(figsize=(12, 6))
width = 0.15
colors = {'balanced': 'blue', 'unbalanced': 'green'}

for r_idx, r in enumerate(['0', '1']):
    for v_idx, (category, pattern) in enumerate(zip(['balanced', 'unbalanced'], ['//', '\\\\'])):
        offset = width * (r_idx * 2 + v_idx) - width * 1.5
        y = [means_generated[r]['average'][category][x_val] for x_val in x_labels]
        y_err = [means_generated[r]['std_dev'][category][x_val] for x_val in x_labels]
        
        ax2.bar(x + offset, y, width, 
                label=f'{category} (Threshold 1.0{r})', 
                color=colors[category],
                yerr=y_err, 
                capsize=3, 
                alpha=0.8)

ax2.set_xlabel('Number of Indexes')
ax2.set_ylabel('Number of Vectors')
ax2.set_title('Generated Vectors per partition by Replication Threshold')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()