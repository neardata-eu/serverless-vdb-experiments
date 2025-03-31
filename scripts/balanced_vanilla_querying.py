import os
from collections import defaultdict
import statistics
import json

# Initialize data structures
results = {
    "balanced": {
        "querying_values": defaultdict(list),  # Store all query times for std dev calculation
        "recall": defaultdict(list)
    },
    "unbalanced": {
        "querying_values": defaultdict(list),  # Store all query times for std dev calculation
        "recall": defaultdict(list)
    }
}

# Process data
for implementation in ["balanced", "unbalanced"]:
    directory = f"../results/deep100k/centroids/0/querying/{implementation}"
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                n_centroids = data["params"]["num_index"]
                num_centroids_search = data["params"]["num_centroids_search"]
                key = (n_centroids, num_centroids_search)
                # Store all query times (not just mean) for std dev calculation
                results[implementation]["querying_values"][key].extend(data["total_querying_times"])
                results[implementation]["recall"][key].append(data["recalls_mean"])

# Sort configurations by n_centroids then num_centroids_search
all_configs = set(results["balanced"]["querying_values"].keys()).union(
              set(results["unbalanced"]["querying_values"].keys()))
sorted_configs = sorted(all_configs, key=lambda x: (x[0], x[1]))

# Generate LaTeX table
latex_output = """\\begin{table}[ht]
\\centering
\\caption{Query Performance Comparison on deep100k}
\\label{tab:query_performance}
\\begin{tabular}{r@{\\hspace{2pt}}r@{~~}rrrrrr}
\\toprule
\\multicolumn{2}{c}{Params} & \\multicolumn{3}{c}{Balanced} & \\multicolumn{3}{c}{Unbalanced} \\\\
\\cmidrule(r){1-2} \\cmidrule(lr){3-5} \\cmidrule(l){6-8}
$N$ & $N_{\\text{search}}$ & Time (s) & $\\pm\\sigma$ & Recall & Time (s) & $\\pm\\sigma$ & Recall \\\\
\\midrule
"""

prev_K = None
for i, config in enumerate(sorted_configs):
    K, K_search = config
    
    # Add midrule when K changes (except after first row)
    if prev_K is not None and K != prev_K:
        latex_output += "\\midrule\n"
    prev_K = K
    
    # Get results or mark as missing
    def get_stats(impl):
        stats = {"time": "---", "std": "---", "recall": "---"}
        if config in results[impl]["querying_values"] and results[impl]["querying_values"][config]:
            query_times = results[impl]["querying_values"][config]
            stats["time"] = f"{statistics.mean(query_times):.3f}"
            stats["std"] = f"{statistics.stdev(query_times):.3f}" if len(query_times) > 1 else "---"
        if config in results[impl]["recall"] and results[impl]["recall"][config]:
            stats["recall"] = f"{statistics.mean(results[impl]['recall'][config]):.2f}"
        return stats
    
    # Balanced results
    b_stats = get_stats("balanced")
    # Unbalanced results
    u_stats = get_stats("unbalanced")
    
    latex_output += f"{K} & {K_search} & {b_stats['time']} & {b_stats['std']} & {b_stats['recall']} & {u_stats['time']} & {u_stats['std']} & {u_stats['recall']} \\\\\n"
    
    # Add special rule when K == K_search (optional)
    if K == K_search and i < len(sorted_configs)-1 and sorted_configs[i+1][0] == K:
        latex_output += "\\cmidrule(r){1-2}\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\n"

latex_output += """\\bottomrule
\\end{tabular}
\\end{table}"""

print(latex_output)