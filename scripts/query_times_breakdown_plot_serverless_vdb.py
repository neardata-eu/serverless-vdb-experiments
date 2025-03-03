import json
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser(description="Plot a stacked bar of all query stages")
parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets (deep100M, deep, gist, sift)")
parser.add_argument("--version", required=True, help="1 Will take everything into consideration\n2 Will discard lambda invokation and results fetching\n3 Will also discard index download and load times")
parser.add_argument("--scale", action="store_true", default=False, help="Toggle show scale-up comparison plot (deep100M vs deep10M)")
parser.add_argument("--plot_export_dest", required=False, help="Name of the destination folder for the plots. If not provided, plots will be shown, but not saved.")
args = parser.parse_args()

datasets = args.datasets
impl = "blocks"

if int(args.version) == 1:
    data_stages = ["shuffle", "map_iterdata", "download_queries", "download_index", "load_index", "query_time", "reduce_time", "reduce_iterdata", "final_reduce", "lambda_invoke_and_result_fetch"]
elif int(args.version) == 2:
    data_stages = ["shuffle", "map_iterdata", "download_queries", "download_index", "load_index", "query_time", "reduce_time", "reduce_iterdata", "final_reduce"]
elif int(args.version) == 3:
    data_stages = ["shuffle", "map_iterdata", "download_queries", "query_time", "reduce_time", "reduce_iterdata", "final_reduce"]

for dataset in datasets:
    directory = f"../results/{dataset}/{impl}/querying/"
    time_sum = defaultdict(list)
    if int(args.version) == 1:
        stage_times = {stage: defaultdict(list) for stage in data_stages[:-1]}
    else:
        stage_times = {stage: defaultdict(list) for stage in data_stages}
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            n_centroids = filename.split('_')[-3] + "_" + filename.split('_')[-2]
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                time_sum[n_centroids].append(data['total_querying_times_mean'])
                stage_times["shuffle"][n_centroids].append(0)
                stage_times["map_iterdata"][n_centroids].append(data['map_iterdata_times'][0])
                for function in data["map_queries_times"][0]:
                    stage_times["download_queries"][n_centroids].append(function[1])
                    if int(args.version) in [1, 2]:
                        stage_times["download_index"][n_centroids].append(sum(function[2]))
                        stage_times["load_index"][n_centroids].append(sum(function[3]))
                    stage_times["query_time"][n_centroids].append(sum(function[4]))
                    stage_times["reduce_time"][n_centroids].append(function[5])
                stage_times["reduce_iterdata"][n_centroids].append(data["reduce_iterdata_times"][0])
                stage_times["final_reduce"][n_centroids].append(data["reduce_queries_times"][0][0])

    if time_sum:
        centroids_sorted = sorted(time_sum.keys(), key=lambda x: tuple(map(int, x.split('_'))))
        x_labels = centroids_sorted
        x = np.arange(len(x_labels))
        width = 0.6
        
        if int(args.version) == 1:
            stage_avg = {stage: [statistics.mean(stage_times[stage][c]) for c in centroids_sorted] for stage in data_stages[:-1]}
        else:
            stage_avg = {stage: [statistics.mean(stage_times[stage][c]) for c in centroids_sorted] for stage in data_stages}
        
        total_querying_avg = [statistics.mean(time_sum[c]) for c in centroids_sorted]
        summed_stages = np.sum([np.array(stage_avg[stage]) for stage in data_stages[:-1]], axis=0)
        if int(args.version) == 1:
            stage_avg["lambda_invoke_and_result_fetch"] = total_querying_avg - summed_stages

        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(x_labels))
        
        for stage in data_stages:
            values = np.array(stage_avg[stage])
            ax.bar(x, values, width, label=stage, bottom=bottom)
            bottom += values
        
        ax.set_xlabel("Config")
        ax.set_ylabel("Query Time (s)")
        ax.set_title(f"{str.upper(dataset)} Stacked Query Time")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()
        plt.tight_layout()
        if args.plot_export_dest:
            plt.savefig(f"{args.plot_export_dest}/reakdown_query_{dataset}_{args.version}.png")
        else:
            plt.show()


# Define the dataset directories   
comparison_files = {
    "deep100M": "deep100M/blocks/querying/",
    "deep": "deep/blocks/querying/"
}

comparison_files = {
    "deep100M": "deep100M/blocks/querying/",
    "deep": "deep/blocks/querying/"
}

result_dirs = ["results"]

comparison_data = {dataset: defaultdict(list) for dataset in comparison_files}
time_sum = defaultdict(list)
for dataset, directory in comparison_files.items():
    avg_time = 0
    count = 0
    directory = "../results/"+directory
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if dataset == "deep" and not filename.startswith("results_deep_blocks_32_8_"):
                continue 
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                time_sum[dataset].append(data['total_querying_times_mean'])
                comparison_data[dataset]["shuffle"].append(0)
                comparison_data[dataset]["map_iterdata"].append(data['map_iterdata_times'][0])
                for function in data["map_queries_times"][0]:
                    comparison_data[dataset]["download_queries"].append(function[1])
                    if int(args.version) in [1, 2]:
                        comparison_data[dataset]["download_index"].append(sum(function[2]))
                        comparison_data[dataset]["load_index"].append(sum(function[3]))
                    comparison_data[dataset]["query_time"].append(sum(function[4]))
                    comparison_data[dataset]["reduce_time"].append(function[5])
                comparison_data[dataset]["reduce_iterdata"].append(data["reduce_iterdata_times"][0])
                comparison_data[dataset]["final_reduce"].append(data["reduce_queries_times"][0][0])

# Compute averages
stage_avg = {dataset: {stage: statistics.mean(values) for stage, values in stages.items()} for dataset, stages in comparison_data.items()}

if int(args.version) == 1:
    # Compute lambda_init
    for dataset in comparison_files:
        total_querying_avg = statistics.mean(time_sum[dataset])

        stage_avg[dataset]["lambda_invoke_and_result_fetch"] = total_querying_avg - sum(stage_avg[dataset].values())

# Plot stacked bars
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comparison_files))
width = 0.6
bottom = np.zeros(len(comparison_files))

for stage in data_stages:
    values = np.array([stage_avg[dataset][stage] for dataset in comparison_files])
    ax.bar(x, values, width, label=stage, bottom=bottom)
    bottom += values

ax.set_xlabel("Dataset")
ax.set_ylabel("Avg Query Time (s)")
ax.set_title("Stacked Query Time Breakdown: deep100M 320_80 vs deep 32_8")
ax.set_xticks(x)
ax.set_xticklabels(["deep100M 320_80", "deep 32_8"])
ax.legend()

plt.tight_layout()
if args.plot_export_dest:
    plt.savefig(f"{args.plot_export_dest}/breakdown_query_deep100M_vs_deep10M_version_{args.version}.png")
else:
    plt.show()
