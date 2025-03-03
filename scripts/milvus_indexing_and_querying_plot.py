import os
import json
import re
import statistics
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_milvus_results(dataset):
    milvus_indexing = defaultdict(list)
    milvus_querying = defaultdict(list)
    
    for query_functions in [4, 8]:
        directory = f"../results/{dataset}/milvus{query_functions}qf/"
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.split(".")[1] != "txt":
                    continue
                n_centroids = filename.split('.')[0]
                with open(os.path.join(directory, filename), 'r') as file:
                    data = file.read()
                    load_time_pattern = re.findall(r"load_duration\(insert \+ optimize\) = ([\d.]+)", data)
                    query_time_pattern = re.findall(r"cost=([\d.]+)s", data)

                    milvus_indexing[f"{n_centroids}"].extend([float(q) for q in load_time_pattern])
                    milvus_querying[f"{n_centroids}_{query_functions}"].extend([float(q) for q in query_time_pattern])

    return milvus_indexing, milvus_querying

def parse_blocks_results(dataset, version):
    blocks_indexing = defaultdict(list)
    blocks_querying = defaultdict(list)
    
    directory_index = f"../results/{dataset}/blocks/indexing/"
    directory_query = f"../results/{dataset}/blocks/querying/"
    
    if os.path.exists(directory_index):
        for filename in os.listdir(directory_index):
            n_centroids = filename.split('_')[-3]
            with open(os.path.join(directory_index, filename), 'r') as file:
                data = json.load(file)
                blocks_indexing[n_centroids].append(data["generate_index_blocks_mean"])
    
    if os.path.exists(directory_query):
        for filename in os.listdir(directory_query):
            n_centroids = filename.split('_')[-3] + "_" + filename.split('_')[-2]
            with open(os.path.join(directory_query, filename), 'r') as file:
                data = json.load(file)
                if version == 1:
                    blocks_querying[n_centroids].append(data['total_querying_times_mean'])
                elif version == 2:
                    blocks_querying[n_centroids].append(
                        data['map_iterdata_times'][0] + 
                        statistics.mean((function[0] for function in data["map_queries_times"][0])) + 
                        data["reduce_iterdata_times"][0] + 
                        data["reduce_queries_times"][0][0]
                        )
                elif version == 3:
                    blocks_querying[n_centroids].append(
                        data['map_iterdata_times'][0] + 
                        statistics.mean((function[1] + sum(function[4]) + function[5] for function in data["map_queries_times"][0])) + 
                        data["reduce_iterdata_times"][0] + 
                        data["reduce_queries_times"][0][0]
                        )
                elif version == 4:
                    total = data['total_querying_times_mean']

                    blocks_querying[n_centroids].append(
                        total -
                        statistics.mean((sum(function[2]) + sum(function[3]) for function in data["map_queries_times"][0]))
                        )
    
    return blocks_indexing, blocks_querying

def plot_comparison(milvus, blocks, title, ylabel, dest, phase, version):
    configs = sorted(set(milvus.keys()) & set(blocks.keys()), key=lambda x: tuple(map(int, x.split("_"))))
    
    milvus_means = [statistics.mean(milvus[c]) for c in configs]
    blocks_means = [statistics.mean(blocks[c]) for c in configs]
    
    x = range(len(configs))
    width = 0.4
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, milvus_means, width, label='Milvus', color='blue')
    plt.bar([i + width for i in x], blocks_means, width, label='Blocks', color='orange')
    
    plt.xlabel("Configurations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([i + width / 2 for i in x], configs, rotation=45)
    plt.legend()
    plt.tight_layout()
    if dest:
        plt.savefig(f"{dest}_milvus_vs_blocks_{phase}_version_{version}.png")
    else:
        plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Compare Milvus and Blocks implementations.")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets")
    parser.add_argument("--version", required=True, help="1. Blocks overall time\n2 Discard lambda invokation and results fetching\n3 Version 2 + discard index download and load times\n4. Discard only index downloading and loading")
    parser.add_argument("--plot_export_dest", required=False, help="Name of the destination folder for the plots. If not provided, plots will be shown, but not saved.")
    args = parser.parse_args()
    
    for dataset in args.datasets:
        milvus_indexing, milvus_querying = parse_milvus_results(dataset)
        blocks_indexing, blocks_querying = parse_blocks_results(dataset, int(args.version))
        plot_comparison(milvus_indexing, blocks_indexing, f"Indexing {dataset}", "Avg Indexing Time (s)", args.plot_export_dest, "indexing", args.version)
        plot_comparison(milvus_querying, blocks_querying, f"Querying {dataset}", "Avg Querying Time (s)", args.plot_export_dest, "querying", args.version)

if __name__ == "__main__":
    main()
