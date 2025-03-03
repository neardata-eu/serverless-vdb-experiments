import json
import os
import statistics
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser(description="Calculate averages and standard deviation of results.")
parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets (deep100k, deep1M, deep, deep100M, gist, sift)")
parser.add_argument("--impl", required=True, help="Implementation type (blocks / centroids)")
args = parser.parse_args()

datasets = args.datasets
impl = args.impl

for dataset in datasets:
    print("=" * 50)
    print(f"📊 Results for dataset: {dataset}")
    print("=" * 50)
    nf_dist = 0
    ## INDEXING ##
    directory = f"../results/{dataset}/{impl}/indexing/"
    time_sum = defaultdict(list)
    generate_index_avg_sum = defaultdict(list)
    distribute_vectors = defaultdict(list)
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            n_centroids = filename.split('_')[-3]
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                if impl == "centroids":
                    nf_dist = len(data["distribute_vectors_centroids"])
                time_sum[n_centroids].append(data["generate_index_blocks_mean" if impl == "blocks" else "total_indexing_centroids"]) #use global_index_centroids to print the global kmeans time on centroids implementation
                if impl == "centroids": distribute_vectors[n_centroids].append(statistics.mean(data["distribute_vectors_centroids"]))
                if impl == "centroids": generate_index_avg_sum[n_centroids].append(statistics.mean(data["generate_index_centroids"]))
                

    if time_sum:  # Check if dictionary is not empty
        average_values = {
            x: (
                statistics.mean(time_sum[x]),
                statistics.stdev(time_sum[x]) if len(time_sum[x]) > 1 else 0.0  # Error bar
            )
            for x in time_sum
        }

        print("\n📌 INDEXING RESULTS:")
        print(f"{'Centroids':<12}{'Avg Time (s)':>15}{'± Std Dev':>15}")
        print("-" * 45)

        for centroids, (avg, std_dev) in sorted(average_values.items(), key=lambda x: int(x[0])):
            print(f"{centroids:<12}{avg:>15.4f}{std_dev:>15.4f}")

    else:
        print("\n⚠️ No indexing results available.")

    if impl == "centroids" and distribute_vectors and generate_index_avg_sum:
        average_values = {
            x: (
                statistics.mean(distribute_vectors[x]),
                statistics.stdev(distribute_vectors[x]) if len(distribute_vectors[x]) > 1 else 0.0  # Error bar
            )
            for x in distribute_vectors
        }

        print(f"\n\t📌 Distribute vectors ({nf_dist} functions):")
        print(f"\t{'Centroids':<12}{'Avg Time (s)':>15}{'± Std Dev':>15}")
        print("\t" + "-" * 45)

        for centroids, (avg, std_dev) in sorted(average_values.items(), key=lambda x: int(x[0])):
            print(f"\t{centroids:<12}{avg:>15.4f}{std_dev:>15.4f}")

        average_values = {
            x: (
                statistics.mean(generate_index_avg_sum[x]),
                statistics.stdev(generate_index_avg_sum[x]) if len(generate_index_avg_sum[x]) > 1 else 0.0  # Error bar
            )
            for x in generate_index_avg_sum
        }

        print("\n\t📌 Generate centroids index:")
        print(f"\t{'Centroids':<12}{'Avg Time (s)':>15}{'± Std Dev':>15}")
        print("\t" + "-" * 45)

        for centroids, (avg, std_dev) in sorted(average_values.items(), key=lambda x: int(x[0])):
            print(f"\t{centroids:<12}{avg:>15.4f}{std_dev:>15.4f}")

    ## QUERYING ##
    directory = f"../results/{dataset}/{impl}/querying/"
    time_sum = defaultdict(list)
    recalls_sum = defaultdict(list)

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            n_centroids = filename.split('_')[-3]+"_"+filename.split('_')[-2]
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                time_sum[n_centroids].append(data['total_querying_times_mean'])
                recalls_sum[n_centroids].append(data['recalls_mean'])

    if time_sum and recalls_sum:  # Check if both are not empty
        average_values = {
            x: (
                statistics.mean(time_sum[x]),
                statistics.stdev(time_sum[x]) if len(time_sum[x]) > 1 else 0.0,  # Error bar
                statistics.mean(recalls_sum[x])
            )
            for x in time_sum
        }

        print("\n📌 QUERYING RESULTS:")
        print(f"{'Centroids':<12}{'Avg Time (s)':>15}{'± Std Dev':>15}{'Recall':>15}")
        print("-" * 60)

        for centroids, (avg_time, std_dev, avg_recall) in sorted(average_values.items(), key=lambda x: int(x[0])):
            print(f"{centroids:<12}{avg_time:>15.4f}{std_dev:>15.4f}{avg_recall:>15.4f}")

    else:
        print("\n⚠️ No querying results available.")

    print("\n" + "=" * 50 + "\n")
