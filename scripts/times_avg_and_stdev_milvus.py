import os
import re
import argparse
from collections import defaultdict
import statistics

parser = argparse.ArgumentParser(description="Calculate averages and standard deviation of results.")
parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets (deep, gist, sift)")
args = parser.parse_args()

datasets = args.datasets


for dataset in datasets:
    indexing = defaultdict(list)
    querying = defaultdict(list)
    recall = defaultdict(list)
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
                    avg_recall_pattern = re.findall(r"avg_recall=([\d.]+)", data)

                    indexing[f"{n_centroids}_{query_functions}"] = [float(q) for q in load_time_pattern]
                    querying[f"{n_centroids}_{query_functions}"] = [float(q) for q in query_time_pattern]
                    recall[f"{n_centroids}_{query_functions}"] = [float(r)*(100 if dataset == "deep" else 1000) for r in avg_recall_pattern]

    indexing = dict(sorted(indexing.items(), key=lambda x: tuple(map(int, x[0].split("_")))))
    querying = dict(sorted(querying.items(), key=lambda x: tuple(map(int, x[0].split("_")))))

    print("=" * 50)
    print(f"📊 Results for dataset: {dataset}")
    print("=" * 50)
    print("\n📌 INDEXING RESULTS:")
    print(f"{'Centroids':<12}{'Avg Time (s)':>15}{'± Std Dev':>15}")
    print("-" * 45)
    for config in indexing:
        avg_index_time = statistics.mean(indexing[config])
        std_dev_index_time = statistics.stdev(indexing[config]) if len(indexing[config]) > 1 else 0.0
        print(f"{config:<12}{avg_index_time:>15.4f}{std_dev_index_time:>15.4f}")

    print("\n📌 QUERYING RESULTS:")
    print(f"{'Centroids':<12}{'Avg Time (s)':>15}{'± Std Dev':>15}{'Recall':>15}")
    print("-" * 60)
    for config in querying:
        avg_query_time = statistics.mean(querying[config])
        std_dev_query_time = statistics.stdev(querying[config]) if len(querying[config]) > 1 else 0.0
        avg_recall = statistics.mean(recall[config])
        print(f"{config:<12}{avg_query_time:>15.4f}{std_dev_query_time:>15.4f}{avg_recall:>15.4f}")

    print("\n" + "=" * 50 + "\n")

