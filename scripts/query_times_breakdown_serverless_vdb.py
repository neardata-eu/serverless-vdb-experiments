import json
import os
import statistics
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser(description="Query time break down over all stages")
parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets (deep100M, deep, gist, sift)")
args = parser.parse_args()

datasets = args.datasets
impl = "blocks"

for dataset in datasets:
    print("=" * 50)
    print(f"📊 Results for dataset: {dataset}")
    print("=" * 50)
    ## QUERYING ##
    directory = f"../results/{dataset}/{impl}/querying/"
    time_sum = defaultdict(list)
    shuffle = defaultdict(list)
    map_iterdata = defaultdict(list)
    #######################################
    total_function = defaultdict(list)
    download_queries = defaultdict(list)
    download_index = defaultdict(list)
    load_index = defaultdict(list)
    query_time = defaultdict(list)
    reduce_time = defaultdict(list)
    #######################################
    reduce_iterdata = defaultdict(list)
    final_reduce = defaultdict(list)

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            n_centroids = filename.split('_')[-3]+"_"+filename.split('_')[-2]
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                time_sum[n_centroids].append(data['total_querying_times_mean'])
                map_iterdata[n_centroids].append(data['map_iterdata_times'][0])
                #############################################################
                for function in data["map_queries_times"][0]:    
                    total_function[n_centroids].append(function[0])
                    download_queries[n_centroids].append(function[1])
                    download_index[n_centroids].append(sum(function[2]))
                    load_index[n_centroids].append(sum(function[3]))
                    query_time[n_centroids].append(sum(function[4]))
                    reduce_time[n_centroids].append(function[5])
                #############################################################
                reduce_iterdata[n_centroids].append(data["reduce_iterdata_times"][0])
                final_reduce[n_centroids].append(data["reduce_queries_times"][0][0])

    if time_sum:  # Check if both are not empty
        average_values = {
            x: (
                statistics.mean(time_sum[x]),
                statistics.stdev(time_sum[x]) if len(time_sum[x]) > 1 else 0.0,
                ##########
                statistics.mean(map_iterdata[x]),
                statistics.stdev(map_iterdata[x]) if len(map_iterdata[x]) > 1 else 0.0,
                ##########
                #############################################################
                ##########
                statistics.mean(total_function[x]),
                statistics.stdev(total_function[x]) if len(total_function[x]) > 1 else 0.0,
                ##########
                statistics.mean(download_queries[x]),
                statistics.stdev(download_queries[x]) if len(download_queries[x]) > 1 else 0.0,
                ##########
                statistics.mean(download_index[x]),
                statistics.stdev(download_index[x]) if len(download_index[x]) > 1 else 0.0,
                ##########
                statistics.mean(load_index[x]),
                statistics.stdev(load_index[x]) if len(load_index[x]) > 1 else 0.0,
                ##########
                statistics.mean(query_time[x]),
                statistics.stdev(query_time[x]) if len(query_time[x]) > 1 else 0.0,
                ##########
                statistics.mean(reduce_time[x]),
                statistics.stdev(reduce_time[x]) if len(reduce_time[x]) > 1 else 0.0,
                ##########
                #############################################################
                ##########
                statistics.mean(reduce_iterdata[x]),
                statistics.stdev(reduce_iterdata[x]) if len(reduce_iterdata[x]) > 1 else 0.0,
                ##########
                ##########
                statistics.mean(final_reduce[x]),
                statistics.stdev(final_reduce[x]) if len(final_reduce[x]) > 1 else 0.0,
                ##########
            )
            for x in time_sum
        }

        print("\n📌 QUERYING RESULTS (Avg time per function):")
        print(f"{'Centroids':<11}{'All Querying':>8}{'±':>2}{'MapIterdata':>22}{'±':>2}{'Function':>18}{'±':>6}{'ReduceIterdata':>22}{'±':>2}{'FinalReduce':>22}{'±':>2}")              
        print("-" * 180)

        for centroids, (avg_time, std_dev, mi, std_dev_mi, tf, std_dev_tf, dq, std_dev_dq, di, std_dev_di, li, std_dev_li, qt, std_dev_qt, rt, std_dev_rt, ri, std_dev_ri, fr, std_dev_fr) in sorted(average_values.items(), key=lambda x: int(x[0])):
            print(f"{centroids:<6}{avg_time:>12.4f}{std_dev:>12.4f}{mi:>12.4f}{std_dev_mi:>12.4f}{tf:>12.4f}{std_dev_tf:>12.4f}{ri:>12.4f}{std_dev_ri:>12.4f}{fr:>12.4f}{std_dev_fr:>12.4f}")


        print("\n📌 QUERYING RESULTS (Avg time per function):")
        print(f"{'Centroids':<11}{'Function':>6}{'±':>6}{'DownloadQuery':>22}{'±':>2}{'Download Index':>22}{'±':>2}{'Load Index':>20}{'±':>4}{'Query Time':>20}{'±':>4}{'Reduce Time':>20}{'±':>4}")              
        print("-" * 180)

        for centroids, (avg_time, std_dev, mi, std_dev_mi, tf, std_dev_tf, dq, std_dev_dq, di, std_dev_di, li, std_dev_li, qt, std_dev_qt, rt, std_dev_rt, ri, std_dev_ri, fr, std_dev_fr) in sorted(average_values.items(), key=lambda x: int(x[0])):
            print(f"{centroids:<6}{tf:>12.4f}{std_dev_tf:>12.4f}{dq:>12.4f}{std_dev_dq:>12.4f}{di:>12.4f}{std_dev_di:>12.4f}{li:>12.4f}{std_dev_li:>12.4f}{qt:>12.4f}{std_dev_qt:>12.4f}{rt:>12.4f}{std_dev_rt:>12.4f}")

    else:
        print("\n⚠️ No querying results available.")

    print("\n" + "=" * 180 + "\n")
