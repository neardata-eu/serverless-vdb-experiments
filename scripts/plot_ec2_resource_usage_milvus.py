import docker
import time
import csv
import matplotlib.pyplot as plt
import json

# Configuration
CONTAINER_NAME = "potato"  # Replace with your container name
LOG_FILE = "gist_8qf.csv"
DURATION = 300  # Total monitoring duration in seconds

# Initialize Docker client
client = docker.from_env()

def log_stats():
    """Streams real-time CPU and memory usage using Docker SDK and logs it to a file."""
    container = client.containers.get(CONTAINER_NAME)

    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU_Usage(%)", "Memory_Usage(MB)"])

        start_time = time.time()

        for stats in container.stats(stream=True):
            if time.time() - start_time > DURATION:
                break  # Stop after DURATION seconds

            try:
                # CPU Usage Calculation
                stats = json.loads(stats)
                cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
                cpu_usage = (cpu_delta / system_delta) * stats["cpu_stats"]["online_cpus"] * 100 if system_delta > 0 else 0

                # Memory Usage
                mem_usage = stats["memory_stats"]["usage"] / (1024 * 1024)  # Convert to MB

                timestamp = time.time()
                print(f"CPU: {cpu_usage:.2f}%, Memory: {mem_usage:.2f}MB")
                writer.writerow([timestamp, cpu_usage, mem_usage])

            except Exception as e:
                print(f"Error processing stats: {e}")
                continue

def plot_data():
    """Reads logged data and generates a dual-axis plot."""
    timestamps, cpu_usages, mem_usages = [], [], []

    with open(LOG_FILE, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            timestamps.append(float(row[0]))
            cpu_usages.append(float(row[1]))
            mem_usages.append(float(row[2]))

    # Convert timestamps to relative time
    start_time = timestamps[0]
    timestamps = [t - start_time for t in timestamps]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # First Y-axis (CPU Usage)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("CPU Usage (%)", color="tab:blue")
    ax1.plot(timestamps, cpu_usages, linestyle="-", linewidth=2, color="tab:blue", label="CPU Usage")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Second Y-axis (Memory Usage)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory Usage (MB)", color="tab:red")
    ax2.plot(timestamps, mem_usages, linestyle="-", linewidth=2, color="tab:red", label="Memory Usage")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Title and grid
    plt.title(f"CPU & Memory Usage for {CONTAINER_NAME}")
    fig.tight_layout()
    plt.grid(True)
    plt.show()



# Run monitoring and plotting
#log_stats()
plot_data()
