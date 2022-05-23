# Copyright (c) Microsoft Corporation.

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import humanfriendly
import tabulate
import argparse


#################
# Configuration #
#################

main_name = 'msccl'
baseline_name = 'nccl'
graph_width = 4
graph_aspect_ratio = 4/3
matplotlib_rc_params = {
    'font.size': '16'
}


#############
# Arguments #
#############

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate graphs for the MSCCL leaderboard page')
# Argument to specify if a unique identifier should be added to all output file names
parser.add_argument('--timestamp', action='store_true',
                    help='Add a unique identifier to all graph file names')
args = parser.parse_args()
timestamp = f'_{int(time.time())}' if args.timestamp else ''


#######################
# Load and parse data #
#######################


def parse_nccl_tests_log(path):
    """ Given a path to a nccl-tests log file load it, parse the timing measurements and output the size and time
    columns as numpy arrays. """
    sizes = []
    times = []
    # Example header and data line:
    #       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error
    #        1024           256   float     sum   4019.3    0.00    0.00  4e+00    49.33    0.02    0.04  5e-07
    pattern = re.compile('\s*(\d+)(?:\s+[^\s]+){3}\s+([\d\.]+).*')
    with open(path) as f:
        for line in f.readlines():
            m = pattern.match(line)
            if m is not None:
                sizes.append(int(m.group(1)))
                times.append(float(m.group(2)))
    return np.array(sizes), np.array(times)


def find_data_logs():
    """ Find all the nccl-test log files under the data/ subdirectory and return a dictionary of their paths
    keyed by (configuration,collective,name) as given by the path data/configuration/collective/name.txt. """
    data_logs = {}
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                config, collective, file_name = path.split('/')[1:]
                name = os.path.splitext(file_name)[0]
                data_logs[(config, collective, name)] = path
    return data_logs


def load_data(data_logs):
    """ Load all the data logs into a dictionary keyed by (configuration, collective, name) and return it. """
    data = {}
    for key, path in data_logs.items():
        data[key] = parse_nccl_tests_log(path)
    return data


# Load all the data logs
data = load_data(find_data_logs())

# Collect list of all the configurations
configs = sorted(set(key[0] for key in data))
collectives = sorted(set(key[1] for key in data))

# For each config and collective, calculate the speedup of the main results over the baseline results
speedups = {}
for config in configs:
    for collective in collectives:
        # Skip if either the main or baseline results are missing
        if (config, collective, main_name) not in data or (config, collective, baseline_name) not in data:
            continue
        # Find the main and baseline results
        main_sizes, main_times = data[(config, collective, main_name)]
        baseline_sizes, baseline_times = data[(
            config, collective, baseline_name)]
        # Check that the sizes in the baseline and main results match
        if any(main_sizes != baseline_sizes):
            # Find index of the first mismatch
            i = 0
            while i < len(main_sizes) and i < len(baseline_sizes) and main_sizes[i] == baseline_sizes[i]:
                i += 1
            # Print a warning
            print(f'WARNING: {baseline_name} and {main_name} results for'
                  f'{config}/{collective} have different sizes at index {i}')
            continue
        # Calculate the speedup
        speedup = baseline_times / main_times
        # Store the speedup
        speedups[(config, collective)] = main_sizes, speedup


#################
# Render graphs #
#################

def format_size(size):
    return humanfriendly.format_size(int(size)).replace('bytes', 'B')


def plot_common(ax, sizes, speedup):
    plt.axhline(y=1, color='black', linestyle='--', linewidth=0.75)
    ax.plot(sizes, speedup, color='red')
    ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format_size(x)))


def thumbnail_path(config, collective):
    return f'graphs/{config}_{collective}_thumbnail{timestamp}.png'


def plot_thumbnail(sizes, speedup):
    # Plot the speedup
    fig, ax = plt.subplots(
        figsize=(graph_width, graph_width / graph_aspect_ratio))
    plot_common(ax, sizes, speedup)
    path = thumbnail_path(config, collective)
    print(f'Writing {os.path.abspath(path)}')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


matplotlib.rcParams.update(matplotlib_rc_params)

# Create a separate figure for each combination of config and collective
for config, collective in speedups:
    sizes, speedup = speedups[(config, collective)]
    plot_thumbnail(sizes, speedup)


################
# Render pages #
################

def thumbnail_embed(config, collective):
    return f'![Speedup for {collective} on {config}]({thumbnail_path(config, collective)})'


# Gather table of thumbnail paths
thumbnails = []
for config in configs:
    row = [config]
    for collective in collectives:
        if (config, collective) in speedups:
            row.append(thumbnail_embed(config, collective))
        else:
            row.append('')
    thumbnails.append(row)

table_path = 'speedups_table.md'
with open(table_path, 'w') as f:
    print(f'Writing {os.path.abspath(table_path)}')
    f.write(tabulate.tabulate(thumbnails, headers=[
            'Configuration'] + collectives, tablefmt='github'))
