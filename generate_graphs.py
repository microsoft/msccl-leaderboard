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
    'font.size': '10',
    'font.family':'sans-serif',
    'font.sans-serif':['Open Sans']
}
secondary_color = '#CED0D1'


#############
# Arguments #
#############

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate graphs for the MSCCL leaderboard page')
parser.add_argument('--prod', action='store_true')
args = parser.parse_args()
timestamp = f'_{int(time.time())}' if args.prod else ''
filetype = 'svg' if args.prod else 'png'


#######################
# Load and parse data #
#######################


def parse_nccl_tests_log(path, inplace):
    """ Given a path to a nccl-tests log file load it, parse the timing measurements and output the size and time
    columns as numpy arrays. """
    sizes = []
    times = []
    # Example header and data line:
    #                                                     out-of-place                       in-place          
    #       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error
    #        1024           256   float     sum   4019.3    0.00    0.00  4e+00    49.33    0.02    0.04  5e-07
    pattern = re.compile(f'(\[1,0\]<stdout>:)?\s*(\d+)(?:\s+[^\s]+){{{7 if inplace else 2}}}\s+([\d\.]+).*')
    with open(path) as f:
        for line in f.readlines():
            m = pattern.match(line)
            if m is not None:
                sizes.append(int(m.group(2)))
                times.append(float(m.group(3)))
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


def is_inplace(collective):
    _, place = collective.split('-')
    if place == 'inplace':
        return True
    elif place == 'outofplace':
        return False
    else:
        raise ValueError(f'Unknown place {place}')


def load_data(data_logs):
    """ Load all the data logs into a dictionary keyed by (configuration, collective, name) and return it. """
    data = {}
    for key, path in data_logs.items():
        config, collective, name = key
        data[key] = parse_nccl_tests_log(path, is_inplace(collective))
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
    plt.axhline(y=1, color=secondary_color, linestyle='--', linewidth=0.75)
    ax.plot(sizes, speedup, color='red')
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format_size(x)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(secondary_color)
    ax.spines['left'].set_color(secondary_color)
    ax.tick_params(axis='x', colors=secondary_color)
    ax.tick_params(axis='y', colors=secondary_color)
    plt.setp(ax.get_xticklabels(), color='black')
    plt.setp(ax.get_yticklabels(), color='black')
    plt.minorticks_off()
    ax.tick_params(direction='in')


def thumbnail_path(config, collective):
    return f'graphs/{config}_{collective}_thumbnail{timestamp}.{filetype}'


def plot_thumbnail(sizes, speedup):
    # Plot the speedup
    fig, ax = plt.subplots(
        figsize=(graph_width, graph_width / graph_aspect_ratio))
    plot_common(ax, sizes, speedup)
    path = thumbnail_path(config, collective)
    print(f'Writing {os.path.abspath(path)}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
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


def format_collective(collective):
    return collective.split('-')[0]


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
            'Configuration'] + [format_collective(x) for x in collectives], tablefmt='github'))
