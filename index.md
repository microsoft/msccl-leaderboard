---
title: MSCCL Leaderboard
---

[Microsoft Collective Communication Library (MSCCL)](https://github.com/microsoft/msccl) is a platform to execute custom
collective communication algorithms for multiple accelerators supported by Microsoft Azure. MSCCL enables hardware and
application specific optimizations that can deliver huge speedups over unspecialized communication algorithms.

The table below shows speedups given by switching from NVIDIA's NCCL to MSCCL. To get these speedups in your own Microsoft Azure workload
follow the instructions in the [msccl-tools](https://github.com/microsoft/msccl-tools#readme) and [msccl](https://github.com/microsoft/msccl#readme) repositories.

{% include_relative speedups_table.md %}

The graphs in the table above show the speedup on the Y axis for a range of user data sizes on the X axis. Each graph shows the
speedup for a specific hardware configuration and collective operation. For example, the graph in the "1xNDv4" row and
"Allreduce" column shows the speedups given by MSCCL for the [Allreduce
collective](https://en.wikipedia.org/wiki/Collective_operation#All-Reduce_[5]) when running on a single [Azure NDv4 VM containing 8
NVIDIA A100 GPUs](https://docs.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series).

# Methods

These speedups were produced by running the relevant benchmarks from [nccl-tests](https://github.com/NVIDIA/nccl-tests)
on the target hardware configuration for MSCCL, with the algorithms available in [msccl-tools](https://github.com/microsoft/msccl-tools), and NCCL {{ site.content.baseline-nccl-version }}.