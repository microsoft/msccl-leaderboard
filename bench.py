import os
import sys
import re
import datetime
import fnmatch
import argparse


# This python program is runs NCCL benchmarks tests using NCCL and MSCCL
# It takes 4 command line arguments: --mode, --directory, --filter, --outputDirectory, --endBufSize, --nGPU
#  mode: is either "test" or "run" where "test" indicates that the program should run in test mode and
#       "run" indicates that the program should actually run the benchmarks.
#  directory: is the directory where the MSCCL xml files are located
#  filter: is a filter in the style of https://docs.python.org/3/library/fnmatch.html for files in the directory that will be tested
#  outputDirectory: is the directory where the output files will be placed
#     The file "results.txt" in the output directory will have results appended to it. It will be created if it doesn't exist
#  endBufSize: is the end buffer size to use for the test. It is optional and defaults to 32MB.
#  nGPU: is the number of GPUs to use for the test. It is optional and defaults to 16.
#
def main():
    # Setup an argument parser to parse the command line arguments
    parser = argparse.ArgumentParser(description='Run NCCL benchmarks tests using NCCL and MSCCL')
    parser.add_argument('-m', '--mode', type=str, help='mode: is either "test" or "run" where "test" indicates that the program should run in test mode and "run" indicates that the program should actually run the benchmarks.', default="run", choices=["test", "run"])
    parser.add_argument('-d', '--directory', type=str, help='directory: is the directory where the MSCCL xml files are located', default="sccl-presynth/sccl_presynth")
    parser.add_argument('-f', '--filter', type=str, help='filter: is a filter in the style of https://docs.python.org/3/library/fnmatch.html for files in the directory that will be tested', required=True)
    parser.add_argument('-o', '--outputDirectory', type=str, help='outputDirectory: is the directory where the output files will be placed', required=True)
    parser.add_argument('-e', '--endBufSize', type=str, help='endBufSize: is the end buffer size to use for the test. It is optional and defaults to 32MB.', default="32MB")
    parser.add_argument('-n', '--nGPU', type=int, help='nGPU: is the number of GPUs to use for the test. It is optional and defaults to 16.', default=16)

    # Parse the command line arguments
    args = parser.parse_args()

    # Make sure sufficient command line arguments are provided
    # and make sure the mode is either "test" or "run"
    # if len(sys.argv) != 5 or (sys.argv[1] != "test" and sys.argv[1] != "run") :
    #    print("Please provide line arguments: mode (test | run), directory, filter, outputDirectory")
    #    return

    mode = args.mode
    directory = args.directory
    filter = args.filter
    outputDirectory = args.outputDirectory
    endBufSize = args.endBufSize
    nGPU = args.nGPU

    # Print all of the command line arguments
    print(f"Arguments: {args}")

    # This is the CSV file with the final results
    resultFileName = "results.txt"

    # Make sure the outputDirectory exists and if it doesn't create it.
    if not os.path.exists(outputDirectory):
        print("Creating output directory: " + outputDirectory)
        os.makedirs(outputDirectory)

    # Make sure a file named resultFile exists in outputDirectory and if it doesn't create it.
    # Make sure the file path works for both linux and Windows
    resultFile = os.path.join(outputDirectory, resultFileName);
    if not os.path.exists(resultFile):
        print("Creating result file: " + resultFile)
        # Put header row into the result file
        with open(resultFile, "w") as f:
            f.writelines(f'Algorithm,Filter,TimeOfTest,Library,NcclVersion,Size,Time_OutOfPlace,Time_InPlace,MSCCL_File\n')

    prossessedFiles = []

    # get current date and time as a string
    timeOfTest = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the result file for appending
    resultsF = open(resultFile, "a")

    # For each file filename in directory that matches the filter in the name do the following:
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, filter):
            # if the file name doesn't contain the number of GPUs specified in the command line arguments then skip it
            if str(nGPU) not in filename:
                print(f"Skipping file {filename} because it doesn't contain the number of GPUs specified in the command line arguments")
                continue

            # Track the names of all files processed from the directory
            prossessedFiles.append(filename)

            outputFileList = runBenchmark(mode, filename, endBufSize, directory, outputDirectory)

            algorithm = filename.split(".")[0]

            # For each file in outputFileList parse the file to find the results
            # And append them to the resultFile in CSV format
            for outputFile in outputFileList:
                # Extract the NCCL version from the output file based on a line like [1,0]<stdout>:NCCL version 2.12.12.MSCCL.0.7.3+cuda11.6
                with open(outputFile, "r") as f:
                    for line in f:
                        if "NCCL version" in line:
                            ncclVersion = line.split("NCCL version ")[1].strip()
                            break

                # Parse the output file to get each line of performance results
                data = parse_test_log(outputFile)
                for line in data:
                    library = "MSCCL" if "msccl" in outputFile else "NCCL"
                    resultsF.write(f'{algorithm},{filter},{timeOfTest},{library},{ncclVersion},{line},{filename}\n')

    resultsF.close()

    # report the names of all files processed from the directory
    if (len(prossessedFiles) == 0):
        print("No files were processed. Consider changing the filter or nGPU command line arguments.")
    else:
        print(f"Processed Files = {prossessedFiles}")


# Execute the test program on the file filename, place results in the output folder
# then returns a list of the names of the output files
def runBenchmark(mode, filename, endBufSize, inputDirectory, outputDirectory):
    print(f"\nRun Benchmark in mode {mode} for file: " + filename)

    outputFileList = []

    # Extract the substring from the name of the file up until the first "." in the file name. Call this substring "algorithm"
    mscclAlgorithm = filename.split(".")[0]

    # Construct the string name of the nccl algorithm from the msccl algorithm.
    # Nccl uses "_" to separate words in the algorithm name and msccl names are more insconsistent.
    # For example, msccl uses "all_reduce" and nccl uses "Allreduce".
    # This code has explicit conversions and will raise an exception if it doesn't know how to convert.
    # This code uses the python 3.10 match statement if the python version is 3.10 or greater.
    ncclAlgorithm = ""
    #    match mscclAlgorithm.lower():
    #        case "allreduce":
    #            ncclAlgorithm = "all_reduce"
    #        case "allgather":
    #            ncclAlgorithm = "all_gather"
    #        case "alltoall":
    #            ncclAlgorithm = "all_to_all"
    #        case _:
    #            raise Exception("Unknown algorithm: " + mscclAlgorithm)
    if mscclAlgorithm.lower() == "allreduce":
        ncclAlgorithm = "all_reduce"
    elif mscclAlgorithm.lower() == "allgather":
        ncclAlgorithm = "all_gather"
    elif mscclAlgorithm.lower() == "alltoall":
        ncclAlgorithm = "all_to_all"
    else:
        raise Exception("Unknown algorithm: " + mscclAlgorithm)

    ncclPerfTest = ncclAlgorithm + "_perf"

    # Construct the command to run the benchmarks using the mpirun command
    # example command:
    #    mpirun --bind-to numa --tag-output --allow-run-as-root -np 8  -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x PATH -x LD_LIBRARY_PATH=~/msccl/build/lib/:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x NCCL_NET_GDR_LEVEL=5 -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x MSCCL_XML_FILES=/home/saemal/test.xml ~/nccl-tests/build/all_reduce_perf -b 1KB -e 32MB -f 2 -g 1 -c 1 -w 100 -n 100
    #
    # below are important components of the command
    #
    commandPrefix = "mpirun --bind-to numa --tag-output --allow-run-as-root"
    numProcs = " -np 16"  # for the -np parameter
    mcaParams = " -hostfile /job/hostfile -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0"

    # The environment variables that are set for the command - consider separating these out variables that are non-constant
    envVars = " -x PATH -x LD_LIBRARY_PATH=/home/saemal/msccl/build/lib/:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x NCCL_NET_GDR_LEVEL=5 -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

    mscclEnvVars = " -x MSCCL_XML_FILES=" + os.path.join(inputDirectory, filename)
    ncclBufferRange = f" -b 1KB -e {endBufSize}"
    ncclParameters = " /home/saemal/msccl-tools/msccl/autosynth/msccl_ndv2_launcher.sh /home/saemal/nccl-tests/build/" + ncclPerfTest + ncclBufferRange + " -f 2 -g 1 -c 1 -w 100 -n 100"

    # Run the NCCL version of the benchmark (without MSCCL) and direct the output to a new file called filename_nccl_result
    # Delete output file if it already exists
    ncclOutputFile = os.path.join(outputDirectory, f"{filename}_nccl_result.txt")
    if os.path.exists(ncclOutputFile):
        os.remove(ncclOutputFile)

    ncclCommandLine = commandPrefix + numProcs + mcaParams + envVars + ncclParameters + " > " + ncclOutputFile
    print(f"Running NCCL-only test: {ncclCommandLine}")
    if mode == "test":
        writeToNcclFile = open(ncclOutputFile, "a")
        writeToNcclFile.write(exampleOutput)
        writeToNcclFile.close()
    else:
        assert mode == "run"
        os.system(f"{ncclCommandLine}")


    # Run the MSCCL version of the benchmark and direct the output to a new file called filename_msccl_result
    # Delete the output file if it already exists
    mscclOutputFile = os.path.join(outputDirectory, f"{filename}_msccl_result.txt")
    mscclCommandLine = commandPrefix + numProcs + mcaParams + envVars + mscclEnvVars + ncclParameters + " > " + mscclOutputFile
    print(f"\nRunning MSCCL test: {mscclCommandLine}")
    if os.path.exists(mscclOutputFile):
        os.remove(mscclOutputFile)

    if mode == "test":
        writeToFile = open(mscclOutputFile, "a")
        writeToFile.write(f"File {filename} Parsed MSCCL (write this to test validation that the file contains the string Parsed MSSCL)")
        writeToFile.write(exampleOutput)
        writeToFile.close()
    else:
        assert mode == "run"
        os.system(f"{mscclCommandLine}")

    # Search both output files for the strings "WARN" or "ERROR" and
    # if either are found print a message that the program failed and provide the name of the output file that contained the string.
    for resultFile in [ncclOutputFile, mscclOutputFile]:
        with open(resultFile) as f:
            contents = f.read()
            if "WARN" in contents.upper() or "ERROR" in contents.upper():
                print(f"The benchmark failed. The output file {resultFile} contains WARN or ERROR.")
            else:
                if resultFile == mscclOutputFile and "Parsed MSCCL" not in contents:
                    print(f"The benchmark failed for {filename} because \"Parsed MSCCL\" not found in output.")
                else:
                    # Successful benchmark run
                    print(f"Successful benchmark run for {filename} and output file is {resultFile}")
                    outputFileList.append(resultFile)

    # Return the list of successful benchmark results
    return outputFileList

# Parse the output file
# For each matching line in the file return size and time for out-of-place and in-place
# Code is based on based on https://github.com/microsoft/msccl-leaderboard/blob/main/generate_graphs.py#L48
def parse_test_log(path):
    # Example input
    # [1,0]<stdout>:#
    # [1,0]<stdout>:#                                                              out-of-place                       in-place
    # [1,0]<stdout>:#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    # [1,0]<stdout>:#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    # [1,0]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18075:18075 [0] NCCL INFO Launch mode Parallel
    # [1,0]<stdout>:        1024           256     float     sum      -1    75.18    0.01    0.03      0    78.30    0.01    0.02      0
    # [1,0]<stdout>:        2048           512     float     sum      -1    79.54    0.03    0.05      0    82.56    0.02    0.05      0

    # output is an array of string
    output = []

    # Create a pattern for lines that begin with '[1,0]<stdout>: ' and contain a number followed by a space and then a number
    pattern = re.compile(f'(\[1,0\]<stdout>:)?\s*(\d+)\s+(\d+)\s+')
    with open(path) as f:
        for line in f.readlines():
            m = pattern.match(line)
            if m is not None:
                # split the line into an array of strings were each string is separated by one or more spaces
                lineArray = line.split()
                # Create a common separated string with the size, first time columns
                output.append(f"{lineArray[1]},{lineArray[6]},{lineArray[10]}")

    return output

# Parse the output file - separete output lines for in and out-of place.
# For each matching line in the file return size and time for out-of-place and in-place as separate lines
# Code is based on based on https://github.com/microsoft/msccl-leaderboard/blob/main/generate_graphs.py#L48
def parse_test_log_inout_separate(path):
    # Example input
    # [1,0]<stdout>:#
    # [1,0]<stdout>:#                                                              out-of-place                       in-place
    # [1,0]<stdout>:#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    # [1,0]<stdout>:#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    # [1,0]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18075:18075 [0] NCCL INFO Launch mode Parallel
    # [1,0]<stdout>:        1024           256     float     sum      -1    75.18    0.01    0.03      0    78.30    0.01    0.02      0
    # [1,0]<stdout>:        2048           512     float     sum      -1    79.54    0.03    0.05      0    82.56    0.02    0.05      0

    # output is an array of string
    output = []

    # Create a pattern for lines that begin with '[1,0]<stdout>: ' and contain a number followed by a space and then a number
    pattern = re.compile(f'(\[1,0\]<stdout>:)?\s*(\d+)\s+(\d+)\s+')
    with open(path) as f:
        for line in f.readlines():
            m = pattern.match(line)
            if m is not None:
                # split the line into an array of strings were each string is separated by one or more spaces
                lineArray = line.split()
                # Create a common separated string with the size, first time columns
                output.append(f"outPlace,{lineArray[1]},{lineArray[6]}")

                # Create a common separate string with the size and the second time columns
                output.append(f"inPlace,{lineArray[1]},{lineArray[10]}")

    return output




# Multi-line string to be used as the contents of the testRun program
exampleOutput = '''
[1,5]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18101:18135 [5] NCCL INFO comm 0x7f3804000fa0 rank 5 nranks 16 cudaDev 5 busId 3bfd00000 - Init COMPLETE
[1,0]<stdout>:NCCL version 2.12.12.MSCCL.0.7.3+cuda11.6
[1,2]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18098:18143 [2] NCCL INFO comm 0x7f0ec0000fa0 rank 2 nranks 16 cudaDev 2 busId d34d00000 - Init COMPLETE
[1,4]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18093:18141 [4] NCCL INFO comm 0x7f76ac000fa0 rank 4 nranks 16 cudaDev 4 busId bd1000000 - Init COMPLETE
[1,0]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18075:18131 [0] NCCL INFO comm 0x7f26f0000fa0 rank 0 nranks 16 cudaDev 0 busId f6a800000 - Init COMPLETE
[1,0]<stdout>:#
[1,0]<stdout>:#                                                              out-of-place                       in-place
[1,0]<stdout>:#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
[1,0]<stdout>:#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
[1,0]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18075:18075 [0] NCCL INFO Launch mode Parallel
[1,0]<stdout>:        1024           256     float     sum      -1    75.18    0.01    0.03      0    78.30    0.01    0.02      0
[1,0]<stdout>:        2048           512     float     sum      -1    79.54    0.03    0.05      0    82.56    0.02    0.05      0
[1,0]<stdout>:        4096          1024     float     sum      -1    84.29    0.05    0.09      0    116.1    0.04    0.07      0
[1,0]<stdout>:        8192          2048     float     sum      -1    115.0    0.07    0.13      0    111.7    0.07    0.14      0
[1,0]<stdout>:       16384          4096     float     sum      -1    129.4    0.13    0.24      0    126.7    0.13    0.24      0
[1,0]<stdout>:       32768          8192     float     sum      -1    208.2    0.16    0.30      0    205.5    0.16    0.30      0
[1,0]<stdout>:       65536         16384     float     sum      -1    313.7    0.21    0.39      0    340.9    0.19    0.36      0
[1,0]<stdout>:      131072         32768     float     sum      -1    297.7    0.44    0.83      0    297.5    0.44    0.83      0
[1,0]<stdout>:      262144         65536     float     sum      -1    521.7    0.50    0.94      0    503.8    0.52    0.98      0
[1,0]<stdout>:      524288        131072     float     sum      -1    895.8    0.59    1.10      0    901.9    0.58    1.09      0
[1,0]<stdout>:     1048576        262144     float     sum      -1   1255.6    0.84    1.57      0   1215.6    0.86    1.62      0
[1,0]<stdout>:     2097152        524288     float     sum      -1   1878.7    1.12    2.09      0   1876.8    1.12    2.10      0
[1,0]<stdout>:     4194304       1048576     float     sum      -1   2246.7    1.87    3.50      0   2229.5    1.88    3.53      0
[1,0]<stdout>:     8388608       2097152     float     sum      -1   4018.2    2.09    3.91      0   4017.9    2.09    3.91      0
[1,0]<stdout>:    16777216       4194304     float     sum      -1   7561.5    2.22    4.16      0   7561.6    2.22    4.16      0
[1,0]<stdout>:    33554432       8388608     float     sum      -1    13589    2.47    4.63      0    13595    2.47    4.63      0
[1,0]<stdout>:    67108864      16777216     float     sum      -1    26102    2.57    4.82      0    26108    2.57    4.82      0
[1,0]<stdout>:   134217728      33554432     float     sum      -1    51259    2.62    4.91      0    51135    2.62    4.92      0
[1,0]<stdout>:   268435456      67108864     float     sum      -1   101295    2.65    4.97      0   101395    2.65    4.96      0
'''



# Run the program
if __name__ == "__main__":
    main()
