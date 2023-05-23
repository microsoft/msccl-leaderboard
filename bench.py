import os
import shutil
import sys
import re
import datetime
import fnmatch
import argparse
import xml.etree.ElementTree as ET


# This python program is runs NCCL benchmarks tests (https://github.com/NVIDIA/nccl-tests)
# with and without MSCCL enabled and appends the results to a CSV file for further analysis.
# Prerequisites:
# 1. Install python 3.6 or greater
# 2. Clone and build the NCCL test repository https://github.com/NVIDIA/nccl-tests.
# 3. Clone and build the MSCCL repository https://github.com/microsoft/msccl.git
# 4. Install MSCCL tools https://github.com/microsoft/msccl-tools.git
# 5. Use MSCCL tools to generate the MSCCL xml files
#    or use pre-generated xml files from https://github.com/parasailteam/sccl-presynth/tree/main/sccl_presynth
#
# Usage: bench.py <bench.py arguments> <mpirun command> <mpirun arguments> 
#                 <optional launcher.sh> <path to nccl-tests/build> <nccl-tests arguments>
# 
# See the argparse.epilog in main()) for more details about the arguments.
# Example command line: python bench.py --mode run --directory sccl-presynth/sccl_presynth --filter *gather.n16* --output output
#
# Example usage:
# python bench.py --runMode run --directory ~/sccl-presynth/sccl_presynth --filter *gather.n16* -o ~/mikezw/output mpirun --bind-to numa --tag-output --allow-run-as-root -hostfile /job/hostfile -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x PATH -x LD_LIBRARY_PATH=~/msccl/build/lib/:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x NCCL_NET_GDR_LEVEL=5 -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ~/msccl-tools/msccl/autosynth/msccl_ndv2_launcher.sh ~/nccl-tests/build -b 1KB -e 1MB -f 2 -g 1 -c 1 -w 100 -n 100

# Details on "--mode test":
# The "--mode test" option is used to test the program without actually running the benchmarks.
# It will still iterate overall of the files in the directory and filter but it will not run the benchmarks.
# The file contents do not matter as the variable exampleMCCLInput is used instead of the file contents.
# The output is also faked with the results being the variable exampleOutput.
#
# TODO comments are scattered throughout the code to indicate areas that need improvement.
#   Primarily these are hard-coded constants that should probably be command line arguments.
#   Such as the separator for the CSV file.
#
def main():

    # Default mpirun commandline
    defaultMpirunCmdLine = "mpirun --bind-to numa --tag-output --allow-run-as-root" + \
        " -hostfile /job/hostfile -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0"

    # Obtain the home directory of the user
    home_dir = os.path.expanduser("~")

    # Iterage over the command line arguments until we find the mpirun command.
    # Save the arguments before the mpirun command as the arguments for this script.
    # The arguments after the mpirun command are expected to be the arguments to mpirun and the nccl-tests.
    #
    # Note: and alternative to this iteration would be the use argparse.parse_known_args(), however
    # it will misinterpret arguments that have a common prefix with expected arguments. For example
    # -hostfile used for mpirun and with match the -h for --help. The approach here avoids this issue.
    #
    mpirun_index = None
    for i, arg in enumerate(sys.argv):
        if arg.endswith("mpirun"):
            mpirun_index = i
            break

    # Save the arguments before the mpirun command
    script_args = sys.argv[1:mpirun_index]

    # Use RawDescriptionHelpFormatter to allow newlines in the description
    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    # Setup an argument parser to parse the command line arguments for the script
    parser = argparse.ArgumentParser(\
        description='Run NCCL benchmarks tests using NCCL and MSCCL', 
        formatter_class=Formatter,
        epilog='''After the script arguments above the next argument must be the mpirun command
(or a path whose suffix is mprirun) followed by the arguments to the mpirun command.\
\nNote that the mpirun -np parameter does not need to be specified as it will be added\
 to the mpirun command based on the MSCC xml file. If -np is provided it will be\
 checked against the expected value from the MSCCL xml file and an error will result\
 if they do not match.\
\n \nNext the path to an optional launcher.sh script can be provided
     (for example msccl-tools/msccl/autosynth/msccl_ndv2_launcher.sh)\
\n \nNext the path to the nccl-tests/build directory must be provided (for example ~/nccl-tests/build)\
\n     This script will automatically append the nccl test name that corresponds to the MSCCL xml file\
\n     to the end of the nccl-tests directory path.
\nFinally the arguments to the nccl-tests must be provided.

          \nExample:\n\
          python bench.py --runMode run --directory ~/sccl-presynth/sccl_presynth --filter *gather.n16* -o ~/mikezw/output mpirun --bind-to numa --tag-output --allow-run-as-root -hostfile /job/hostfile -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x PATH -x LD_LIBRARY_PATH=~/msccl/build/lib/:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x NCCL_NET_GDR_LEVEL=5 -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ~/msccl-tools/msccl/autosynth/msccl_ndv2_launcher.sh ~/nccl-tests/build -b 1KB -e 1MB -f 2 -g 1 -c 1 -w 100 -n 100''')
    
    # Add the script arguments
    parser.add_argument('-r', '--runMode', type=str, help='runMode: is either "test" or "run" where "test" indicates that the program should run in test mode and "run" indicates that the program should actually run the benchmarks.', default="run", choices=["test", "run"])
    parser.add_argument('-d', '--directory', type=str, help='directory: is the directory where the MSCCL xml files are located', default=home_dir + "/sccl-presynth/sccl_presynth")
    parser.add_argument('-f', '--filter', type=str, help='filter: is a filter in the style of https://docs.python.org/3/library/fnmatch.html for files in the directory that will be tested', required=True)
    parser.add_argument('-o', '--outputDirectory', type=str,\
                         help='''outputDirectory: is the directory where the output files will be placed.
                         Each mpirun execution will result in a separate output file named after the MSCCL xml file
                         with a suffix of nccl_result.txt for a pure NCCL run and suffix msccl_result.txt for an MSCCL run.
                         All test results will be parsed from these files and appended to OUTPUTDIRECTORY/result.txt with
                         tab separated columns. ''', required=True)

    # Append additional strings to the usage string to make it more clear
    parser.usage = parser.format_usage() +\
         " <mpirun command> <mpirun arguments> <optional launcher.sh> <path to nccl-tests/build> <nccl-tests arguments>" +\
         "\nUse --help to see the full list of arguments."
    
        # Parse the command line arguments from the Namespace within the returned tuple
    args = parser.parse_args(script_args)
   
    # Make --help show the default values
    parser.set_defaults(**vars(args))

    # Assign the command line arguments to variables
    mode = args.runMode
    directory = args.directory
    filter = args.filter
    outputDirectory = args.outputDirectory

    # Save the arguments after the mpirun command
    remainder_args = sys.argv[mpirun_index:]
    
    print(f"script arguments: {args}    remainder_args: {remainder_args}")

    if mpirun_index is None:
        print("Error: mpirun command not found in arguments.")
        sys.exit(1)

    # In run mode, verify that the first unknown argument is the mpirun command
    # and check that mpirun is in the OS path if it does not have a path separator.
    # In test mode, just verify that the first unknown argument is the mpirun command.
    mpirun = remainder_args.pop(0)
    if mode == "run":
        if os.path.sep not in mpirun:
            if shutil.which(mpirun) is None:
                print(f"Error: {mpirun} must be a valid executable for the mpirun command")
                return
        elif os.path.exists(mpirun) and not os.access(mpirun, os.X_OK): 
            print(f"Error: {mpirun} must be a valid executable for the mpirun command")
            return
    else:
        if mpirun.endswith("mpirun") == False:
            print(f"Error: {mpirun} must be the mpirun command")
            return
            
    # After all of the mpirun arguments we exepect to see an argument with a suffix that is one of these strings
    NCCL_TESTS_SUFFIX = f"nccl-tests{os.path.sep}build"
    NCCL_LAUNCH_SUFFIX = "launcher.sh"

    # Pop all elements in remainder_args into mpirun_args until we detect either the ncclTestsSuffix or ncclLaunchSuffix
    mpirun_args = []
    np_value = None
    while len(remainder_args) > 0 and\
          remainder_args[0].endswith(NCCL_TESTS_SUFFIX) == False and remainder_args[0].endswith(NCCL_LAUNCH_SUFFIX) == False:
        # If we encounter the -np parameter, remember its value (the next argument), but do not add it to mpirunArgs
        if remainder_args[0] == "-np":
            remainder_args.pop(0)
            np_value = remainder_args.pop(0)
            continue
        # If we encounter a -x parameter with a MSCCL_XML_FILES environment variable, flag an error
        if remainder_args[0] == "-x" and remainder_args[1].startswith("MSCCL_XML_FILES="):
            print(f"Error: MSCCL_XML_FILES environment variable should not be set in the mpirun command line since the --directory argument will be used instead.")
            return
        mpirun_args.append(remainder_args.pop(0))

    # If the next argument is the ncclLaunchSuffix then we need to pop it and save it
    nccl_launch = ""
    if len(remainder_args) > 0 and remainder_args[0].endswith(NCCL_LAUNCH_SUFFIX):
        nccl_launch = remainder_args.pop(0)
    
    # If the next argument is the ncclTestsSuffix then we need to pop it and save it
    # Otherwise report an error
    nccl_tests = None
    if len(remainder_args) > 0 and remainder_args[0].endswith(NCCL_TESTS_SUFFIX):
        nccl_tests = remainder_args.pop(0)
    else:
        print(f"Error: nccl-tests{os.path.sep}build not found in command line arguments")
        return
    
    # If this is run mode, ensure ncclTests is a valid directory
    if mode == "run" and os.path.isdir(nccl_tests) == False:
        print(f"Error: nccl-tests/build must be a valid directory")
        return
    
    # At this point the remainder_args should be for the nccl-tests command. Save them
    nccl_tests_args = remainder_args

    # Separator to use in the CSV result file
    # TODO: consider making this a command line argument
    separator = "\t"

    # This is the CSV file with the final results
    result_file_name = "results.txt"

    # Make sure the outputDirectory exists and if it doesn't create it.
    if not os.path.exists(outputDirectory):
        print("Creating output directory: " + outputDirectory)
        os.makedirs(outputDirectory)

    # Make sure a file named resultFile exists in outputDirectory and if it doesn't create it.
    # Make sure the file path works for both linux and Windows
    resultFile = os.path.join(outputDirectory, result_file_name)
    if not os.path.exists(resultFile):
        print("Creating result file: " + resultFile)
        # Put header row into the result file
        with open(resultFile, "w") as f:
            f.writelines(format_header_row(separator))

    prossessed_files = []

    # get current date and time as a string
    time_of_test = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the result file for appending
    resultsF = open(resultFile, "a")

    # For each file filename in directory that matches the filter in the name do the following:
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, filter):

            # Track the names of all files processed from the directory
            prossessed_files.append(filename)

            # Parse the XML file to get the algorithm attributes for the benchmark
            # Example: algo {'name': 'Allgather(n=16)-Distributed...', 'nchannels': '8', 'nchunksperloop': '128', 'proto': 'Simple', 'ngpus': '16', 'coll': 'allgather', 'inplace': '1'}
            if (mode == "test"):
                algorithm_attributes = parse_msccl_xml(exampleMCCLInput)
            else:
                # Parse the XML file to get the algorithm attributes for the benchmark. Input is the contents of the file.
                algorithm_attributes = parse_msccl_xml(open(os.path.join(directory, filename), "r").read())

            # If the -np parameter was specified, make sure that it matches the ngpus attribute in the XML file
            assert algorithm_attributes.get("ngpus") != None, f"Error: ngpus attribute not found in XML file {filename}"
            if np_value != None and int(np_value) != int(algorithm_attributes.get("ngpus")):
                print(f"Error: -np value {np_value} does not match ngpus value {algorithm_attributes.get('ngpus')} in XML file {filename}")
                return

            output_file_list = run_benchmark(mode, algorithm_attributes, directory, filename, outputDirectory,\
                                            mpirun, mpirun_args, nccl_launch, nccl_tests, nccl_tests_args)

            # For each file in outputFileList parse the file to find the results
            # And append them to the resultFile in CSV format
            for output_file_name in output_file_list:
                # Extract the NCCL version from the output file based on a line like [1,0]<stdout>:NCCL version 2.12.12.MSCCL.0.7.3+cuda11.6
                with open(output_file_name, "r") as f:
                    for line in f:
                        if "NCCL version" in line:
                            nccl_version = line.split("NCCL version ")[1].strip()
                            break

                # Parse the output file to get each line of performance results
                lines = parse_test_log(output_file_name)
                for line in lines:
                    library = "MSCCL" if "msccl_result" in output_file_name else "NCCL"
                    in_place_str = algorithm_attributes.get("inplace")
                    assert in_place_str != None, f"Error: inplace attribute not found in XML file {filename}"
                    in_place = int(in_place_str)

                    # For each line in the output file write one output line for a MSCCL run corresponding
                    # to in_place or out_of_place (since and MSCCL algorithm is only one or the other)
                    # and and two lines for a NCCL run covering both in_place and out_of_place.
                    for place in {0, 1}:
                        if library == "MSCCL" and (place == 0 and in_place == 1) or (place == 1 and in_place == 0):
                            continue
                        
                        # Write the results to the result file in CSV format.
                        resultsF.write(algorithm_attributes.get("coll")
                            + separator + filter
                            + separator + time_of_test
                            + separator + algorithm_attributes.get("ngpus")
                            + separator + algorithm_attributes.get("proto")
                            + separator + library
                            + separator + nccl_version
                            + separator + str(place) # in_place flag (0 or 1 for out-of-place or in-place)
                            + separator + line[0] # buffer size
                            + separator + (line[2] if place == 1 else line[1]) # if in_place is 1 then use line[2] otherwise use line[1]
                            + separator + algorithm_attributes.get("name")
                            + separator + filename
                            + '\n')

    resultsF.close()

    # report the names of all files processed from the directory
    if (len(prossessed_files) == 0):
        print("No files were processed. Consider changing the filter or nGPU command line arguments.")
    else:
        print(f"Processed Files = {prossessed_files}")


# Execute mprirun for the NCCL benchmark corresponding to the filename, place results in the output folder
# then returns a list of the names of the output files
def run_benchmark(mode, algorithmAttributes, inputDirectory, filename, outputDirectory,\
                    mpirun, mpirunArgs, ncclLaunch, ncclTests, ncclTestArgs) -> list:

    """
    Runs a benchmark for a given file and returns a list of the names of the output files.

    Args:
        mode (str): The mode in which to run the benchmark.
        algorithmAttributes (dict): A dictionary containing the attributes of the algorithm to be benchmarked.
        inputDirectory (str): The directory containing the input files.
        filename (str): The name of the file to be benchmarked.
        outputDirectory (str): The directory where the output files will be stored.
        mpirun (str): path to the mpirun command.
        mpirunArgs (str): arguments for the mpirun command.
        ncclLaunch (str): The path to the NCCL launch command (may be None).
        ncclTests (str): The path to the NCCL tests build folder.
        ncclTestArgs (str): The arguments to be passed to the NCCL tests.

    Returns:
        list: A list of the names of the output files.
    """
    print(f"\nRun Benchmark in mode {mode} for file: " + filename)

    # The list of output files that will be returned (one for NCCl and one for MSCCL)
    outputFileList = []

    # Extract the substring name of the algorithm up until the first "." in the file name. Call this substring "algorithm"
    mscclCollective = algorithmAttributes.get("coll")

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
    if mscclCollective.lower() == "allreduce":
        ncclAlgorithm = "all_reduce"
    elif mscclCollective.lower() == "allgather":
        ncclAlgorithm = "all_gather"
    elif mscclCollective.lower() == "alltoall":
        ncclAlgorithm = "all_to_all"
    else:
        raise Exception("Unexpected algorithm: " + mscclAlgorithmNamePrefix)

    ncclPerfTest = ncclAlgorithm + "_perf"

    npParameter = f" -np {algorithmAttributes.get('ngpus')}"

    # Direct the output to a new file called filename_nccl_result
    # Delete output file if it already exists
    ncclOutputFile = os.path.join(outputDirectory, f"{filename}_nccl_result.txt")
    if os.path.exists(ncclOutputFile):
        os.remove(ncclOutputFile)

    mpirunArgsString = " ".join(mpirunArgs)
    ncclTestArgsString = " ".join(ncclTestArgs)
    ncclCommandLine = f"{mpirun} {npParameter} {mpirunArgsString} {ncclLaunch} {os.path.join(ncclTests, ncclPerfTest)} {ncclTestArgsString} > {ncclOutputFile}"
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
    mscclXMLFile = " -x MSCCL_XML_FILES=" + os.path.join(inputDirectory, filename)
    mscclOutputFile = os.path.join(outputDirectory, f"{filename}_msccl_result.txt")
    mscclCommandLine = f"{mpirun} {npParameter} {mpirunArgsString} {mscclXMLFile} {ncclLaunch} {os.path.join(ncclTests, ncclPerfTest)} {ncclTestArgsString} > {mscclOutputFile}"
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

            # If the output file contains NCCL warnings or errors the benchmark failed.
            # Don't look for just WARN or ERROR because the output file may contain other innoccuous warnings.
            if "NCCL WARN" in contents.upper() or "NCCL ERROR" in contents.upper():
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

# Parse MSCCL XML file content and return a dictionary of attributes for the algo element in the file:
#   - algo: algorithm name
#   - proto: protocol name
#   - ngpus: number of GPUs
#   - coll: collective name
#   - inplace: boolean for in-place or out-of-place
# See example input from exampleMSCCLInput variable below
def parse_msccl_xml(xmlContent):
    # parse the xmlContent
    root = ET.fromstring(xmlContent)
    assert root.tag == "algo"
    return root.attrib

# Parse the output file
# For each matching line in the file return size and time for out-of-place and in-place
# param path: path to the file to parse
# Code is based on based on https://github.com/microsoft/msccl-leaderboard/blob/main/generate_graphs.py#L48
def parse_test_log(path) -> list[tuple[int, float, float]]:
    # Example input
    # [1,0]<stdout>:#
    # [1,0]<stdout>:#                                                              out-of-place                       in-place
    # [1,0]<stdout>:#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    # [1,0]<stdout>:#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    # [1,0]<stdout>:az-eus-v100-32gb-5-worker-zphjiy:18075:18075 [0] NCCL INFO Launch mode Parallel
    # [1,0]<stdout>:        1024           256     float     sum      -1    75.18    0.01    0.03      0    78.30    0.01    0.02      0
    # [1,0]<stdout>:        2048           512     float     sum      -1    79.54    0.03    0.05      0    82.56    0.02    0.05      0

    # output is an array of lists
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
                output.append([lineArray[1], lineArray[6], lineArray[10]])

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

# Format the header row for the CSV output file
# param: separator is the character used to separate the columns in the CSV file
# Returns the string for the header row
def format_header_row(separator) -> str:
    header = ('Collective'
        + separator + 'Filter'
        + separator + 'TimeOfTest'
        + separator + 'nGPUs'
        + separator + 'Protocol'
        + separator + 'Library'
        + separator + 'NcclVersion'
        + separator + 'InPlace'
        + separator + 'Size'
        + separator + 'Time'
        + separator + 'MSCCL_Algo_Name'
        + separator + 'MSCCL_File'
        + '\n')
    return header

# This is and example of a MSCCL input file
# Based on sccl-presynth/sccl_presynth/Allgather.n16-DistributedRelayedSwitch.localDGX1.copies2-steps20-gurobisol-improve-1630536923_i8_scRemote1_IBContig_h5-noring.sccl.xml
exampleMCCLInput = '''
<algo name="Allgather(n=16)-DistributedRelayedSwitch(local=DGX1,copies=2)-steps=20-gurobisol-improve-1630536923" nchannels="8" nchunksperloop="128" proto="Simple" ngpus="16" coll="allgather" inplace="1">
 <gpu id="0" i_chunks="8" o_chunks="128" s_chunks="0">
    <tb id="0" send="-1" recv="1" chan="0">
      <step s="0" type="r" srcbuf="o" srcoff="8" dstbuf="o" dstoff="8" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="1" type="r" srcbuf="o" srcoff="16" dstbuf="o" dstoff="16" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="2" type="r" srcbuf="o" srcoff="56" dstbuf="o" dstoff="56" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="3" type="r" srcbuf="o" srcoff="72" dstbuf="o" dstoff="72" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="4" type="r" srcbuf="o" srcoff="96" dstbuf="o" dstoff="96" cnt="1" depid="-1" deps="-1" hasdep="0"/>
      <step s="5" type="r" srcbuf="o" srcoff="88" dstbuf="o" dstoff="88" cnt="1" depid="-1" deps="-1" hasdep="0"/>
      <step s="6" type="r" srcbuf="o" srcoff="80" dstbuf="o" dstoff="80" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="7" type="r" srcbuf="o" srcoff="112" dstbuf="o" dstoff="112" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="8" type="r" srcbuf="o" srcoff="120" dstbuf="o" dstoff="120" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="9" type="r" srcbuf="o" srcoff="104" dstbuf="o" dstoff="104" cnt="1" depid="-1" deps="-1" hasdep="0"/>
    </tb>
    <tb id="1" send="-1" recv="1" chan="1">
      <step s="0" type="r" srcbuf="o" srcoff="9" dstbuf="o" dstoff="9" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="1" type="r" srcbuf="o" srcoff="17" dstbuf="o" dstoff="17" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="2" type="r" srcbuf="o" srcoff="57" dstbuf="o" dstoff="57" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="3" type="r" srcbuf="o" srcoff="73" dstbuf="o" dstoff="73" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="4" type="r" srcbuf="o" srcoff="97" dstbuf="o" dstoff="97" cnt="1" depid="-1" deps="-1" hasdep="0"/>
      <step s="5" type="r" srcbuf="o" srcoff="89" dstbuf="o" dstoff="89" cnt="1" depid="-1" deps="-1" hasdep="0"/>
      <step s="6" type="r" srcbuf="o" srcoff="81" dstbuf="o" dstoff="81" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="7" type="r" srcbuf="o" srcoff="113" dstbuf="o" dstoff="113" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="8" type="r" srcbuf="o" srcoff="121" dstbuf="o" dstoff="121" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="9" type="r" srcbuf="o" srcoff="105" dstbuf="o" dstoff="105" cnt="1" depid="-1" deps="-1" hasdep="0"/>
    </tb>
  </gpu>
</algo>
'''

# This is an example of output from the NCCL benchmark. This is used for testing.
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


# Usage: this program takes the following arguments: --mode, --directory, --filter, --outputDirectory, --endBufSize
# See the main function for more details about the arguments.
# Example command line: python bench.py --mode run --directory sccl-presynth/sccl_presynth --filter *gather.n16* --output output
#
# Details on "--mode test":
# The "--mode test" option is used to test the program without actually running the benchmarks.
# It will still iterate overall of the files in the directory and filter but it will not run the benchmarks.
# The file contents do not matter as the variable exampleMCCLInput is used instead of the file contents.
# The output is also faked with the results being the variable exampleOutput.
#
# TODO comments are scattered throughout the code to indicate areas that need improvement.
#   Primarily these are hard-coded constants that should probably be command line arguments.
#   Such as the separator for the CSV file.
#
def main_prior_version():

    # Default mpirun commandline
    defaultMpirunCmdLine = "mpirun --bind-to numa --tag-output --allow-run-as-root" + \
        " -hostfile /job/hostfile -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0"



    # Obtain the home directory of the user
    homeDir = os.path.expanduser("~")

    # Setup an argument parser to parse the command line arguments
    parser = argparse.ArgumentParser(description='Run NCCL benchmarks tests using NCCL and MSCCL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--runMode', type=str, help='runMode: is either "test" or "run" where "test" indicates that the program should run in test mode and "run" indicates that the program should actually run the benchmarks.', default="run", choices=["test", "run"])
    parser.add_argument('-d', '--directory', type=str, help='directory: is the directory where the MSCCL xml files are located', default=homeDir + "/sccl-presynth/sccl_presynth")
    parser.add_argument('-f', '--filter', type=str, help='filter: is a filter in the style of https://docs.python.org/3/library/fnmatch.html for files in the directory that will be tested', required=True)
    parser.add_argument('-o', '--outputDirectory', type=str, help='outputDirectory: is the directory where the output files will be placed', required=True)
    parser.add_argument('-m', '--msccl', type=str, help='path to msccl repo working directory', default=homeDir + "/msccl")
    parser.add_argument('-t', '--mscclTools', type=str, help='path to msccl-tools repo working directory', default=homeDir + "/msccl-tools")
    parser.add_argument('-n', '--ncclTests', type=str, help='path to nccl-tests repo working directory', default=homeDir + "/nccl-tests")
    parser.add_argument('-a', '--ncclTestArgs', type=str, help='nccl test arguments (see --help on the nccl test)', default="-b 1KB -e 32MB -f 2 -g 1 -c 1 -w 100 -n 100")
    parser.add_argument('-c', '--mpirunCmdLine', type=str, help='''
        mpirun: mpirun command line with any arguments necessary for the test runs.
            * This should not include the paths to the nccl-tests or the MSCCL files as
              these will be appended to the final command line automatically by the script.
            * This should not include the -np argument as this will be automatically
              obtained from the msccl xml file.''',
        default = defaultMpirunCmdLine)

    # Parse the command line arguments
    args = parser.parse_args()

    # Make --help show the default values
    parser.set_defaults(**vars(args))

    # Assign the command line arguments to variables
    mode = args.runMode
    directory = args.directory
    filter = args.filter
    outputDirectory = args.outputDirectory
    msccl = args.msccl
    mscclTools = args.mscclTools
    ncclTests = args.ncclTests
    mpirunCmdLine = args.mpirunCmdLine

    # Print all of the command line arguments
    print(f"Arguments: {args}")

    # Separator to use in the CSV result file
    # TODO: consider making this a command line argument
    separator = "\t"

    # This is the CSV file with the final results
    resultFileName = "results.txt"

    # Make sure the outputDirectory exists and if it doesn't create it.
    if not os.path.exists(outputDirectory):
        print("Creating output directory: " + outputDirectory)
        os.makedirs(outputDirectory)

    # Make sure a file named resultFile exists in outputDirectory and if it doesn't create it.
    # Make sure the file path works for both linux and Windows
    resultFile = os.path.join(outputDirectory, resultFileName)
    if not os.path.exists(resultFile):
        print("Creating result file: " + resultFile)
        # Put header row into the result file
        with open(resultFile, "w") as f:
            f.writelines(format_header_row(separator))

    prossessedFiles = []


    # get current date and time as a string
    timeOfTest = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the result file for appending
    resultsF = open(resultFile, "a")

    # For each file filename in directory that matches the filter in the name do the following:
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, filter):

            # Track the names of all files processed from the directory
            prossessedFiles.append(filename)

            # Parse the XML file to get the algorithm attributes for the benchmark
            # Example: algo {'name': 'Allgather(n=16)-Distributed...', 'nchannels': '8', 'nchunksperloop': '128', 'proto': 'Simple', 'ngpus': '16', 'coll': 'allgather', 'inplace': '1'}
            if (mode == "test"):
                algorithmAttributes = parse_msccl_xml(exampleMCCLInput)
            else:
                # Parse the XML file to get the algorithm attributes for the benchmark. Input is the contents of the file.
                algorithmAttributes = parse_msccl_xml(open(os.path.join(directory, filename), "r").read())

            outputFileList = run_benchmark(mode, algorithmAttributes, directory, filename, \
                                            msccl, mscclTools, ncclTests, args.ncclTestArgs, mpirunCmdLine, \
                                            algorithmAttributes.get("ngpus"), outputDirectory)

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
                data = parse_test_log(outputFile, separator)
                for line in data:
                    library = "MSCCL" if "msccl" in outputFile else "NCCL"
                    # Write the results to the result file in CSV format.
                    resultsF.write(algorithmAttributes.get("coll")
                        + separator + filter
                        + separator + timeOfTest
                        + separator + algorithmAttributes.get("ngpus")
                        + separator + algorithmAttributes.get("proto")
                        + separator + library
                        + separator + ncclVersion
                        + separator + line # line is a string with Size, Time_OutOfPlace, Time_InPlace in CSV format.
                        + separator + algorithmAttributes.get("name")
                        + separator + filename
                        + '\n')

    resultsF.close()

    # report the names of all files processed from the directory
    if (len(prossessedFiles) == 0):
        print("No files were processed. Consider changing the filter or nGPU command line arguments.")
    else:
        print(f"Processed Files = {prossessedFiles}")

