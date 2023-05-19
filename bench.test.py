import unittest
import subprocess
from bench import main

class TestBench(unittest.TestCase):
    def test_main(self):
        # Run the main function with some test arguments
        result = subprocess.run(['python', 'bench.py', '--runMode', 'test', '--directory', '.', '--filter', '*', '--outputDirectory', 'output', 'mpirun', 'nccl-tests'], capture_output=True, text=True)

        # Check that the output is as expected
        expected_output = 'Known Arguments: Namespace(directory=\'.\', filter=\'*\', outputDirectory=\'output\', runMode=\'test\'), Unknown Arguments: [\'mpirun\', \'nccl-tests\']\nmode: test, directory: ., filter: *, outputDirectory: output\n'
        self.assertEqual(result.stdout, expected_output)

if __name__ == '__main__':
    unittest.main()