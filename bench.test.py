import unittest
import subprocess
from bench import main

class TestBench(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_main(self):
        # Run the main function with some test arguments that should not fail
        result = subprocess.run(['python', 'bench.py', '--runMode', 'test', '--directory', '.',\
                                  '--filter', '*', '--outputDirectory', 'output', 'mpirun', 'nccl-tests/build'],\
                                capture_output=True, text=True)
        self.assertFalse(result.stderr.find('error') > -1)

        # Run with missing mpirun argument and verify that usage is printed
        result = subprocess.run(['python', 'bench.py', '--runMode', 'test', '--directory', '.',\
                                    '--filter', '*', '--outputDirectory', 'output', 'nccl-tests/build'],\
                                    capture_output=True, text=True)
        self.assertTrue(result.stderr.find('mpirun') > -1)
        self.assertTrue(result.stderr.find('usage') > -1)

        # Run with missing nccl-tests/build argument and verify that usage is printed
        result = subprocess.run(['python', 'bench.py', '--runMode', 'test', '--directory', '.',\
                                    '--filter', '*', '--outputDirectory', 'output', 'mpirun'],\
                                    capture_output=True, text=True)
        self.assertTrue(result.stderr.find('nccl-tests') > -1)
        self.assertTrue(result.stderr.find('usage') > -1)

        # Run with MSCCL_XML_FILES environment variable set and verify error is printed
        result = subprocess.run(['python', 'bench.py', '--runMode', 'test', '--directory', '.',\
                                    '--filter', '*', '--outputDirectory', 'output', 'mpirun', '-x', 'MSCCL_XML_FILES=foo'],\
                                    capture_output=True, text=True)
        # print("Result.stderr: " + result.stderr)
        # print("Result.stdout: " + result.stdout)
        self.assertTrue(result.stderr.find('MSCCL_XML_FILES') > -1)


if __name__ == '__main__':
    unittest.main()

