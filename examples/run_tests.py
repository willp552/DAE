import os

"""
Python script which runs all test scripts.
"""

DIR = "../testing"
FILES = ["ocp_index_one_test_I.py", "ocp_index_two_test_I.py"]

for test in FILES:
    try:
        os.system("python {0} --folder {1}".format(test, DIR))
    except:
        with open(os.path.join(DIR, "log.txt"), "a+") as f:
            f.write("Test {0} failed...".format(test))
