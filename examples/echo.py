from pprint import pformat
import time


def run(working_directory, parameters):
    time.sleep(1)
    print("working_directory:", working_directory)
    print("parameters:", pformat(parameters))
