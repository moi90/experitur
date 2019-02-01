from pprint import pformat

from experitur.util import apply_parameters, set_default_parameters


def run(working_directory, parameters):
    s = "An apple, two bananas, three oranges and some grapes."

    print("parameters (before):", pformat(parameters))

    set_default_parameters("split_", parameters, s.split)

    print("parameters (after):", pformat(parameters))

    result = apply_parameters("split_", parameters, s.split)

    print("result:", result)
