import sys

import pytest
from tqdm import tqdm

from experitur.helpers import tqdm_redirect


class TestException(Exception):
    pass


def test_redirect_stdout(capsys):
    # print
    with tqdm_redirect.redirect_stdout():
        print("test1")

    captured = capsys.readouterr()
    assert captured.out == "test1\n"

    # tqdm.write
    with tqdm_redirect.redirect_stdout():
        tqdm.write("test2")

    captured = capsys.readouterr()
    assert captured.out == "test2\n"

    # Nested print
    with tqdm_redirect.redirect_stdout():
        with tqdm_redirect.redirect_stdout():
            print("test3")

    captured = capsys.readouterr()
    assert captured.out == "test3\n"

    # Nested tqdm.write
    with tqdm_redirect.redirect_stdout():
        with tqdm_redirect.redirect_stdout():
            tqdm.write("test4")

    captured = capsys.readouterr()
    assert captured.out == "test4\n"

    # tqdm.write with explicit file
    with tqdm_redirect.redirect_stdout():
        tqdm.write("test5", file=sys.stdout)

    captured = capsys.readouterr()
    assert captured.out == "test5\n"

    # tqdm progress bar with explicit file
    with tqdm_redirect.redirect_stdout():
        for _ in tqdm(range(10), file=sys.stdout):
            pass

    # tqdm progress bar with explicit file, exception
    with pytest.raises(TestException):
        with tqdm_redirect.redirect_stdout():
            for _ in tqdm(range(10), file=sys.stdout):
                raise TestException()


def test_redirect_stdout_display():
    # tqdm.display
    with tqdm_redirect.redirect_stdout():
        pbar = tqdm(file=sys.stdout)
        pbar.display()
