from experitur.storage.yaml_dict import YAMLDict
import pytest
import glob
import os


@pytest.mark.parametrize('pattern', [None, "{}/foo.yaml"])
def test_yaml_dict(tmp_path, pattern):
    d = YAMLDict(str(tmp_path), pattern)

    # Create a false positive
    os.makedirs(tmp_path/d.pattern.format("ignore"))

    d["foo"] = {1: "foo", "bar": 2}
    assert d["foo"] == {1: "foo", "bar": 2}

    assert "foo" in d
    assert "foo" in d.keys()
    assert {1: "foo", "bar": 2} in d.values()

    print("glob:", glob.glob(str(tmp_path/"*")))

    print("list(d):", list(d))

    assert len(d) == 1

    print(str(tmp_path/"*"))
    print("glob:", glob.glob(str(tmp_path/"*")))

    del d["foo"]
    print("glob:", glob.glob(str(tmp_path/"*")))
    assert len(d) == 0

    with pytest.raises(KeyError):
        del d["foo"]
