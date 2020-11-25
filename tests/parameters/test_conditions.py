from experitur import Experiment
from experitur.core.context import Context
from experitur.parameters import Const, Grid, Conditions


def test_Conditions(tmp_path):
    with Context(str(tmp_path)):

        @Experiment()
        def exp(trial):
            pass

        sampler = Conditions("x", {1: Const(y=1), 2: Const(y=2)})
        assert sorted(sampler.independent_parameters.items()) == [
            ("x", [1, 2]),
            ("y", [1, 2]),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in sampler.generate(exp)
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1)),
            (("x", 2), ("y", 2)),
        ]

        sampler = Conditions("x", {1: Const(y=1), 2: Grid({"y": [2, 3]})})
        assert sorted(sampler.independent_parameters.items()) == [
            ("x", [1, 2]),
            ("y", [1, 2, 3]),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in sampler.generate(exp)
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1)),
            (("x", 2), ("y", 2)),
            (("x", 2), ("y", 3)),
        ]

        # Test passing sub-configurations as simple dict (should get converted)
        sampler = Conditions("x", {1: {"y": [1]}})
        assert sorted(sampler.independent_parameters.items()) == [
            ("x", [1]),
            ("y", [1]),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in sampler.generate(exp)
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1)),
        ]

        # Test passing list of sub-configurations (only invariant)
        sampler = Conditions("x", {1: [Const(y=1), Const(z=1)]})
        print(str(sampler.sub_generators))
        assert sorted(sampler.independent_parameters.items()) == [
            ("x", [1]),
            ("y", [1]),
            ("z", [1]),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in sampler.generate(exp)
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1), ("z", 1)),
        ]

        # Test passing list of sub-configurations (variant)
        sampler = Conditions("x", {1: [Const(y=1), {"z": [1, 2]}]})
        print(str(sampler.sub_generators))
        assert sorted(sampler.independent_parameters.items()) == [
            ("x", [1]),
            ("y", [1]),
            ("z", [1, 2]),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in sampler.generate(exp)
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1), ("z", 1)),
            (("x", 1), ("y", 1), ("z", 2)),
        ]
