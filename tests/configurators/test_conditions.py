from experitur import Experiment
from experitur import configurators
from experitur.core.context import Context
from experitur.configurators import Const, Grid, Conditions


def test_Conditions(tmp_path):
    with Context(str(tmp_path)):

        @Experiment()
        def exp(trial):
            pass

        configurator = Conditions(
            "x", {1: Const(y=1), 2: [Const(y=2), Grid({"z": [1, 2, 3]})]}
        )
        assert sorted(configurator.parameter_values.items()) == [
            ("x", (1, 2)),
            ("y", (1, 2)),
            ("z", (1, 2, 3)),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in configurator.build_sampler()
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1)),
            (("x", 2), ("y", 2), ("z", 1)),
            (("x", 2), ("y", 2), ("z", 2)),
            (("x", 2), ("y", 2), ("z", 3)),
        ]

        configurator = Conditions("x", {1: Const(y=1), 2: Grid({"y": [2, 3]})})
        assert sorted(configurator.parameter_values.items()) == [
            ("x", (1, 2)),
            ("y", (1, 2, 3)),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in configurator.build_sampler()
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1)),
            (("x", 2), ("y", 2)),
            (("x", 2), ("y", 3)),
        ]

        # Condition name overwrites sub-config name
        configurator = Conditions("x", {1: Const(x=2)})
        assert sorted(configurator.parameter_values.items()) == [
            ("x", (1,)),
        ]

        # Assert exististence of all specified values
        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in configurator.build_sampler()
        )
        assert samples == [
            (("x", 1),),
        ]

        # Test passing sub-configurations as simple dict (should get converted)
        configurator = Conditions("x", {1: {"y": [1]}})
        assert sorted(configurator.parameter_values.items()) == [
            ("x", (1,)),
            ("y", (1,)),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in configurator.build_sampler()
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1)),
        ]

        # Test passing list of sub-configurations (only invariant)
        configurator = Conditions("x", {1: [Const(y=1), Const(z=1)]})
        print(str(configurator.conditions))
        assert sorted(configurator.parameter_values.items()) == [
            ("x", (1,)),
            ("y", (1,)),
            ("z", (1,)),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in configurator.build_sampler()
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1), ("z", 1)),
        ]

        # Test passing list of sub-configurations (variant)
        configurator = Conditions("x", {1: [Const(y=1), {"z": [1, 2]}]})
        print(str(configurator.conditions))
        assert sorted(configurator.parameter_values.items()) == [
            ("x", (1,)),
            ("y", (1,)),
            ("z", (1, 2)),
        ]

        samples = sorted(
            tuple(sorted(d["parameters"].items())) for d in configurator.build_sampler()
        )

        # Assert exististence of all specified values
        assert samples == [
            (("x", 1), ("y", 1), ("z", 1)),
            (("x", 1), ("y", 1), ("z", 2)),
        ]
