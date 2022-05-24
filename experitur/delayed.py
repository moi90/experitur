from experitur.core.context import get_current_context
from experitur.util import callable_to_name
from typing import Callable, Generator, Optional
from experitur import Experiment


class DelayedExperiment(Experiment):
    func: Optional[Callable]

    def __init__(self):
        super().__init__()

    def run(self, *args, **kwargs):
        ctx = get_current_context()

        if self.func is None:
            raise ValueError("No function registered")

        # Func either returns a single Experiment or a generator of experiments
        result = self.func()

        if isinstance(result, Experiment):
            if result.name is None:
                result.name = self.name

            print("ACTIVE", result.active)

            with ctx.set_current_experiment(result):
                return result.run(*args, **kwargs)

        if isinstance(result, Generator):
            experiments = result
        else:
            raise ValueError(f"Unknown return type: {result} ({type(result)})")

        while True:
            try:
                experiment = next(experiments)
            except StopIteration as exc:
                if exc.value is not None:
                    raise ValueError(
                        f"{callable_to_name(self.func)} is a generator but also returned a value: {exc.value!r}"
                    ) from None
                break

            with ctx.set_current_experiment(experiment):
                experiment.run(*args, **kwargs)

