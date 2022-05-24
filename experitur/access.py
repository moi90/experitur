from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experitur.core.trial import Trial


def get_trial(trial_id) -> "Trial":
    from experitur.core.context import get_current_context

    ctx = get_current_context()
    return ctx.get_trial(trial_id)


def get_current_trial() -> "Trial":
    from experitur.core.context import get_current_context

    ctx = get_current_context()
    trial = ctx.current_trial

    if trial is None:
        raise ValueError("No current trial")

    return trial
