from experitur.core.context import get_current_context


def get_trial(trial_id):
    ctx = get_current_context()
    return ctx.get_trial(trial_id)
