from typing import TYPE_CHECKING

import zerorpc

from experitur.core.trial import TrialData

if TYPE_CHECKING:
    from experitur.core.context import Context


class ExperiturServer(zerorpc.Server):
    def __init__(self, ctx: "Context"):
        super().__init__()
        self.ctx = ctx

    def get_trial_data(self, trial_id):
        return self.ctx.store[trial_id]

    def set_trial_data(self, trial_id, trial_data):
        self.ctx.store[trial_id] = TrialData(self.ctx.store, trial_data)
