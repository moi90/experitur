from pytorch_lightning.loggers.base import LightningLoggerBase
from experitur.core.trial import Trial
from typing import Dict, Optional
import warnings


class PytorchLightningLogger(LightningLoggerBase):
    def __init__(self, trial: Trial):
        super().__init__()

        self.trial = trial

    @property
    def experiment(self) -> Trial:
        """Return the trial object associated with this logger."""
        return self.trial

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Records metrics.
        This method logs metrics as as soon as it received them. If you want to aggregate
        metrics for one specific `step`, use the
        :meth:`~pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics` method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """

        self.trial.log(metrics, step=step)

    def log_hyperparams(self, *_):
        warnings.warn("Experitur does not support hyperparameter logging via a Logger.")

    @property
    def name(self) -> str:
        """Return the trial id."""
        return self.trial.id

    @property
    def version(self) -> int:
        """Return the trial version."""
        return 0
