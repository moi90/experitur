import logging
from typing import Dict, Optional

import pytorch_lightning
import pytorch_lightning.callbacks
from pytorch_lightning.loggers.logger import Logger

from experitur.core.trial import Trial


class PytorchLightningLogger(Logger):
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
        """Experitur does not support hyperparameter logging via a Logger."""
        pass

    @property
    def name(self) -> str:
        """Return the trial ID as the experiment name."""
        return self.trial.id

    @property
    def version(self) -> int:
        """Return the trial version."""
        return 0


class PruningCallback(pytorch_lightning.callbacks.Callback):
    def __init__(self, trial: Trial):
        self.trial = trial
        self.last_global_step_checked = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def on_validation_end(self, trainer: pytorch_lightning.Trainer, pl_module):
        global_step = trainer.global_step

        if (
            trainer.sanity_checking  # don't save anything during sanity check
            or self.last_global_step_checked
            == global_step  # already saved at the last step
        ):
            return

        self.last_global_step_checked = global_step

        try:
            if self.trial.should_prune(
                {"epoch": trainer.current_epoch, "step": trainer.global_step}
            ):
                trainer.should_stop = True
        except Exception:  # pylint: disable=broad-except
            self._logger.exception(
                f"Exception in trial.should_prune in epoch {trainer.current_epoch}."
            )


class SaveTrialCallback(pytorch_lightning.callbacks.Callback):
    """Save trial after each epoch."""

    def __init__(self, trial: Trial):
        self.trial = trial

    def on_train_epoch_end(self, trainer, pl_module):
        self.trial.save()
