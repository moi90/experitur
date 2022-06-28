import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experitur.core.trial import Trial


class TrialLogHandler(logging.Handler):
    """
    Hook into Python's logging to save entries in the trial log.
    
    Example:
        with TrialLogHandler(trial):
            ...
    """

    def __init__(self, trial: "Trial"):
        super().__init__()
        self.trial = trial

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            self.trial.log_msg(msg)
        except RecursionError:
            raise
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)

    def __enter__(self):
        logging.root.addHandler(self)

    def __exit__(self, *_, **__):
        logging.root.removeHandler(self)
