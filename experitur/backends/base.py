from abc import ABC, abstractmethod
import itertools


class BaseBackend(ABC):
    @abstractmethod
    def get_trial_by_id(self, trial_id):
        """
        Get the data of a trial by its trial_id.

        Returns a dict with trial data.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_trials_by_parameters(self, parameters):
        """
        Find trials with matching parameters.

        "Matching" means that parameters.keys()
        is a subset of the matching parameters_(pre|post).keys()
        and that the values are the same for all keys.

        Returns a dict of dicts with trial data.
        """
        raise NotImplementedError()

    @abstractmethod
    def trials(self):
        raise NotImplementedError()

    def _is_match(self, parameters_1, parameters_2):
        """
        Decide whether parameters_1 are a subset of parameters_2.
        """

        if set(parameters_1.keys()) <= set(parameters_2.keys()):
            return all(v == parameters_2[k] for k, v in parameters_1.items())

        return False

    def format_independent_parameters(self, trial_parameters, independent_parameters):
        if len(independent_parameters) > 0:
            trial_id = "_".join("{}-{!s}".format(k, trial_parameters[k])
                                for k in independent_parameters)
            trial_id = trial_id.replace("/", "_")
        else:
            trial_id = "_"

        return trial_id

    def make_trial_id(self, trial_parameters, independent_parameters):
        trial_id = self.format_independent_parameters(
            trial_parameters, independent_parameters)

        try:
            existing_trial = self.get_trial_by_id(trial_id)
        except KeyError:
            # If there is no existing trial with this id, it is unique
            return trial_id

        # Otherwise, we have to incorporate more independent parameters
        new_independent_parameters = []

        existing_trial.setdefault("parameters_post", {})

        # Look for parameters in existing_trial that have differing values
        for name, value in existing_trial["parameters_post"].items():
            if name in trial_parameters and trial_parameters[name] != value:
                new_independent_parameters.append(name)

        # Look for parameters that did not exist previously
        for name in trial_parameters.keys():
            if name not in existing_trial["parameters_post"]:
                new_independent_parameters.append(name)

        if new_independent_parameters:
            # If we found parameters where this trial is different from the existing one, append these to independent
            independent_parameters.extend(new_independent_parameters)
            return self.make_trial_id(trial_parameters, independent_parameters)

        # Otherwise, we just append a version number
        for i in itertools.count():
            test_trial_id = "{}.{}".format(trial_id, i)

            try:
                existing_trial = self.get_trial_by_id(test_trial_id)
            except KeyError:
                # If there is no existing trial with this id, it is unique
                return test_trial_id
