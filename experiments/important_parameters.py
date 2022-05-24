from experitur.parameters.conditions import Conditions
from typing import Dict, Optional
from experitur.core.parameters import (
    Const,
    Grid,
    ParameterGenerator,
    ParameterGeneratorIter,
)
from experitur.core.trial import Trial
from experitur import Experiment


@Experiment()
def exp1(trial: Trial):
    return dict(trial)


@Experiment()
def exp2(trial: Trial):
    return dict(trial)


class Final(ParameterGenerator):
    def __init__(self, child: ParameterGenerator) -> None:
        self.child = child

    def generate(
        self, experiment: "Experiment", parent: Optional[ParameterGeneratorIter]
    ):
        return self.child.generate(experiment, parent)


train = Experiment()
features = Experiment()
dataset_conditions = {}

train_none = Experiment(parameters=Const(train=False), parent=train)
train_classifier = Experiment(
    parameters=[Conditions("dataset", dataset_conditions, active=["ecotaxa",],),],
    parent=train,
)

training_conditions = {"none": train_none, "classifier": train_classifier, "moco": ...}


@Grid({"dataset": ["uvp", "zooscan"], "training_condition": training_conditions.keys()})
@Experiment()
def best_features(trial: Trial):
    train_: Experiment = trial.choice("training_condition", training_conditions)

    # Execute training
    train_.run()

    # Select best trials
    #
    # What constitutes the "best" trial?
    #   a) Smallest validation metric?
    #       - When using loss, can not compare different losses.
    #       => Silhouette score of validation set?
    #   b) Best metric on a suitable test set?
    #       - What is "suitable"? Labels for end-goal dataset must not be used.
    #
    trials = train_.trials

    if len(trials) > 1:
        trials = trials.best_n()


# Question: PCA or built in dimensionality reduction? Which one is better for transferable features?
# Method: Train different models on dataset A, evaluate on dataset B. Assume a small labeled part of the evaluation dataset.
