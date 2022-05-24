from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experitur import Experiment, Trial
from experitur.configurators import Grid


@Grid({"svc_kernel": ["linear", "poly", "rbf", "sigmoid"]})
@Experiment()
def classifier_svm(trial: Trial):
    X, y = datasets.load_digits(return_X_y=True)

    n_samples = len(X)

    # Flatten
    X = X.reshape((n_samples, -1))

    # Extract parameters prefixed with "svc_"
    svc_parameters = trial.prefixed("svc_")

    # Create a support vector classifier
    classifier = svc_parameters.call(svm.SVC)

    # svc_parameters.call automatically filled `parameters` in with the default values:
    assert "svc_gamma" in trial
    assert trial["svc_gamma"] == "scale"

    print("Classifier:", classifier)

    # Fit
    X_train = X[: n_samples // 2]
    y_train = y[: n_samples // 2]
    classifier.fit(X_train, y_train)

    # Predict
    X_test = X[n_samples // 2 :]
    y_test = y[n_samples // 2 :]
    y_test_pred = classifier.predict(X_test)

    # Calculate some metrics
    macro_prfs = precision_recall_fscore_support(y_test, y_test_pred, average="macro")

    result = dict(zip(("macro_precision", "macro_recall", "macro_f_score"), macro_prfs))

    result["accuracy"] = accuracy_score(y_test, y_test_pred)

    print(result)

    return result
