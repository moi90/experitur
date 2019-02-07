from sklearn import datasets, metrics, svm
from sklearn.metrics.classification import (accuracy_score,
                                            precision_recall_fscore_support)

from experitur.util import apply_parameters, set_default_parameters
from inspect import signature


def run(working_directory, parameters):
    digits = datasets.load_digits()

    n_samples = len(digits.images)

    # Flatten
    data = digits.images.reshape((n_samples, -1))

    # Set default parameters
    set_default_parameters(
        "svc_",
        parameters,
        svm.SVC,
        gamma="scale"
    )

    assert "svc_gamma" in parameters
    assert parameters["svc_gamma"] == "scale"

    # for param in signature(svm.SVC).parameters.values():
    #     print(param, param.kind)

    # Create a support vector classifier
    classifier = apply_parameters(
        "svc_",
        parameters,
        svm.SVC,
    )

    print("Classifier:", classifier)

    # Fit
    X_train = data[:n_samples // 2]
    y_train = digits.target[:n_samples // 2]
    classifier.fit(X_train, y_train)

    # Predict
    X_test = data[n_samples // 2:]
    y_test = digits.target[n_samples // 2:]
    y_test_pred = classifier.predict(X_test)

    # Calculate some metrics
    macro_prfs = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro")

    result = dict(zip(
        ("macro_precision", "macro_recall", "macro_f_score"),
        macro_prfs))

    result["accuracy"] = accuracy_score(y_test, y_test_pred)

    return result
