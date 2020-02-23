from sklearn import datasets, metrics, svm
from sklearn.metrics.classification import (
    accuracy_score,
    precision_recall_fscore_support,
)

from experitur import experiment


@experiment(
    parameter_grid={
        "svc_kernel": ["linear", "poly", "rbf", "sigmoid"],
        "svc_shrinking": [True, False],
    }
)
def classifier_svm(trial):
    digits = datasets.load_digits()

    n_samples = len(digits.images)

    # Flatten
    data = digits.images.reshape((n_samples, -1))

    # Record all defaults of svm.SVC
    trial.record_defaults("svc_", svm.SVC, gamma="scale")

    assert "svc_gamma" in trial
    assert trial["svc_gamma"] == "scale"

    # Create a support vector classifier
    classifier = trial.apply("svc_", svm.SVC)

    print("Classifier:", classifier)

    # Fit
    X_train = data[: n_samples // 2]
    y_train = digits.target[: n_samples // 2]
    classifier.fit(X_train, y_train)

    # Predict
    X_test = data[n_samples // 2 :]
    y_test = digits.target[n_samples // 2 :]
    y_test_pred = classifier.predict(X_test)

    # Calculate some metrics
    macro_prfs = precision_recall_fscore_support(y_test, y_test_pred, average="macro")

    result = dict(zip(("macro_precision", "macro_recall", "macro_f_score"), macro_prfs))

    result["accuracy"] = accuracy_score(y_test, y_test_pred)

    return result
