---
-   id: svmbaseline
    run: mnist:run
-   id: svmgrid
    base: svmbaseline
    parameter_grid:
        svc_kernel: ['linear', 'poly', 'rbf', 'sigmoid']
        svc_shrinking: [True, False]
---
# MNIST example

TODO: Do multiple experiments with different classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html