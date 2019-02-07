---
-   id: baseline
    run: mnist:run
-   id: gridsearch
    base: baseline
    parameter_grid:
        svc_kernel: ['linear', 'poly', 'rbf', 'sigmoid']
        svc_shrinking: [True, False]
---