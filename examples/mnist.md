---
-   id: svmbaseline
    run: mnist:run
-   id: svmgrid
    base: svmbaseline
    parameter_grid:
        svc_kernel: ['linear', 'poly', 'rbf', 'sigmoid']
        svc_shrinking: [True, False]
---