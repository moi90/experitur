# Hyper-parameter Optimization in Experitur

- Framework agnostic
- 

## Configuration

```yaml
-	id: <experiment_id>
	# Regular parameters
	parameter_grid:
		a: [1,2,3]
		b: [4,5,6]
		c: [7,8,9]
		
	optimize:
		suggest: "<module>:<function>"
		
		# `suggest` is called with the results of prior runs,
		# these settings and the current grid cell as parameters:
		# suggest(results, settings, parameters)
		settings:
			init_points: 2
    		n_iter: 3
    		# ...
    		
    		
    	# Experitur needs to known about the optimized parameters so that it removes
    	# them from the grid
        parameters:
            # Parameter `a` is not included in the grid anymore but optimized
            a: ["int", 0, 10]
            a: ["uniform", 0, 10]
            a: ["loguniform", 0, 10]
```

