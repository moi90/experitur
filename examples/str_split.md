---
-   id: baseline
    run: str_split:run
-   id: extended
    base: baseline
    parameter_grid:
        split_maxsplit: [1, 2]
        split_sep: [' ', ',']
---
# str.split
In this example you can see `set_default_parameters` and `apply_parameters` at work.

```
$ experitur run str_split.md
Running str_split.md...
Running experiment baseline...
Independent parameters: []
parameters (before): {}
parameters (after): {'split_maxsplit': -1, 'split_sep': None}
result: ['An', 'apple,', 'two', 'bananas,', 'three', 'oranges', 'and', 'some', 'grapes.']
Total time: 0.007s
  _: 0.007s (93%)

Running experiment extended...
Independent parameters: ['split_maxsplit', 'split_sep']
    split_maxsplit: 1
    split_sep:  
parameters (before): {'split_maxsplit': 1, 'split_sep': ' '}
parameters (after): {'split_maxsplit': 1, 'split_sep': ' '}
result: ['An', 'apple, two bananas, three oranges and some grapes.']
    split_maxsplit: 1
    split_sep: ,
parameters (before): {'split_maxsplit': 1, 'split_sep': ','}
parameters (after): {'split_maxsplit': 1, 'split_sep': ','}
result: ['An apple', ' two bananas, three oranges and some grapes.']
    split_maxsplit: 2
    split_sep:  
parameters (before): {'split_maxsplit': 2, 'split_sep': ' '}
parameters (after): {'split_maxsplit': 2, 'split_sep': ' '}
result: ['An', 'apple,', 'two bananas, three oranges and some grapes.']
    split_maxsplit: 2
    split_sep: ,
parameters (before): {'split_maxsplit': 2, 'split_sep': ','}
parameters (after): {'split_maxsplit': 2, 'split_sep': ','}
result: ['An apple', ' two bananas', ' three oranges and some grapes.']
Total time: 0.015s
  split_maxsplit-2_split_sep-,: 0.004s (28%)
  split_maxsplit-2_split_sep- : 0.003s (20%)
  split_maxsplit-1_split_sep- : 0.003s (17%)
  split_maxsplit-1_split_sep-,: 0.002s (16%)
```