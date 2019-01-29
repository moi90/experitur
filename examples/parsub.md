---
id: parsub
run: "echo:run"
parameter_grid:
    a_1: [foo]
    a_2: [bar]
    a: ["{a_{b}}"]
    b: [1,2]
---
