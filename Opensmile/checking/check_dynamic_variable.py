import numpy as np

x = np.ones(3)
func_names = ['sin', 'cos', 'tan']

for func in func_names:
    exec(f"print(np.{func}(x))")

for letter in ['A', 'B']:
    exec(f"x_{letter} = 1")
