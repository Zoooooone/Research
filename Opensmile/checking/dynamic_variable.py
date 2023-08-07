import numpy as np

func_names = ['mean', 'sin', 'sqrt']
x = np.ones(3)

for func in func_names:
    exec(f'print(np.{func}(x))')
