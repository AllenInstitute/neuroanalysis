"""
Define a set of scaled unit symbols to make the code more clear

All numerical values are expressed in unscaled units--V, A, s, etc. (except where otherwise specified).
To improve readability, we define a set of symbols that allow physical values to
be written with scaled units. For example, `10*mV` is equivalent to `10e-3`.
"""
for unit in ['m', 's', 'V', 'A', 'Ohm', 'Ω', 'F', 'S', 'Pa', 'Hz']:
    for pfx, val in [('f', -15), ('p', -12), ('n', -9), ('u', -6), ('m', -3), 
                     ('c', -2), ('k', 3), ('M', 6), ('G', 9)]:
        locals()[pfx+unit] = 10**val
