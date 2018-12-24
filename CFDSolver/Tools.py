import pdb


class IncompatibleListException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def calc_error(real: list, calculated: list) -> float:
    if len(real) != len(calculated):
        raise IncompatibleListException
    cum_tot = 0
    for i in range(0, len(real) - 1):
        cum_tot = cum_tot + abs((calculated[i] - real[i]) / real[i])
    return 100 / len(real) * cum_tot


def solveTDM(a, b, c, x, d):
    """ Use Tri-diagonal matrix algorithm to solve Ax=d
    
    Given lists containing coefficients of A: a_i, b_i, c_i.
    Where b_i is main diagonal, c_i is above, a_i is below
    we use solveTDM to solve for vector x 
    
    Will treat any non-zero values in x as fixed
    Takes only list inputs"""
    if len(a) != len(c) or len(a) + 1 != len(b) or len(d) != len(b):
        raise AttributeError("Not a Tri-diagonal matrix")
    copy_x = x  # Don't want to change original grid
    p = [0] * len(d)
    q = [0] * len(d)
    a = [0, *a]  # Needed to make a_i access correct element
    c = [*c, 0]

    # Get modified coefficients
    for (i, el) in enumerate(copy_x):
        if i == 0:
            if el != 0:
                p[0] = 0
                q[0] = el
            else:
                p[0] = c[0] / b[0]
                q[0] = d[0] / b[0]
        elif i == len(copy_x) - 1:
            if el != 0:
                copy_x[i] = el
            else:
                copy_x[i] = (d[i] - a[i] * q[i - 1]) / (b[i] - a[i] * p[i - 1])
        else:
            p[i] = c[i] / (b[i] - a[i] * p[i - 1])
            q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] - a[i] * p[i - 1])

    # Back substitute for answer
    for i in range(len(copy_x) - 1, 0, -1):
        copy_x[i - 1] = q[i - 1] - p[i - 1] * copy_x[i]

    return copy_x
