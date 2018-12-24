import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from scipy.interpolate import make_interp_spline, BSpline

from CFDSolver.ConvectionDiffusionSolver import (
    OneDimensionalConvectionDiffusionSystem as CDS,
)
from CFDSolver.ConvectionDiffusionSolver import FlowProperties, DifferencingScheme
from CFDSolver.Tools import calc_error

import pdb

### Create system
length = 1  # in m
grid_size = 0.1  # in m
scalars = FlowProperties(
    velocity=1
)  # flow properties just specifying velocity, others default to 1

# Override if provided in command line
if __name__ == "__main__":
    """
    Provide cmd line args for grid size, velocity, diffusion and density
    """
    import sys

    try:
        if 3 <= len(sys.argv) and len(sys.argv) <= 5:
            grid_size = float(sys.argv[1])
            scalars = FlowProperties(
                *map(float, sys.argv[2:])
            )  # flow properties just specifying velocity, others default to 1
        else:
            raise IndexError
    except IndexError:
        print("Invalid parameters - using script defined values")

# Declare and instantiate spacial and phi domain
x = np.arange(0, length + grid_size, grid_size)
phi_grid = np.zeros_like(x)
phi_grid[0] = 100
phi_grid[-1] = 20

# Create solver
system = CDS(phi_grid, scalars, length)

### Solve
an_sol = system.solve_analytically()
an_sol_error = round(calc_error(an_sol, an_sol), 3)
cd_sol = system.solve_numerically(convection_scheme=DifferencingScheme.CENTRAL)
cd_sol_error = round(calc_error(an_sol, cd_sol), 3)
uw_sol = system.solve_numerically(convection_scheme=DifferencingScheme.UPWIND)
uw_sol_error = round(calc_error(an_sol, uw_sol), 3)
pl_sol = system.solve_numerically(convection_scheme=DifferencingScheme.POWER_LAW)
pl_sol_error = round(calc_error(an_sol, pl_sol), 3)
pdb.set_trace()


### Plot results for given parameters

# Smooth out analytical solution for plotting
x_smooth = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, an_sol, k=3)
an_sol_smooth = spl(x_smooth)

plt.plot(x_smooth, an_sol_smooth, label="Analytical - Err {}%".format(an_sol_error))
# plt.plot(x, cd_sol, "bo", label="Central Differencing - Err {}%".format(cd_sol_error))
plt.plot(x, uw_sol, "ro", label="Upwind Differencing - Err {}%".format(uw_sol_error))
plt.plot(x, pl_sol, "go", label="Power-law Differencing - Err {}%".format(pl_sol_error))

plt.legend(loc="best")
plt.title(
    u"U = {}, ∆X = {} \n Γ = {}, ρ = {}, N = {} \n Global Pe = {}, Local Pe = {}".format(
        scalars.velocity,
        grid_size,
        scalars.diffusion_coefficient,
        scalars.density,
        len(phi_grid),
        scalars.density * scalars.velocity * length / scalars.diffusion_coefficient,
        scalars.density * scalars.velocity * grid_size / scalars.diffusion_coefficient,
    )
)
plt.xlabel("X\nm")
plt.ylabel("Φ(X)\nUnits")
plt.show()
