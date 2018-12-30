import matplotlib.pyplot as plt
import time
from scipy.interpolate import make_interp_spline, BSpline

from CFDSolver.ConvectionDiffusionSolver import (
    OneDimensionalConvectionDiffusionSystem as CDS,
)
from CFDSolver.ConvectionDiffusionSolver import FlowProperties, DifferencingScheme
from CFDSolver.Tools import calc_error

# Define system parameters - in SI units where applicable
length = 1
nodes = 5
scalars = FlowProperties(velocity=1)

# Override system parameters if provided in command line
# In positional order: # of nodes, velocity, diffusion and density
if __name__ == "__main__":
    import sys

    try:
        if 3 <= len(sys.argv) and len(sys.argv) <= 5:
            nodes = int(sys.argv[1])
            scalars = FlowProperties(*map(float, sys.argv[2:]))
        else:
            raise IndexError
    except IndexError:
        print("Inappropriate # of arguments passed - using code defaults")

# Declare and instantiate spacial and phi domain
x = [x / (nodes-1) * length for x in range(0, nodes)]
phi_grid = [0] * len(x)
phi_grid[0] = 100
phi_grid[-1] = 20

# Create solver
system = CDS(phi_grid, scalars, length)

# Solve
an_sol_achievable = False
try:
    an_sol = system.solve_analytically()
    an_sol_error = round(calc_error(an_sol, an_sol), 3)
    # Smooth out analytical solution for plotting
    x_smooth = [x / 300 * length for x in range(0, 300)]
    spl = make_interp_spline(x, an_sol, k=3)
    an_sol_smooth = spl(x_smooth)
    plt.plot(x, an_sol, "bo")
    plt.plot(
        x_smooth, an_sol_smooth, label="Analytical(Splined) - {}%".format(an_sol_error)
    )
    an_sol_achievable = True
except OverflowError:
    print("Cannot determine analytical solution as exponent too large")
sol = dict()  # Dict of (solution,error,processing time)
for scheme in DifferencingScheme:
    start = round(time.perf_counter_ns() / 1000)
    solution = system.solve_numerically(convection_scheme=scheme)
    end = round(time.perf_counter_ns() / 1000)
    if an_sol_achievable:
        sol[scheme.name] = (
            solution,
            round(calc_error(an_sol, solution), 3),
            end - start,
        )
    else:
        sol[scheme.name] = (solution, "N/A ", end - start)

# Display results
for scheme in DifferencingScheme:
    plt.plot(
        x,
        sol[scheme.name][0],
        "o--",
        label="{} {}us - {}% error".format(
            scheme.name.capitalize(), sol[scheme.name][2], sol[scheme.name][1]
        ),
    )
plt.legend(loc="best")
plt.title(
    u"U = {}, ∆X = {} \n Γ = {}, ρ = {}, N = {} \n Global Pe = {}, Local Pe = {}\n{}".format(
        scalars.velocity,
        length / nodes,
        scalars.diffusion_coefficient,
        scalars.density,
        len(phi_grid),
        scalars.density * scalars.velocity * length / scalars.diffusion_coefficient,
        scalars.density
        * scalars.velocity
        * length
        / nodes
        / scalars.diffusion_coefficient,
        "" if system.cd_stable else "Unstable for CD",
    )
)
plt.xlabel("X\nm")
plt.ylabel("Φ(X)\nUnits")
plt.show()
