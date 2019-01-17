import matplotlib.pyplot as plt
import time
from scipy.interpolate import make_interp_spline, BSpline

from pdb import set_trace

from CFDSolver.ConvectionDiffusionSolver import (
    OneDimensionalConvectionDiffusionSystem as CDS,
)
from CFDSolver.ConvectionDiffusionSolver import FlowProperties, DifferencingScheme
from CFDSolver.Tools import calc_error

# Define system parameters - in SI units where applicable
length = 1
scalars = FlowProperties(velocity=1)
max_Nodes = 2000

# Declare and instantiate spacial and phi domain
node_range = range(3, max_Nodes)
phi_grid = dict()
x = phi_grid
for nodes in node_range:
    x[nodes] = [x / (nodes - 1) * length for x in range(0, nodes)]
    phi_grid[nodes] = [0] * len(x[nodes])
    phi_grid[nodes][0] = 100
    phi_grid[nodes][-1] = 20

# Create solvers for each # of nodes
systems = dict()
for nodes in phi_grid:
    # set_trace()
    systems[nodes] = CDS(phi_grid[nodes], scalars, length)

# Solve
error = dict()
time_taken = dict()
for scheme in DifferencingScheme:
    error[scheme.name] = []
    time_taken[scheme.name] = []
    for nodes in systems:
        start = round(time.perf_counter_ns() / 1000)
        solution = systems[nodes].solve_numerically(convection_scheme=scheme)
        end = round(time.perf_counter_ns() / 1000)
        # set_trace()
        error[scheme.name] = error[scheme.name] + [
            calc_error(systems[nodes].solve_analytically(), solution)
        ]
        time_taken[scheme.name] = time_taken[scheme.name] + [end - start]

# Display results
fig, ax1 = plt.subplots()
ax1.set_xlabel("Number of Nodes, N")
ax1.set_ylabel("Error, %")
ax2 = ax1.twinx()
ax2.set_ylabel("Time taken, μs")
for scheme in DifferencingScheme:
    ax1.plot(node_range, error[scheme.name], label=scheme.name)
    ax2.plot(node_range, time_taken[scheme.name], 'x',label=scheme.name, markersize = 4)
plt.legend(loc="best")
plt.title(
    u"Variation of error with number of nodes\nU = {},  Γ = {}, ρ = {}".format(
        scalars.velocity, scalars.diffusion_coefficient, scalars.density
    )
)
plt.show()

