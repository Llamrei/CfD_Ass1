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
velocity_range = range(-20, 20)
scalars = dict()
for vel in velocity_range:
    scalars[vel] = FlowProperties(velocity=vel)
nodes = 10

# Declare and instantiate spacial and phi domain
x = [x / (nodes - 1) * length for x in range(0, nodes)]
phi_grid = [0] * len(x)
phi_grid[0] = 100
phi_grid[-1] = 20

# Create solvers for each velocity - corresponding to a given peclet
systems = dict()
for vel in scalars:
    systems[vel] = CDS(phi_grid, scalars[vel], length)

# Solve
error = dict()
time_taken = dict()
for scheme in DifferencingScheme:
    error[scheme.name] = []
    time_taken[scheme.name] = []
    color = {"CENTRAL": "b", "UPWIND": "orange", "POWER_LAW": "g"}
    for vel in systems:
        start = round(time.perf_counter_ns() / 1000)
        solution = systems[vel].solve_numerically(convection_scheme=scheme)
        end = round(time.perf_counter_ns() / 1000)
        plt.plot(x, solution, color[scheme.name], alpha=abs(vel) / 20)
        error[scheme.name] = error[scheme.name] + [
            calc_error(systems[vel].solve_analytically(), solution)
        ]
        time_taken[scheme.name] = time_taken[scheme.name] + [end - start]

# Display results
# fig, ax1 = plt.subplots()
# ax1.set_xlabel("Velocity, m/s")
# ax1.set_ylabel("Error, %")
# ax2 = ax1.twinx()
# ax2.set_ylabel("Time taken, μs")
# for scheme in DifferencingScheme:
#     ax1.plot(velocity_range, error[scheme.name], label=scheme.name)
#     ax2.plot(
#         velocity_range, time_taken[scheme.name], "x", label=scheme.name, markersize=6
#     )
plt.legend()
plt.title(
    u"Variation of solution with velocity\nN = {},  Γ = {}, ρ = {}".format(
        nodes, scalars[0].diffusion_coefficient, scalars[0].density
    )
)
plt.show()

