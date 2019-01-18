import matplotlib.pyplot as plt
import time
import timeit
from scipy.interpolate import make_interp_spline, BSpline

from pdb import set_trace

from CFDSolver.ConvectionDiffusionSolver import (
    OneDimensionalConvectionDiffusionSystem as CDS,
)
from CFDSolver.ConvectionDiffusionSolver import FlowProperties, DifferencingScheme
from CFDSolver.Tools import calc_error

# Define system parameters - in SI units where applicable
length = 1
density = 1.23
diffusion_coefficient = 0.5918
nodes = 200
local_peclet_limit = 4
vel_limit = round(
    local_peclet_limit / (length / (nodes - 1)) / density * diffusion_coefficient
)
velocity_range = range(-1 * vel_limit, vel_limit + 1)
scalars = dict()
for vel in velocity_range:
    scalars[vel] = FlowProperties(
        velocity=vel, density=density, diffusion_coefficient=diffusion_coefficient
    )

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
solution = dict()
time_taken = dict()
for scheme in DifferencingScheme:
    error[scheme.name] = []
    time_taken[scheme.name] = []
    for vel in systems:
        start = round(time.perf_counter_ns() / 1000)
        solution[vel] = systems[vel].solve_numerically(convection_scheme=scheme)
        end = round(time.perf_counter_ns() / 1000)
        error[scheme.name] = error[scheme.name] + [
            calc_error(systems[vel].solve_analytically(), solution[vel])
        ]
        time_taken[scheme.name] = time_taken[scheme.name] + [end - start]

# Display results
fig, ax1 = plt.subplots()
ax1.set_xlabel("Velocity, m/s")
ax1.set_ylabel("Error, %")
ax2 = ax1.twinx()
ax2.set_ylabel("Time taken, μs")
for scheme in DifferencingScheme:
    ax1.plot(velocity_range, error[scheme.name], label=scheme.name, linewidth=2.0)
    ax2.plot(velocity_range, time_taken[scheme.name], alpha=0.5)
axPe = ax1.twiny()
axPe.set_xlabel("Local Peclet, Dimensionless")
axPe.set_xlim(ax1.get_xlim())
new_tick_locations = [
    -1 * vel_limit,
    round(-1 * vel_limit / 2),
    0,
    round(vel_limit / 2),
    vel_limit,
]


def tick_function(U):
    Pe = U * density * length / (nodes - 1) / diffusion_coefficient
    return "%.1f" % Pe


axPe.set_xticks(new_tick_locations)
axPe.set_xticklabels(map(tick_function, new_tick_locations))
axPe.xaxis.set_ticks_position(
    "bottom"
)  # set the position of the second x-axis to bottom
axPe.xaxis.set_label_position(
    "bottom"
)  # set the position of the second x-axis to bottom
axPe.spines["bottom"].set_position(("outward", 36))
ax1.legend()
plt.title(
    u"N = {},  Γ = {}, ρ = {}".format(
        nodes, scalars[0].diffusion_coefficient, scalars[0].density
    )
)
plt.tight_layout()
plt.show()

