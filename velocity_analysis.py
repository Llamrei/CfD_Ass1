import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
fig, (ax1, ax3) = plt.subplots(1, 2)
color = {"CENTRAL": "b", "UPWIND": "orange", "POWER_LAW": "g"}
custom_legend_lines = []
custom_legend_keys = []
error = dict()
time_taken = dict()
for scheme in DifferencingScheme:
    error[scheme.name] = []
    time_taken[scheme.name] = []
    custom_legend_keys = custom_legend_keys + [scheme.name]
    custom_legend_lines = custom_legend_lines + [
        Line2D([0], [0], color=color[scheme.name], lw=4)
    ]
    for vel in systems:
        start = round(time.perf_counter_ns() / 1000)
        solution = systems[vel].solve_numerically(convection_scheme=scheme)
        end = round(time.perf_counter_ns() / 1000)
        ax3.plot(x, solution, color[scheme.name], alpha=abs(vel) / vel_limit)
        error[scheme.name] = error[scheme.name] + [
            calc_error(systems[vel].solve_analytically(), solution)
        ]
        time_taken[scheme.name] = time_taken[scheme.name] + [end - start]
ax3.legend(custom_legend_lines, custom_legend_keys)

# Display results
ax1.set_xlabel("Velocity, m/s")
ax1.set_ylabel("Error, %")
ax2 = ax1.twinx()
ax2.set_ylabel("Time taken, μs")
for scheme in DifferencingScheme:
    ax1.plot(velocity_range, error[scheme.name], label=scheme.name)
    ax2.plot(
        velocity_range, time_taken[scheme.name], "x", label=scheme.name, markersize=6
    )
ax4 = ax1.twiny()
ax4.set_xlabel("Local Peclet, Dimensionless")
ax4.set_xlim(ax1.get_xlim())
new_tick_locations = list(range(-1 * vel_limit, vel_limit + 1, round(vel_limit / 4)))


def tick_function(U):
    Pe = U * density * length / (nodes - 1) / diffusion_coefficient
    return "%.1f" % Pe


ax4.set_xticks(new_tick_locations)
ax4.set_xticklabels(map(tick_function, new_tick_locations))
ax4.xaxis.set_ticks_position(
    "bottom"
)  # set the position of the second x-axis to bottom
ax4.xaxis.set_label_position(
    "bottom"
)  # set the position of the second x-axis to bottom
ax4.spines["bottom"].set_position(("outward", 36))
ax1.legend()
plt.title(
    u"Variation of error with velocity\nN = {},  Γ = {}, ρ = {}".format(
        nodes, scalars[0].diffusion_coefficient, scalars[0].density
    )
)
plt.show()
plt.savefig("velocity_analysis.png", bbox_inches="tight", pad_inches=0.02, dpi=150)

