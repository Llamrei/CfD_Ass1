from math import expm1
from enum import Enum
from CFDSolver.Tools import solveTDM

import pdb

"""
Future work:
Implement for solving of velocities also
"""


class DifferencingScheme(Enum):
    # Potential future work, writing a parser of differential eqns to turn
    # string-like diff eqn into discretized eqn
    CENTRAL = 1
    UPWIND = 2
    POWER_LAW = 3


class FlowProperties:
    # Object to store scalar values that are assumed constant throughout the domain
    # default to 1 for density and diffusion - but we are required to vary velocity in exercise
    def __init__(
        self, velocity: float, diffusion_coefficient: float = 1, density: float = 1
    ):
        self.density = density
        self.diffusion_coefficient = diffusion_coefficient
        self.velocity = velocity


class OneDimensionalConvectionDiffusionSystem:
    def __init__(self, initial_phi_grid, scalars: FlowProperties, real_length):
        # Store passed system in object and calculate useful quantities
        self.initial_phi_grid = initial_phi_grid
        self.scalars = scalars
        self.real_length = real_length
        self.n = len(initial_phi_grid)
        self._constants = (
            self.scalars.density
            * self.scalars.velocity
            / self.scalars.diffusion_coefficient
        )
        self._delta = self.initial_phi_grid[-1] - self.initial_phi_grid[0]
        self._grid_size = self.real_length / (self.n - 1)

    def _sol(self, i):
        # Calculates analytical answer at node i of self.n - primarily to make code neater
        return (
            self.initial_phi_grid[0]
            + expm1(self._constants * i / (self.n - 1) * self.real_length)
            / expm1(self._constants * self.real_length)
            * self._delta
        )

    def solve_analytically(self):
        # Solve analytically for all nodes
        result = [self._sol(i) for i in range(0, self.n)]
        return result

    def solve_numerically(
        self,
        convection_scheme: DifferencingScheme,
        diffusion_scheme: DifferencingScheme = DifferencingScheme.CENTRAL,
    ):
        """Solve system numerically using a_p\*Phi_p = a_e\*Phi_e + a_w\*Phi_w"""
        normalized_flux = self.scalars.density * self.scalars.velocity
        normalized_diffusion = self.scalars.diffusion_coefficient / self._grid_size
        if diffusion_scheme != DifferencingScheme.CENTRAL:
            # In future could implement other differencing schemes on the diffusion element too
            raise NotImplementedError
        if convection_scheme == DifferencingScheme.CENTRAL:
            a_e = normalized_diffusion - 0.5 * normalized_flux
            a_w = normalized_diffusion + 0.5 * normalized_flux
            # Density and velocity constant so in a_p last term cancels itself
            a_p = a_e + a_w
        elif convection_scheme == DifferencingScheme.UPWIND:
            a_e = normalized_diffusion + max([-1 * normalized_flux, 0])
            a_w = normalized_diffusion + max([1 * normalized_flux, 0])
            # Density and velocity constant so in a_p last term cancels itself
            a_p = a_e + a_w
        elif convection_scheme == DifferencingScheme.POWER_LAW:
            a_e = normalized_diffusion * max(
                [(1 - 0.1 * abs(normalized_flux / normalized_diffusion)) ** 5, 0]
            ) + max([-1 * normalized_flux, 0])
            a_w = normalized_diffusion * max(
                [(1 - 0.1 * abs(normalized_flux / normalized_diffusion)) ** 5, 0]
            ) + max([1 * normalized_flux, 0])
            # Density and velocity constant so in a_p last term cancels itself
            a_p = a_e + a_w
        else:
            raise NotImplementedError
        a = [-1 * a_w] * (self.n - 1)
        b = [a_p] * (self.n)
        c = [-1 * a_e] * (self.n - 1)
        d = [0] * (self.n)
        return solveTDM(a, b, c, self.initial_phi_grid, d)

