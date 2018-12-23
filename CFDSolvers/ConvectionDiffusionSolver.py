import numpy as np
from enum import Enum
import pdb


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
        self.n = len(initial_phi_grid)  # Index nodes from 0
        self._constants = (
            self.scalars.density
            * self.scalars.velocity
            / self.scalars.diffusion_coefficient
        )
        self._delta = self.initial_phi_grid[-1] - self.initial_phi_grid[0]

    def _sol(self, i):
        # Calculates analytical answer at node i of self.n - primarily to make code neater
        return (
            self.initial_phi_grid[0]
            + np.expm1(self._constants * i / self.n * self.real_length)
            / np.expm1(self._constants * self.real_length)
            * self._delta
        )

    def solve_analytically(self):
        result = [self._sol(i) for i in range(0, self.n)]
        return result

    def solve_numerically(
        self,
        convection_scheme: DifferencingScheme,
        diffusion_scheme: DifferencingScheme = DifferencingScheme.CENTRAL,
    ):
        if diffusion_scheme != DifferencingScheme.CENTRAL:
            # In future could implement other differencing schemes on the diffusion element too
            raise NotImplementedError

        if convection_scheme == DifferencingScheme.CENTRAL:
            pass
        elif convection_scheme == DifferencingScheme.UPWIND:
            pass
        elif convection_scheme == DifferencingScheme.POWER_LAW:
            pass
        else:
            raise NameError("Invalid discretisation scheme selected for convection")
