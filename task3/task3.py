from mpi4py import MPI

# import time
import numpy as np

from config import Config
from grid import Grid

from draw import DrawSolution


class Solver:
    cfg = Config()

    def __init__(self):
        self.mpi = MPI.COMM_WORLD
        self.rank = self.mpi.Get_rank()

        self.grid = Grid(self.cfg, self.mpi)

    def Solve(self):
        if self.rank == 0:
            x_start = 1
        else:
            x_start = 0

        for t in range(self.cfg.K - 1):
            for x in range(x_start, self.grid.M_i):
                self._implicitLeftCorner(t, x)

        if self.rank == 0:
            res = np.empty((self.cfg.M, self.cfg.K), dtype=np.float64)
            self.mpi.Gather(self.grid.grid, res, root=0)

            DrawSolution(res, self.cfg)
            # print(res)
            return

        self.mpi.Gather(self.grid.grid, None, root=0)

    def _implicitLeftCorner(self, t: int, x: int):
        self.grid[t + 1, x] = (
            self.grid[t, x]
            + self.cfg.courant * self.grid[t + 1, x - 1]
            + self.cfg.tau * self.cfg.f_x_t(self.grid.offset, x, t)
        ) / (1 + self.cfg.courant)


Solver().Solve()
