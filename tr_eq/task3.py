from mpi4py import MPI

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
        x_start = 0
        if self.rank == 0:
            x_start = 1

        for t in range(self.cfg.K - 1):
            for x in range(x_start, self.grid.M_i):
                self._rectangle(t, x)

        self.grid.waitSendReqs()

        b = np.ascontiguousarray(self.grid.npGrid.transpose())
        if self.rank == 0:
            res = np.empty((self.cfg.M, self.cfg.K), dtype=np.float64)
            self.mpi.Gather(b, res, root=0)

            DrawSolution(res.transpose(), self.cfg)
            return

        self.mpi.Gather(b, None, root=0)

    def _implicitLeftCorner(self, t: int, x: int):
        self.grid[t + 1, x] = (
            self.grid[t, x]
            + self.cfg.courant * self.grid[t + 1, x - 1]
            + self.cfg.tau * self.cfg.f_x_t(self.grid.offset, x, t)
        ) / (1 + self.cfg.courant)

    def _rectangle(self, t: int, x: int):
        self.grid[t + 1, x] = (
            (self.grid[t, x] - self.grid[t + 1, x - 1]) * (1 - self.cfg.courant)
            + 2 * self.cfg.tau * self.cfg.f_x_t(self.grid.offset, x, t)
        ) / (1 + self.cfg.courant) + self.grid[t, x - 1]


Solver().Solve()
