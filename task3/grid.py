import numpy as np

from mpi4py import MPI
from config import Config


class Grid:
    def __init__(
        self,
        cfg: Config,
        mpi: MPI.Intracomm,
    ):
        self.mpi = mpi
        self.cfg = cfg

        self.M_i = cfg.M // mpi.Get_size()
        self.offset = mpi.Get_rank() * self.M_i

        self.grid = np.empty((cfg.K, self.M_i), dtype=np.float64)

        self._applyBoundaryCond()
        self._applyInitCond()

    def __getitem__(self, idx: tuple[int, int]):
        if idx[1] == -1:
            return self.mpi.recv(source=self.mpi.Get_rank() - 1)

        return self.grid[idx]

    def __setitem__(self, idx: tuple[int, int], val: np.float64):
        if self.mpi.Get_rank() != self.mpi.Get_size() - 1:
            if idx[1] == self.M_i - 1:
                self.mpi.send(val, self.mpi.Get_rank() + 1)

        self.grid[idx] = val

    def _applyInitCond(self):
        for m in range(self.M_i):
            self.grid[0, m] = self.cfg.init_cond(self.offset, m)

    def _applyBoundaryCond(self):
        for k in range(self.cfg.K):
            self.grid[k, 0] = self.cfg.boundary_cond(k)
